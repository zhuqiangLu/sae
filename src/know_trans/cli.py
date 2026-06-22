"""know_trans command line interface.

A single ``argparse`` entry point that drives the full SAE-based cross-family
distillation pipeline. Each subcommand loads a ``--config`` YAML into a
:class:`know_trans.config.Config` and then calls the relevant module functions
with exactly the signatures defined in ``docs/INTERFACE_SPEC.md``.

Subcommands (run in this order)::

    capture          # 01  dump teacher + student MLP activations on a corpus
    train-sae        # 02  train a TopK SAE per (model, layer)
    build-concepts   # 03  build the labeled concept battery from benchmarks
    score            # 04  score every SAE feature as a concept detector (AUC)
    match            # 05  label-anchored teacher<->student concept matching
    distill          # 06  train the student with the concept-distillation loss

Usage::

    python -m know_trans.cli capture       --config configs/pair_llama8b_qwen0p6b.yaml
    python -m know_trans.cli train-sae     --config configs/pair_llama8b_qwen0p6b.yaml
    python -m know_trans.cli build-concepts --config configs/pair_llama8b_qwen0p6b.yaml
    python -m know_trans.cli score         --config configs/pair_llama8b_qwen0p6b.yaml
    python -m know_trans.cli match         --config configs/pair_llama8b_qwen0p6b.yaml
    python -m know_trans.cli distill       --config configs/pair_llama8b_qwen0p6b.yaml

Design notes
------------
* **Nothing heavy happens at import time.** ``torch``/``transformers`` and the
  sibling modules are imported lazily inside each subcommand handler so that
  ``python -m know_trans.cli --help`` (and unit tests) stay fast and never
  trigger model loads or downloads.
* **Dims are never hardcoded.** ``d_in`` is inferred from the activation shards
  by the SAE trainer; the per-model layer list is resolved from the model's own
  ``config.json`` when ``ModelCfg.layers == "all"`` (or any non-list value).
* **Layer correspondence is discovered downstream** (in ``score`` / ``match``),
  never assumed equal between families.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Only ``config`` and ``utils`` are safe to import at module load time: they are
# lightweight (no model loads). Everything else is imported lazily per command.
# ---------------------------------------------------------------------------
from know_trans import config as cfg_mod
from know_trans.config import Config, ModelCfg
from know_trans import utils


# ===========================================================================
# Helpers shared by the subcommand handlers
# ===========================================================================
def _resolve_layers(model_cfg: ModelCfg) -> list[int]:
    """Resolve a :class:`ModelCfg`'s ``layers`` field to concrete indices.

    ``ModelCfg.layers`` may be an explicit ``list[int]`` or the sentinel string
    ``"all"`` (the default). When it is not already a list, we infer the layer
    count from the model's ``config.json`` (``num_hidden_layers``) and expand to
    ``range(num_hidden_layers)``. Dims/depths are therefore never hardcoded.

    Parameters
    ----------
    model_cfg:
        The teacher or student model configuration.

    Returns
    -------
    list[int]
        Sorted, de-duplicated list of layer indices to operate on.

    Raises
    ------
    FileNotFoundError
        If ``layers == "all"`` but the model's ``config.json`` cannot be found.
    KeyError
        If the model's ``config.json`` lacks ``num_hidden_layers``.
    """
    layers = model_cfg.layers
    if isinstance(layers, (list, tuple)):
        return sorted({int(layer) for layer in layers})

    # Anything that is not an explicit list (e.g. the string "all") => expand
    # from the model's own config.json so we never hardcode the depth.
    config_path = os.path.join(model_cfg.path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"layers={layers!r} for model {model_cfg.name!r} requires inferring "
            f"the layer count, but no config.json was found at {config_path!r}. "
            f"Either point ModelCfg.path at a valid HF model dir or set "
            f"ModelCfg.layers to an explicit list of ints."
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        hf_config: dict[str, Any] = json.load(handle)
    if "num_hidden_layers" not in hf_config:
        raise KeyError(
            f"{config_path!r} has no 'num_hidden_layers' field; cannot expand "
            f"layers={layers!r}. Set ModelCfg.layers to an explicit list."
        )
    n_layers = int(hf_config["num_hidden_layers"])
    return list(range(n_layers))


def _activations_dir(config: Config, model_cfg: ModelCfg) -> str:
    """Directory holding one model's captured activations: ``<data>/activations/<name>``."""
    return os.path.join(config.paths.data, "activations", model_cfg.name)


def _saes_dir(config: Config, model_cfg: ModelCfg) -> str:
    """Directory holding one model's trained SAEs: ``<data>/saes/<name>``."""
    return os.path.join(config.paths.data, "saes", model_cfg.name)


def _concepts_dir(config: Config) -> str:
    """Directory holding the concept battery: ``<data>/concepts``."""
    return os.path.join(config.paths.data, "concepts")


def _feature_scores_path(config: Config, model_cfg: ModelCfg) -> str:
    """Parquet path for one model's per-feature concept scores."""
    return os.path.join(
        config.paths.data, "feature_scores", f"{model_cfg.name}.parquet"
    )


def _matches_path(config: Config) -> str:
    """JSON path for the teacher<->student concept matches."""
    return os.path.join(config.paths.data, "matches", "matches.json")


def _iter_models(config: Config, only: str | None) -> list[tuple[str, ModelCfg]]:
    """Yield the (role, ModelCfg) pairs selected by a ``--model`` filter.

    Parameters
    ----------
    config:
        Loaded run configuration.
    only:
        ``"teacher"``, ``"student"``, or ``None``/``"both"`` for both models.

    Returns
    -------
    list[tuple[str, ModelCfg]]
        Selected ``(role, model_cfg)`` pairs.

    Raises
    ------
    ValueError
        If ``only`` is not a recognised value.
    """
    if only in (None, "both"):
        return [("teacher", config.teacher), ("student", config.student)]
    if only == "teacher":
        return [("teacher", config.teacher)]
    if only == "student":
        return [("student", config.student)]
    raise ValueError(
        f"--model must be one of teacher|student|both, got {only!r}"
    )


def _load_corpus_texts(
    config: Config, n_texts: int | None, logger: Any
) -> list[str]:
    """Load the capture corpus as a flat ``list[str]``.

    The corpus is selected by ``config.capture.corpus``:

    * ``"benchmarks"`` (default) — harvest natural-language text from the real
      benchmark files under ``config.paths.benchmarks`` (AdvBench behaviors,
      HarmBench behaviors, and Global-MMLU-Lite questions). These give a broad,
      on-distribution corpus that exercises the safety / topic / language
      concepts the battery will later probe.
    * a path to a ``.txt`` file — one text per line.
    * a path to a ``.jsonl`` file — one JSON object per line; the first of
      ``{"text", "prompt", "question", "Behavior"}`` present is used.

    Parameters
    ----------
    config:
        Loaded run configuration.
    n_texts:
        Cap on the number of texts (``None`` => ``config.capture.n_texts``).
    logger:
        Logger for progress messages.

    Returns
    -------
    list[str]
        Up to ``n_texts`` non-empty corpus strings.
    """
    cap = config.capture
    limit = int(cap.n_texts if n_texts is None else n_texts)
    corpus = cap.corpus

    texts: list[str]
    if corpus == "benchmarks":
        texts = _harvest_benchmark_texts(config.paths.benchmarks, limit, logger)
    elif corpus.endswith(".txt") and os.path.isfile(corpus):
        with open(corpus, "r", encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]
    elif corpus.endswith(".jsonl") and os.path.isfile(corpus):
        texts = []
        with open(corpus, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("text", "prompt", "question", "Behavior"):
                    if key in obj and obj[key]:
                        texts.append(str(obj[key]))
                        break
    else:
        raise FileNotFoundError(
            f"capture.corpus={corpus!r} is neither the literal 'benchmarks' nor "
            f"an existing .txt/.jsonl file."
        )

    texts = [t for t in texts if t and t.strip()][:limit]
    logger.info("Loaded %d corpus texts (corpus=%r).", len(texts), corpus)
    if not texts:
        raise RuntimeError(
            f"No corpus texts loaded for corpus={corpus!r}; check the path / "
            f"benchmarks dir {config.paths.benchmarks!r}."
        )
    return texts


def _harvest_benchmark_texts(
    benchmarks_dir: str, limit: int, logger: Any
) -> list[str]:
    """Harvest a mixed corpus of natural-language text from the benchmark files.

    Reads, in round-robin-ish order: AdvBench behaviors, HarmBench behaviors,
    and Global-MMLU-Lite questions. Parquet reads are done lazily via pandas so
    this stays cheap when the caller only needs a small ``limit``.

    Parameters
    ----------
    benchmarks_dir:
        Root directory containing ``AdvBench/``, ``HarmBench/``,
        ``Global-MMLU-Lite/`` etc.
    limit:
        Maximum number of texts to return.
    logger:
        Logger for progress / warnings.

    Returns
    -------
    list[str]
        Harvested corpus texts (best-effort; missing files are skipped).
    """
    import glob

    texts: list[str] = []

    # --- AdvBench behaviors -------------------------------------------------
    advbench = os.path.join(benchmarks_dir, "AdvBench", "advbench_behaviors.jsonl")
    if os.path.isfile(advbench):
        try:
            # utf-8-sig strips the BOM that ships with advbench_behaviors.jsonl.
            with open(advbench, "r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    val = obj.get("Behavior") or obj.get("behavior")
                    if val:
                        texts.append(str(val))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            logger.warning("Could not read AdvBench: %s", exc)
    else:
        logger.warning("AdvBench file not found at %s; skipping.", advbench)

    # --- HarmBench behaviors ------------------------------------------------
    harmbench = os.path.join(benchmarks_dir, "HarmBench", "metadata.csv")
    if os.path.isfile(harmbench):
        try:
            import pandas as pd  # local import: heavy-ish

            df = pd.read_csv(harmbench)
            if "Behavior" in df.columns:
                texts.extend(str(v) for v in df["Behavior"].dropna().tolist())
        except Exception as exc:  # pragma: no cover  (pandas/csv quirks)
            logger.warning("Could not read HarmBench metadata: %s", exc)
    else:
        logger.warning("HarmBench metadata not found at %s; skipping.", harmbench)

    # --- Global-MMLU-Lite questions ----------------------------------------
    mmlu_root = os.path.join(benchmarks_dir, "Global-MMLU-Lite")
    if os.path.isdir(mmlu_root):
        parquets = sorted(glob.glob(os.path.join(mmlu_root, "**", "*.parquet"),
                                    recursive=True))
        if parquets:
            try:
                import pandas as pd  # local import

                for pq in parquets:
                    if len(texts) >= limit * 3:  # plenty harvested already
                        break
                    df = pd.read_parquet(pq)
                    col = next(
                        (c for c in ("question", "Question", "text")
                         if c in df.columns),
                        None,
                    )
                    if col is not None:
                        texts.extend(str(v) for v in df[col].dropna().tolist())
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not read Global-MMLU-Lite parquets: %s", exc)
        else:
            logger.warning("No parquet files under %s; skipping.", mmlu_root)
    else:
        logger.warning("Global-MMLU-Lite dir not found at %s; skipping.", mmlu_root)

    return texts


# ===========================================================================
# Subcommand handlers
# ===========================================================================
def cmd_capture(args: argparse.Namespace) -> None:
    """Capture per-layer MLP activations for the selected model(s).

    Calls :func:`know_trans.capture.capture_activations` once per model with the
    exact contract signature::

        capture_activations(model, tokenizer, texts, layers, out_dir,
                            batch_size=16, max_len=512, hook_point="mlp",
                            dtype="float16") -> None

    Writes ``<data>/activations/<model_name>/{layer{L}.safetensors,
    index.parquet, meta.json}``.
    """
    config = cfg_mod.load_config(args.config)
    logger = utils.get_logger("know_trans.cli.capture")
    utils.set_seed(config.seed)

    # Lazy heavy imports (model loading + capture machinery).
    from know_trans.capture import capture_activations

    cap = config.capture
    batch_size = args.batch_size if args.batch_size is not None else cap.batch_size
    max_len = args.max_len if args.max_len is not None else cap.max_len
    hook_point = args.hook_point if args.hook_point is not None else cap.hook_point
    dtype = args.dtype  # capture stores float16 by default per contract

    texts = _load_corpus_texts(config, args.n_texts, logger)

    for role, model_cfg in _iter_models(config, args.model):
        layers = args.layers if args.layers is not None else _resolve_layers(model_cfg)
        out_dir = utils.ensure_dir(_activations_dir(config, model_cfg))
        logger.info(
            "[%s] capturing model=%s layers=%s -> %s",
            role, model_cfg.name, layers, out_dir,
        )
        model, tokenizer = utils.load_model_and_tokenizer(
            model_cfg.path, dtype=model_cfg.dtype, device=args.device
        )
        try:
            capture_activations(
                model,
                tokenizer,
                texts,
                layers,
                out_dir,
                batch_size=batch_size,
                max_len=max_len,
                hook_point=hook_point,
                dtype=dtype,
            )
        finally:
            # Free GPU memory between the (large) teacher and the student.
            _free_model(model)
        logger.info("[%s] capture complete -> %s", role, out_dir)


def cmd_train_sae(args: argparse.Namespace) -> None:
    """Train one TopK SAE per (selected model, selected layer).

    Calls :func:`know_trans.sae.train_sae` with the contract signature::

        train_sae(reader: ActivationReader, layer, cfg: SAECfg, seed, out_path)
            -> TopKSAE

    ``d_in`` is inferred inside ``train_sae`` from the activation shards;
    ``d_hidden = expansion * d_in``. SAEs are written to
    ``<data>/saes/<model_name>/layer{L}.safetensors`` (or whatever filename the
    ``sae`` module chooses for that ``out_path`` stem).
    """
    config = cfg_mod.load_config(args.config)
    logger = utils.get_logger("know_trans.cli.train-sae")

    from know_trans.capture import ActivationReader
    from know_trans.sae import train_sae

    sae_cfg = config.sae
    if args.steps is not None:
        sae_cfg = _replace_cfg(sae_cfg, steps=args.steps)

    for role, model_cfg in _iter_models(config, args.model):
        act_dir = _activations_dir(config, model_cfg)
        if not os.path.isdir(act_dir):
            raise FileNotFoundError(
                f"No captured activations for {model_cfg.name!r} at {act_dir!r}; "
                f"run `cli capture` first."
            )
        reader = ActivationReader(act_dir)
        # Restrict to layers that were actually captured.
        captured = set(int(layer) for layer in reader.layers)
        requested = (
            set(args.layers) if args.layers is not None
            else set(_resolve_layers(model_cfg))
        )
        layers = sorted(captured & requested) or sorted(captured)
        out_root = utils.ensure_dir(_saes_dir(config, model_cfg))

        for layer in layers:
            out_path = os.path.join(out_root, f"layer{layer}.safetensors")
            logger.info(
                "[%s] training SAE model=%s layer=%d (expansion=%d, k=%d, "
                "steps=%d) -> %s",
                role, model_cfg.name, layer, sae_cfg.expansion, sae_cfg.k,
                sae_cfg.steps, out_path,
            )
            train_sae(
                reader,
                layer,
                sae_cfg,
                config.seed,
                out_path,
            )
        logger.info("[%s] SAE training complete -> %s", role, out_root)


def cmd_build_concepts(args: argparse.Namespace) -> None:
    """Build the labeled concept battery from the real benchmark files.

    Calls :func:`know_trans.concepts.build_battery_from_benchmarks`::

        build_battery_from_benchmarks(benchmarks_dir, out_dir,
                                      n_per_concept=200) -> list[Concept]

    Writes one jsonl per concept under ``<data>/concepts/``.
    """
    config = cfg_mod.load_config(args.config)
    logger = utils.get_logger("know_trans.cli.build-concepts")

    from know_trans.concepts import build_battery_from_benchmarks

    benchmarks_dir = args.benchmarks_dir or config.paths.benchmarks
    out_dir = utils.ensure_dir(args.out_dir or _concepts_dir(config))
    logger.info(
        "Building concept battery from %s -> %s (n_per_concept=%d)",
        benchmarks_dir, out_dir, args.n_per_concept,
    )
    battery = build_battery_from_benchmarks(
        benchmarks_dir,
        out_dir,
        n_per_concept=args.n_per_concept,
    )
    logger.info("Built %d concepts -> %s", len(battery), out_dir)


def cmd_score(args: argparse.Namespace) -> None:
    """Score every SAE feature as a concept detector (ROC-AUC).

    Calls :func:`know_trans.score.score_features` once per selected model::

        score_features(model, tokenizer, sae_bundle, battery, layers,
                       out_path, max_len=512, batch_size=16) -> DataFrame

    Loads each model's SAEs via :meth:`SAEBundle.load` and the battery via
    :func:`load_battery`, then writes ``<data>/feature_scores/<model_name>.parquet``.
    """
    config = cfg_mod.load_config(args.config)
    logger = utils.get_logger("know_trans.cli.score")

    from know_trans.sae import SAEBundle
    from know_trans.score import score_features
    from know_trans.concepts import load_battery

    concepts_dir = args.concepts_dir or _concepts_dir(config)
    if not os.path.isdir(concepts_dir):
        raise FileNotFoundError(
            f"Concept battery dir {concepts_dir!r} not found; run "
            f"`cli build-concepts` first."
        )
    battery = load_battery(concepts_dir)
    logger.info("Loaded battery of %d concepts from %s", len(battery), concepts_dir)

    max_len = args.max_len if args.max_len is not None else config.capture.max_len
    batch_size = (
        args.batch_size if args.batch_size is not None else config.capture.batch_size
    )

    for role, model_cfg in _iter_models(config, args.model):
        sae_dir = _saes_dir(config, model_cfg)
        if not os.path.isdir(sae_dir):
            raise FileNotFoundError(
                f"No SAEs for {model_cfg.name!r} at {sae_dir!r}; run "
                f"`cli train-sae` first."
            )
        sae_bundle = SAEBundle.load(sae_dir)
        layers = (
            args.layers if args.layers is not None
            else sorted(int(layer) for layer in sae_bundle.layers)
        )
        out_path = _feature_scores_path(config, model_cfg)
        utils.ensure_dir(os.path.dirname(out_path))
        logger.info(
            "[%s] scoring features model=%s layers=%s -> %s",
            role, model_cfg.name, layers, out_path,
        )
        model, tokenizer = utils.load_model_and_tokenizer(
            model_cfg.path, dtype=model_cfg.dtype, device=args.device
        )
        try:
            score_features(
                model,
                tokenizer,
                sae_bundle,
                battery,
                layers,
                out_path,
                max_len=max_len,
                batch_size=batch_size,
            )
        finally:
            _free_model(model)
        logger.info("[%s] scoring complete -> %s", role, out_path)


def cmd_match(args: argparse.Namespace) -> None:
    """Run label-anchored teacher<->student concept matching.

    Builds per-model concept->feature-set dicts via
    :func:`know_trans.score.concept_feature_sets`, then::

        run_matching(teacher_sets, student_sets) -> dict
        write_matches(matches, path) -> None

    Writes ``<data>/matches/matches.json``. Teacher-only concepts (``shared:
    false``) are logged separately as the headline finding.
    """
    config = cfg_mod.load_config(args.config)
    logger = utils.get_logger("know_trans.cli.match")

    import pandas as pd  # parquet read
    from know_trans.score import concept_feature_sets
    from know_trans.match import run_matching, write_matches

    teacher_scores = _feature_scores_path(config, config.teacher)
    student_scores = _feature_scores_path(config, config.student)
    for path, role in ((teacher_scores, "teacher"), (student_scores, "student")):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No {role} feature scores at {path!r}; run `cli score` first."
            )

    teacher_df = pd.read_parquet(teacher_scores)
    student_df = pd.read_parquet(student_scores)
    teacher_sets = concept_feature_sets(
        teacher_df, auc_threshold=args.auc_threshold, top_k=args.top_k
    )
    student_sets = concept_feature_sets(
        student_df, auc_threshold=args.auc_threshold, top_k=args.top_k
    )
    logger.info(
        "Concept feature sets: teacher=%d concepts, student=%d concepts "
        "(auc>=%.2f, top_k=%d).",
        len(teacher_sets), len(student_sets), args.auc_threshold, args.top_k,
    )

    matches = run_matching(teacher_sets, student_sets)

    # Surface the headline finding: teacher-only concepts.
    shared, teacher_only = [], []
    for concept, info in matches.items():
        if isinstance(info, dict) and info.get("shared"):
            shared.append(concept)
        else:
            teacher_only.append(concept)
    logger.info(
        "Matching: %d shared concept(s); %d teacher-only concept(s) "
        "(headline finding).",
        len(shared), len(teacher_only),
    )
    if teacher_only:
        logger.info("Teacher-only concepts: %s", ", ".join(sorted(teacher_only)))

    out_path = args.out or _matches_path(config)
    utils.ensure_dir(os.path.dirname(out_path))
    write_matches(matches, out_path)
    logger.info("Wrote matches -> %s", out_path)


def cmd_distill(args: argparse.Namespace) -> None:
    """Train the student with the concept-distillation loss.

    Thin wrapper over :func:`know_trans.distill.train_distill`::

        train_distill(cfg: Config, matches_path: str) -> None
    """
    config = cfg_mod.load_config(args.config)
    logger = utils.get_logger("know_trans.cli.distill")
    utils.set_seed(config.seed)

    from know_trans.distill import train_distill

    matches_path = args.matches or _matches_path(config)
    if not os.path.isfile(matches_path):
        raise FileNotFoundError(
            f"Matches file {matches_path!r} not found; run `cli match` first."
        )
    logger.info(
        "Distilling student=%s from teacher=%s using matches=%s "
        "(lambda_concept=%.3f, steps=%d, lr=%g, pool=%s).",
        config.student.name, config.teacher.name, matches_path,
        config.distill.lambda_concept, config.distill.steps,
        config.distill.lr, config.distill.pool,
    )
    train_distill(config, matches_path)
    logger.info("Distillation complete.")


# ===========================================================================
# Small utilities used by the handlers (kept here to avoid hard deps)
# ===========================================================================
def _replace_cfg(cfg_obj: Any, **changes: Any) -> Any:
    """Return a copy of a dataclass config with ``changes`` applied.

    Uses :func:`dataclasses.replace` when ``cfg_obj`` is a dataclass; otherwise
    mutates a shallow copy. Lets CLI flags (e.g. ``--steps``) override config
    values without mutating the loaded :class:`Config`.
    """
    import copy
    import dataclasses

    if dataclasses.is_dataclass(cfg_obj) and not isinstance(cfg_obj, type):
        return dataclasses.replace(cfg_obj, **changes)
    clone = copy.copy(cfg_obj)
    for key, value in changes.items():
        setattr(clone, key, value)
    return clone


def _free_model(model: Any) -> None:
    """Best-effort release of a model and its GPU memory between stages."""
    try:
        import gc
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover  (cleanup must never crash a run)
        pass


# ===========================================================================
# Argument parser
# ===========================================================================
def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--config`` argument to a subparser."""
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the run config YAML (e.g. configs/pair_llama8b_qwen0p6b.yaml).",
    )


def _add_model_filter(parser: argparse.ArgumentParser) -> None:
    """Attach a ``--model {teacher,student,both}`` filter to a subparser."""
    parser.add_argument(
        "--model",
        choices=["teacher", "student", "both"],
        default="both",
        help="Which model(s) to operate on (default: both).",
    )


def _int_list(value: str) -> list[int]:
    """argparse type: parse a comma-separated list of ints, e.g. ``"4,8,12"``."""
    value = value.strip()
    if not value:
        return []
    try:
        return [int(part) for part in value.split(",") if part.strip() != ""]
    except ValueError as exc:  # pragma: no cover
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of ints, got {value!r}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all six subcommands.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser. Each subparser sets ``func`` to its handler.
    """
    parser = argparse.ArgumentParser(
        prog="know_trans",
        description=(
            "SAE-based cross-family knowledge distillation pipeline. "
            "Run the subcommands in order: capture -> train-sae -> "
            "build-concepts -> score -> match -> distill."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="{capture,train-sae,"
                                "build-concepts,score,match,distill}")
    sub.required = True

    # --- capture ------------------------------------------------------------
    p_capture = sub.add_parser(
        "capture",
        help="Capture per-layer MLP activations for teacher/student.",
        description=cmd_capture.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(p_capture)
    _add_model_filter(p_capture)
    p_capture.add_argument("--layers", type=_int_list, default=None,
                           help="Comma-separated layer indices (default: from "
                                "ModelCfg.layers, expanding 'all').")
    p_capture.add_argument("--n-texts", dest="n_texts", type=int, default=None,
                           help="Cap on corpus texts (default: capture.n_texts).")
    p_capture.add_argument("--batch-size", dest="batch_size", type=int,
                           default=None,
                           help="Forward batch size (default: capture.batch_size).")
    p_capture.add_argument("--max-len", dest="max_len", type=int, default=None,
                           help="Max tokens per text (default: capture.max_len).")
    p_capture.add_argument("--hook-point", dest="hook_point", default=None,
                           help="Hook point (default: capture.hook_point).")
    p_capture.add_argument("--dtype", default="float16",
                           help="Stored activation dtype (contract: float16).")
    p_capture.add_argument("--device", default=None,
                           help="Device override (default: auto via utils).")
    p_capture.set_defaults(func=cmd_capture)

    # --- train-sae ----------------------------------------------------------
    p_sae = sub.add_parser(
        "train-sae",
        help="Train a TopK SAE per (model, layer).",
        description=cmd_train_sae.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(p_sae)
    _add_model_filter(p_sae)
    p_sae.add_argument("--layers", type=_int_list, default=None,
                       help="Comma-separated layers (default: all captured "
                            "layers intersected with ModelCfg.layers).")
    p_sae.add_argument("--steps", type=int, default=None,
                       help="Override SAECfg.steps for a quick run.")
    p_sae.set_defaults(func=cmd_train_sae)

    # --- build-concepts -----------------------------------------------------
    p_concepts = sub.add_parser(
        "build-concepts",
        help="Build the labeled concept battery from benchmarks.",
        description=cmd_build_concepts.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(p_concepts)
    p_concepts.add_argument("--benchmarks-dir", dest="benchmarks_dir",
                            default=None,
                            help="Benchmarks root (default: paths.benchmarks).")
    p_concepts.add_argument("--out-dir", dest="out_dir", default=None,
                            help="Battery output dir (default: <data>/concepts).")
    p_concepts.add_argument("--n-per-concept", dest="n_per_concept", type=int,
                            default=200,
                            help="Max examples per concept side.")
    p_concepts.set_defaults(func=cmd_build_concepts)

    # --- score --------------------------------------------------------------
    p_score = sub.add_parser(
        "score",
        help="Score every SAE feature as a concept detector (AUC).",
        description=cmd_score.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(p_score)
    _add_model_filter(p_score)
    p_score.add_argument("--layers", type=_int_list, default=None,
                         help="Comma-separated layers (default: all SAE layers).")
    p_score.add_argument("--concepts-dir", dest="concepts_dir", default=None,
                         help="Battery dir (default: <data>/concepts).")
    p_score.add_argument("--max-len", dest="max_len", type=int, default=None,
                         help="Max tokens per example (default: capture.max_len).")
    p_score.add_argument("--batch-size", dest="batch_size", type=int,
                         default=None,
                         help="Forward batch size (default: capture.batch_size).")
    p_score.add_argument("--device", default=None,
                         help="Device override (default: auto via utils).")
    p_score.set_defaults(func=cmd_score)

    # --- match --------------------------------------------------------------
    p_match = sub.add_parser(
        "match",
        help="Label-anchored teacher<->student concept matching.",
        description=cmd_match.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(p_match)
    p_match.add_argument("--auc-threshold", dest="auc_threshold", type=float,
                         default=0.8,
                         help="Min AUC for a feature to enter a concept set.")
    p_match.add_argument("--top-k", dest="top_k", type=int, default=10,
                         help="Max features per concept set.")
    p_match.add_argument("--out", default=None,
                         help="Matches JSON output (default: <data>/matches/"
                              "matches.json).")
    p_match.set_defaults(func=cmd_match)

    # --- distill ------------------------------------------------------------
    p_distill = sub.add_parser(
        "distill",
        help="Train the student with the concept-distillation loss.",
        description=cmd_distill.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(p_distill)
    p_distill.add_argument("--matches", default=None,
                           help="Matches JSON (default: <data>/matches/"
                                "matches.json).")
    p_distill.set_defaults(func=cmd_distill)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument vector (excluding the program name). Defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 on success, non-zero on handled error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("Interrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        # Expected, user-facing errors: print cleanly without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
