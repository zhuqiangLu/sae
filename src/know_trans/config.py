"""Configuration dataclasses + YAML loader for know_trans.

A run is fully described by a :class:`Config`, which nests one dataclass per
concern (models, SAE, capture, distillation, paths). Dimensions are *never*
hardcoded here — ``d_model`` is inferred from activation shards at runtime and
``d_hidden = expansion * d_in`` (see ``sae.py``). The defaults below mirror
``docs/INTERFACE_SPEC.md`` exactly.

Example
-------
>>> from know_trans.config import load_config
>>> cfg = load_config("configs/pair_llama8b_qwen0p6b.yaml")
>>> cfg.teacher.family, cfg.student.family
('llama', 'qwen')
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

try:  # PyYAML is the only hard dependency of this module.
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "config.load_config requires PyYAML. Install it with `pip install pyyaml`."
    ) from exc


# --------------------------------------------------------------------------- #
# Per-concern dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ModelCfg:
    """A single model (teacher or student).

    Parameters
    ----------
    name:
        Human-readable model name, e.g. ``"Llama-3.1-8B"``.
    path:
        Filesystem path to the HuggingFace checkpoint directory.
    family:
        Model family tag (``"llama"``, ``"qwen"``, ...). Used to reason about
        cross-family transfer and the family-distance ablation.
    layers:
        Which decoder layers to capture / train SAEs on. Either an explicit
        list of integer indices, or the string ``"all"`` (resolved against the
        model's ``num_hidden_layers`` at capture time).
    dtype:
        Compute dtype for the model forward pass.
    """

    name: str
    path: str
    family: str
    layers: list[int] | str = "all"
    dtype: str = "bfloat16"


@dataclass
class SAECfg:
    """Hyper-parameters for the OpenAI-style TopK SAE.

    Dictionary size: if ``d_hidden`` is set (absolute), it is used for *every*
    model so teacher and student share an identical-size dictionary (controls for
    SAE capacity in cross-model comparison). If ``d_hidden`` is ``None`` it falls
    back to ``expansion * d_in`` (``d_in`` inferred from the activation shards).
    """

    expansion: int = 16
    d_hidden: int | None = None   # absolute dict size for ALL models; overrides expansion*d_in
    k: int = 32
    lr: float = 4e-4
    batch_size: int = 4096
    steps: int = 50000
    aux_k: int = 256
    aux_coef: float = 1 / 32
    activation: str = "topk"


@dataclass
class CaptureCfg:
    """Activation-capture settings.

    ``hook_point`` selects the sub-module whose output is recorded (default the
    MLP output of each decoder layer).
    """

    corpus: str = "benchmarks"
    n_texts: int = 20000
    max_len: int = 512
    batch_size: int = 16
    hook_point: str = "mlp"


@dataclass
class DistillCfg:
    """Distillation settings for the concept-activation-space objective.

    Parameters
    ----------
    lambda_concept:
        Weight on the concept-alignment loss relative to the student LM loss.
    freeze_saes:
        If ``True`` both SAEs are frozen and gradients flow into the student
        only (the default; see open decision #8 in the design narrative).
    lr, steps:
        Student fine-tuning schedule.
    pool:
        Example-pooling mode (``"mean"`` / ``"max"`` / ``"last"``), kept in sync
        with :func:`know_trans.utils.pool_examples`.
    """

    lambda_concept: float = 1.0
    freeze_saes: bool = True
    lr: float = 1e-5
    steps: int = 2000
    pool: str = "mean"


@dataclass
class Paths:
    """Top-level filesystem anchors.

    Parameters
    ----------
    models:
        Root directory holding model checkpoints.
    benchmarks:
        Root directory holding the benchmark corpora (AdvBench, HarmBench,
        Global-MMLU-Lite, ...).
    data:
        Root directory for all pipeline outputs (activations, SAEs, concepts,
        feature scores, matches). Gitignored.
    """

    models: str
    benchmarks: str
    data: str


@dataclass
class Config:
    """The full configuration for a single teacher/student run."""

    teacher: ModelCfg
    student: ModelCfg
    sae: SAECfg
    capture: CaptureCfg
    distill: DistillCfg
    paths: Paths
    seed: int = 0


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Instantiate dataclass ``cls`` from ``data``, ignoring unknown keys.

    Unknown keys are dropped (with the dataclass defaults filling any gaps) so
    that configs can carry forward-compatible comments / extra fields without
    breaking older code.

    Parameters
    ----------
    cls:
        A dataclass type.
    data:
        Mapping of field name -> value (``None`` is treated as an empty dict).

    Returns
    -------
    Any
        An instance of ``cls``.
    """
    data = dict(data or {})
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs = {key: value for key, value in data.items() if key in valid}
    return cls(**kwargs)


def load_config(path: str) -> Config:
    """Parse a YAML file into a :class:`Config`.

    The YAML is expected to have top-level keys ``teacher``, ``student``,
    ``sae``, ``capture``, ``distill``, ``paths`` and optionally ``seed``. Any
    section omitted falls back to the dataclass defaults; unknown keys are
    ignored.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Fully populated configuration object.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    return Config(
        teacher=_build(ModelCfg, raw.get("teacher")),
        student=_build(ModelCfg, raw.get("student")),
        sae=_build(SAECfg, raw.get("sae")),
        capture=_build(CaptureCfg, raw.get("capture")),
        distill=_build(DistillCfg, raw.get("distill")),
        paths=_build(Paths, raw.get("paths")),
        seed=int(raw.get("seed", 0)),
    )
