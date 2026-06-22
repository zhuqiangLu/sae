"""Cross-family concept matching for SAE-based knowledge distillation.

This module is the *correspondence* stage of ``know_trans``. Given per-model
concept->feature-set tables (produced by :func:`know_trans.score.concept_feature_sets`),
it decides which teacher concepts have a usable student counterpart and produces a
``matches`` mapping that :mod:`know_trans.distill` consumes to build the shared
concept-activation space.

The scientific premise being tested is that a *stronger teacher* and a *weaker
student of a different family* can be aligned in interpretable **concept space**
(SAE features) rather than raw-activation space. Because the tokenizers differ
across families, alignment is **never** by token index; concepts are anchored by
their *human label* (the concept name) and everything downstream is pooled to the
example/span level.

Design notes / gates (each public function documents its own gate):

* ``run_matching``   -- label-anchored matching; teacher-only concepts are kept
  but flagged ``shared: false`` (the headline negative finding).
* ``stability_score``-- Jaccard agreement of feature sets across SAE training
  seeds. A concept whose feature set is unstable across seeds is not a reliable
  carrier of knowledge.
* ``null_control``   -- shuffled-pair gap. Establishes that real (same-label)
  matches carry more signal than random (mismatched-label) pairings.
* ``causal_steer_test`` -- a *real, runnable* interventional gate: ablate / steer
  the matched feature via the SAE decoder direction at the hook point and measure
  a behavioral delta (next-token log-prob shift) on prompts. This is the strongest
  evidence that a matched feature is causal rather than merely correlational.
* ``write_matches``  -- persist the matches mapping as JSON.

The data contract for a "concept set" (one element of a model's
``concept -> [feature, ...]`` mapping) is::

    {"layer": int, "feature": int, "auc": float}

as returned by :func:`know_trans.score.concept_feature_sets`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

# ``ConceptSets`` is the public shape produced by ``score.concept_feature_sets``:
#   {concept_name: [{"layer": int, "feature": int, "auc": float}, ...]}
ConceptSets = Mapping[str, Sequence[Mapping[str, Any]]]

__all__ = [
    "run_matching",
    "stability_score",
    "null_control",
    "causal_steer_test",
    "write_matches",
]


# --------------------------------------------------------------------------- #
# Internal helpers (not part of the public API)                               #
# --------------------------------------------------------------------------- #
def _feature_keys(feature_set: Sequence[Mapping[str, Any]]) -> set[tuple[int, int]]:
    """Return the set of ``(layer, feature)`` identity tuples for a feature set.

    Used for Jaccard / overlap computations where a "feature" is identified by
    its ``(layer, feature_index)`` pair (feature splitting means a concept maps
    to a *set* of features, possibly across layers).

    Args:
        feature_set: a list of ``{"layer", "feature", "auc"}`` dicts.

    Returns:
        Set of ``(layer, feature)`` tuples.
    """
    keys: set[tuple[int, int]] = set()
    for f in feature_set:
        keys.add((int(f["layer"]), int(f["feature"])))
    return keys


def _best_layer(feature_set: Sequence[Mapping[str, Any]]) -> int:
    """Discover the single best layer for a concept's feature set.

    Layer correspondence is *discovered*, never hardcoded: the best layer is the
    one carrying the highest-AUC feature for this concept. The matched feature
    list returned by :func:`run_matching` is then restricted to that layer so the
    distiller pools features that live in the same representation.

    Args:
        feature_set: a non-empty list of ``{"layer", "feature", "auc"}`` dicts.

    Returns:
        The integer layer index of the highest-AUC feature.

    Raises:
        ValueError: if ``feature_set`` is empty.
    """
    if not feature_set:
        raise ValueError("cannot pick a best layer from an empty feature set")
    best = max(feature_set, key=lambda f: float(f["auc"]))
    return int(best["layer"])


def _layer_features(
    feature_set: Sequence[Mapping[str, Any]], layer: int
) -> list[int]:
    """Return the feature indices belonging to ``layer``, sorted by AUC desc.

    Args:
        feature_set: list of ``{"layer", "feature", "auc"}`` dicts.
        layer: the layer to filter on.

    Returns:
        Feature indices at ``layer``, ordered by descending AUC.
    """
    rows = [f for f in feature_set if int(f["layer"]) == int(layer)]
    rows.sort(key=lambda f: float(f["auc"]), reverse=True)
    return [int(f["feature"]) for f in rows]


def _mean_auc(feature_set: Sequence[Mapping[str, Any]], layer: int | None = None) -> float:
    """Mean AUC over a feature set (optionally restricted to one layer).

    Args:
        feature_set: list of ``{"layer", "feature", "auc"}`` dicts.
        layer: if given, average only features at this layer.

    Returns:
        Mean AUC, or ``0.0`` for an empty selection.
    """
    rows = feature_set if layer is None else [f for f in feature_set if int(f["layer"]) == int(layer)]
    if not rows:
        return 0.0
    return float(sum(float(f["auc"]) for f in rows) / len(rows))


def _jaccard(a: set[Any], b: set[Any]) -> float:
    """Jaccard similarity ``|a & b| / |a | b|``; empty/empty -> 1.0.

    Two concepts that both map to *no* features are treated as trivially
    identical (similarity 1.0), so an absent concept never spuriously lowers a
    stability score.

    Args:
        a: first set.
        b: second set.

    Returns:
        Jaccard index in ``[0, 1]``.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_matching(teacher_sets: ConceptSets, student_sets: ConceptSets) -> dict:
    """Label-anchor teacher and student concept feature sets into a match map.

    **Gate (label-anchored correspondence).** Because the two models come from
    different families with different tokenizers, we cannot align by raw geometry
    or token index. Instead we anchor on the *human concept label*: a concept is
    considered transferable iff it has a feature set on **both** sides for the
    same label. The best layer is discovered per side (highest-AUC feature), and
    only that layer's features enter the matched set so the distiller pools
    comparable representations.

    Teacher concepts with **no** student counterpart are *not* dropped: they are
    recorded with ``shared: false``. These teacher-only concepts are the headline
    negative finding (knowledge the teacher has that the weaker student appears
    unable to represent), and callers should log/report them separately.

    Args:
        teacher_sets: ``{concept: [{"layer", "feature", "auc"}, ...]}`` for the
            teacher (e.g. output of ``concept_feature_sets`` on the teacher SAEs).
        student_sets: same shape for the student.

    Returns:
        A dict keyed by concept name. Shared concepts look like::

            {concept: {
                "teacher": {"layer": int, "features": [int, ...], "mean_auc": float},
                "student": {"layer": int, "features": [int, ...], "mean_auc": float},
                "shared": True,
            }}

        Teacher-only concepts look like::

            {concept: {
                "teacher": {"layer": int, "features": [int, ...], "mean_auc": float},
                "student": None,
                "shared": False,
            }}

        Student-only concepts (no teacher signal) are *not* included: with no
        teacher target there is nothing to distill.

    Note:
        The top-level ``matches`` object passed to ``distill`` / ``write_matches``
        is exactly this returned dict.
    """
    matches: dict[str, dict] = {}
    teacher_only: list[str] = []

    for concept, t_set in teacher_sets.items():
        if not t_set:
            # Teacher label present but carried no features above threshold.
            continue

        t_layer = _best_layer(t_set)
        t_entry = {
            "layer": t_layer,
            "features": _layer_features(t_set, t_layer),
            "mean_auc": _mean_auc(t_set, t_layer),
        }

        s_set = student_sets.get(concept)
        if s_set:
            s_layer = _best_layer(s_set)
            s_entry = {
                "layer": s_layer,
                "features": _layer_features(s_set, s_layer),
                "mean_auc": _mean_auc(s_set, s_layer),
            }
            matches[concept] = {
                "teacher": t_entry,
                "student": s_entry,
                "shared": True,
            }
        else:
            matches[concept] = {
                "teacher": t_entry,
                "student": None,
                "shared": False,
            }
            teacher_only.append(concept)

    # Surface the headline finding in the payload so reporting can log it
    # separately without recomputing.
    n_shared = sum(1 for v in matches.values() if v["shared"])
    matches["__summary__"] = {
        "n_concepts": len(matches),
        "n_shared": n_shared,
        "n_teacher_only": len(teacher_only),
        "teacher_only": sorted(teacher_only),
    }
    return matches


def stability_score(setsA: ConceptSets, setsB: ConceptSets) -> dict[str, float]:
    """Per-concept Jaccard stability of feature sets across two SAE seeds.

    **Gate (seed stability).** A concept is only a trustworthy carrier of
    knowledge if the SAE re-discovers (roughly) the same feature set when trained
    with a different random seed. ``setsA`` and ``setsB`` are the
    ``concept_feature_sets`` outputs for the *same model* under two SAE training
    seeds; the score is the Jaccard overlap of the ``(layer, feature)`` identities
    each concept maps to.

    Low stability means the matched feature is seed-dependent noise and should be
    down-weighted (or excluded) before it is trusted for distillation.

    Args:
        setsA: concept feature sets from SAE seed A.
        setsB: concept feature sets from SAE seed B.

    Returns:
        ``{concept: jaccard_in_[0,1]}`` for the union of concept names, plus a
        ``"__mean__"`` key with the macro-average over concepts (excluding itself).
    """
    concepts = set(setsA.keys()) | set(setsB.keys())
    out: dict[str, float] = {}
    for c in sorted(concepts):
        a = _feature_keys(setsA.get(c, []))
        b = _feature_keys(setsB.get(c, []))
        out[c] = _jaccard(a, b)
    if out:
        out["__mean__"] = float(sum(out.values()) / len(out))
    else:
        out["__mean__"] = 0.0
    return out


def null_control(
    teacher_sets: ConceptSets,
    student_sets: ConceptSets,
    n_shuffle: int = 100,
    seed: int = 0,
) -> dict:
    """Shuffled-pair null control quantifying real-vs-random match signal.

    **Gate (null control / shuffled-pair gap).** Label-anchored matching could
    look impressive simply because *any* two concept feature sets share some
    structure. To rule this out, we compare a *real* match score (teacher concept
    aligned to the **same-label** student concept) against a *null* distribution
    built by repeatedly aligning teacher concepts to **randomly permuted** student
    concepts.

    The per-pair "match score" must be *sensitive to which student concept is
    paired with which teacher concept* (a permutation-invariant score such as the
    mean of independent strengths would give a guaranteed-zero gap and prove
    nothing). Feature indices are not comparable across families, so we instead
    score a pair by the **concordance of encoding strength**: ``min(t, s)`` where
    ``t`` and ``s`` are the teacher- and student-side best-layer mean AUCs,
    centered at the chance level 0.5 so that a feature at chance contributes
    nothing, i.e. ``score = min(t - 0.5, s - 0.5)`` clamped at 0. A true
    same-label pair where *both* sides strongly encode the concept scores high;
    pairing a strong teacher concept with a weakly-encoded (shuffled) student
    concept is throttled by the ``min``, so random permutations score lower. The
    reported ``gap`` is ``real_mean - null_mean``; a large positive gap with a
    small null spread (high ``z``) is the pass condition.

    Args:
        teacher_sets: teacher concept feature sets.
        student_sets: student concept feature sets.
        n_shuffle: number of random permutations used to build the null
            distribution.
        seed: RNG seed for reproducible shuffles.

    Returns:
        ``{
            "n_shared": int,            # concepts present on both sides
            "real_mean": float,         # mean match score on true pairs
            "null_mean": float,         # mean match score over shuffles
            "null_std": float,          # std of per-shuffle means
            "gap": float,               # real_mean - null_mean
            "z": float,                 # gap / null_std (inf if null_std == 0)
            "n_shuffle": int,
        }``
        Empty fields default to ``0.0`` when there are no shared concepts.
    """
    rng = random.Random(seed)

    shared = [c for c in teacher_sets if c in student_sets and teacher_sets[c] and student_sets[c]]
    if not shared:
        return {
            "n_shared": 0,
            "real_mean": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
            "gap": 0.0,
            "z": 0.0,
            "n_shuffle": int(n_shuffle),
        }

    # Pre-compute each side's best-layer mean AUC so scoring a pair is O(1).
    t_strength: dict[str, float] = {}
    s_strength: dict[str, float] = {}
    for c in shared:
        t_set = teacher_sets[c]
        s_set = student_sets[c]
        t_strength[c] = _mean_auc(t_set, _best_layer(t_set))
        s_strength[c] = _mean_auc(s_set, _best_layer(s_set))

    def _pair_score(t_concept: str, s_concept: str) -> float:
        """Cross-side match score for (teacher concept, student concept).

        Concordance of above-chance encoding strength: ``min(t-0.5, s-0.5)``
        clamped at 0. This is permutation-sensitive (a strong teacher paired with
        a weak student is throttled by the ``min``), which is what makes the
        shuffled-pair gap meaningful.
        """
        t = t_strength[t_concept] - 0.5
        s = s_strength[s_concept] - 0.5
        return max(0.0, min(t, s))

    # Real (same-label) pairings.
    real_scores = [_pair_score(c, c) for c in shared]
    real_mean = float(sum(real_scores) / len(real_scores))

    # Null distribution: align teacher concepts to a random permutation of the
    # student concepts and take the mean over all pairs, repeated n_shuffle times.
    null_means: list[float] = []
    s_pool = list(shared)
    for _ in range(int(n_shuffle)):
        permuted = s_pool[:]
        rng.shuffle(permuted)
        scores = [_pair_score(t, s) for t, s in zip(shared, permuted)]
        null_means.append(sum(scores) / len(scores))

    null_mean = float(sum(null_means) / len(null_means))
    if len(null_means) > 1:
        var = sum((m - null_mean) ** 2 for m in null_means) / (len(null_means) - 1)
        null_std = float(var ** 0.5)
    else:
        null_std = 0.0

    gap = real_mean - null_mean
    z = float("inf") if null_std == 0.0 else gap / null_std

    return {
        "n_shared": len(shared),
        "real_mean": real_mean,
        "null_mean": null_mean,
        "null_std": null_std,
        "gap": float(gap),
        "z": z,
        "n_shuffle": int(n_shuffle),
    }


def causal_steer_test(
    teacher,
    student,
    t_sae,
    s_sae,
    match: Mapping[str, Any],
    prompts: Sequence[str],
    *,
    teacher_tokenizer=None,
    student_tokenizer=None,
    hook_point: str = "mlp",
    alpha: float = 8.0,
    mode: str = "steer",
    max_len: int = 128,
    device: str | None = None,
) -> dict:
    """Causally ablate/steer a matched feature and measure a behavioral delta.

    **Gate (causality, runnable).** Stability and the null control are
    correlational. This test is interventional and *actually executes the
    models*: for a single matched concept it adds (``mode="steer"``) or removes
    (``mode="ablate"``) the matched SAE feature's **decoder direction** at the
    model's hook point (the MLP output of the discovered layer) and measures how
    much the next-token distribution moves. A matched feature that is genuinely
    the same concept on both sides should produce a *correlated* behavioral shift
    on teacher and student.

    The intervention is applied directly in activation space using the SAE
    decoder column for the feature (``W_dec[:, feature]``, unit-norm by the SAE
    training contract), so it does not require a reconstruction round-trip:

    * ``steer``  -> ``h' = h + alpha * d_hat`` (push the concept *up*)
    * ``ablate`` -> ``h' = h - (h . d_hat) d_hat`` (project the concept *out*)

    where ``d_hat`` is the (re-normalised) decoder direction. The behavioral
    metric is the mean shift in the per-token log-probability assigned to the
    argmax of the *clean* run (a model-internal, tokenizer-agnostic quantity that
    needs no shared label space), averaged over prompts.

    Args:
        teacher: the teacher ``nn.Module`` (HF causal LM). Frozen / eval.
        student: the student ``nn.Module`` (HF causal LM). Frozen / eval.
        t_sae: the teacher :class:`know_trans.sae.TopKSAE` for the teacher's
            matched layer (used only for ``decode_direction``/``W_dec``).
        s_sae: the student ``TopKSAE`` for the student's matched layer.
        match: a single match entry as produced by :func:`run_matching`, i.e.
            ``{"teacher": {"layer", "features", ...},
               "student": {"layer", "features", ...} | None, "shared": bool}``.
            The first feature of each side's ``features`` list is steered.
        prompts: text prompts to evaluate the intervention on.
        teacher_tokenizer: tokenizer for ``teacher``. If ``None`` and ``teacher``
            exposes a ``.tokenizer`` attribute it is used; otherwise this side is
            skipped.
        student_tokenizer: tokenizer for ``student`` (same fallback rule).
        hook_point: which submodule output to patch (currently ``"mlp"``; matches
            ``capture.MLPHook`` / the spec's MLP hook point).
        alpha: steering coefficient (in units of the unit-norm decoder direction).
        mode: ``"steer"`` (add direction) or ``"ablate"`` (project out direction).
        max_len: max tokenized prompt length.
        device: device override; defaults to each model's own device.

    Returns:
        ``{
            "concept_shared": bool,
            "mode": str, "alpha": float, "hook_point": str,
            "teacher": {"layer", "feature", "delta_logprob", "delta_kl", "n_prompts"} | None,
            "student": {"layer", "feature", "delta_logprob", "delta_kl", "n_prompts"} | None,
            "delta_corr": float | None,   # Pearson r of per-prompt teacher vs
                                          # student log-prob deltas (None unless
                                          # both sides ran on equal #prompts)
        }``

        For a ``shared: false`` (teacher-only) match the ``student`` side is
        ``None`` and ``delta_corr`` is ``None`` -- there is no student feature to
        steer, which is itself the finding.

    Raises:
        ValueError: if ``mode`` is not ``"steer"`` or ``"ablate"``, or if a
            requested side has no matched features.
    """
    import torch  # local import: never pull torch at module import time

    if mode not in ("steer", "ablate"):
        raise ValueError(f"mode must be 'steer' or 'ablate', got {mode!r}")
    if not prompts:
        raise ValueError("prompts must be a non-empty sequence")

    # ------------------------------------------------------------------ #
    # Resolve tokenizers (allow attribute fallback for convenience).     #
    # ------------------------------------------------------------------ #
    if teacher_tokenizer is None:
        teacher_tokenizer = getattr(teacher, "tokenizer", None)
    if student_tokenizer is None:
        student_tokenizer = getattr(student, "tokenizer", None)

    def _resolve_hook_module(model, layer: int):
        """Return the submodule whose output we patch for ``hook_point``."""
        block = model.model.layers[layer]
        if hook_point == "mlp":
            return block.mlp
        if hook_point in ("resid", "residual", "block"):
            return block
        raise ValueError(f"unsupported hook_point {hook_point!r}")

    def _decoder_direction(sae, feature: int, ref: "torch.Tensor") -> "torch.Tensor":
        """Unit-norm decoder column for ``feature`` on ref's dtype/device.

        Prefers a public ``decode_direction`` if the SAE provides one; otherwise
        reads the decoder weight matrix (``W_dec``/``decoder.weight``). The
        TopKSAE contract keeps decoder columns unit-norm, but we re-normalise
        defensively so ``alpha`` is always in unit-direction space.
        """
        d: "torch.Tensor"
        if hasattr(sae, "decode_direction"):
            d = sae.decode_direction(int(feature))
        elif hasattr(sae, "W_dec"):
            W = sae.W_dec  # expected shape [d_in, d_hidden]
            d = W[:, int(feature)] if W.shape[1] != W.shape[0] else W[:, int(feature)]
        elif hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
            # nn.Linear(d_hidden -> d_in): weight is [d_in, d_hidden].
            d = sae.decoder.weight[:, int(feature)]
        else:
            raise ValueError(
                "SAE exposes no decoder direction (need decode_direction, "
                "W_dec, or decoder.weight)"
            )
        d = d.detach().to(device=ref.device, dtype=ref.dtype).reshape(-1)
        n = torch.linalg.vector_norm(d)
        if float(n) > 0:
            d = d / n
        return d

    @torch.no_grad()
    def _run_side(model, tokenizer, sae, side: Mapping[str, Any] | None) -> dict | None:
        """Run clean + intervened forward passes for one model; return deltas.

        Returns ``None`` if the side has no tokenizer or no matched feature.
        """
        if side is None or tokenizer is None:
            return None
        feats = list(side.get("features") or [])
        if not feats:
            raise ValueError("matched side has no features to steer")
        layer = int(side["layer"])
        feature = int(feats[0])

        model.eval()
        dev = device or next(model.parameters()).device
        hook_mod = _resolve_hook_module(model, layer)

        # Mutable cell so the forward hook can switch behavior between passes.
        steer_state: dict[str, Any] = {"active": False, "dir": None}

        def _hook(_module, _inp, output):
            if not steer_state["active"]:
                return output
            # MLP/block output may be a Tensor or a tuple (hidden, ...).
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            d = steer_state["dir"]  # [d_in], unit norm, h.dtype/device
            if mode == "steer":
                h = h + alpha * d
            else:  # ablate: remove the component along d
                coeff = (h * d).sum(dim=-1, keepdim=True)
                h = h - coeff * d
            if is_tuple:
                return (h,) + tuple(output[1:])
            return h

        handle = hook_mod.register_forward_hook(_hook)
        per_prompt_delta_lp: list[float] = []
        per_prompt_delta_kl: list[float] = []
        try:
            for text in prompts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                enc = {k: v.to(dev) for k, v in enc.items()}

                # Clean pass.
                steer_state["active"] = False
                clean_logits = model(**enc).logits[0, -1].float()
                clean_logprobs = torch.log_softmax(clean_logits, dim=-1)
                top_id = int(clean_logits.argmax())

                # Intervened pass. Build the decoder direction on the model's
                # activation dtype/device (a zero-length ref tensor only carries
                # dtype+device, which is all ``_decoder_direction`` needs).
                act_ref = torch.empty(
                    0, device=dev, dtype=next(model.parameters()).dtype
                )
                steer_state["dir"] = _decoder_direction(sae, feature, act_ref)
                steer_state["active"] = True
                steered_logits = model(**enc).logits[0, -1].float()
                steered_logprobs = torch.log_softmax(steered_logits, dim=-1)

                # Behavioral delta: shift in log-prob of the clean argmax token,
                # plus symmetric KL of the full next-token distribution.
                dlp = float(steered_logprobs[top_id] - clean_logprobs[top_id])
                kl = float(
                    torch.sum(
                        torch.exp(clean_logprobs) * (clean_logprobs - steered_logprobs)
                    )
                )
                per_prompt_delta_lp.append(dlp)
                per_prompt_delta_kl.append(kl)
        finally:
            handle.remove()
            steer_state["active"] = False

        n = len(per_prompt_delta_lp)
        return {
            "layer": layer,
            "feature": feature,
            "delta_logprob": float(sum(per_prompt_delta_lp) / n) if n else 0.0,
            "delta_kl": float(sum(per_prompt_delta_kl) / n) if n else 0.0,
            "n_prompts": n,
            "_per_prompt": per_prompt_delta_lp,  # internal: for correlation
        }

    t_side = match.get("teacher")
    s_side = match.get("student")
    shared = bool(match.get("shared", s_side is not None))

    t_res = _run_side(teacher, teacher_tokenizer, t_sae, t_side)
    s_res = _run_side(student, student_tokenizer, s_sae, s_side)

    # Cross-model correlation of per-prompt deltas (only when both sides ran on
    # the same set of prompts, i.e. equal lengths).
    delta_corr: float | None = None
    if (
        t_res is not None
        and s_res is not None
        and len(t_res["_per_prompt"]) == len(s_res["_per_prompt"])
        and len(t_res["_per_prompt"]) >= 2
    ):
        delta_corr = _pearson(t_res["_per_prompt"], s_res["_per_prompt"])

    # Strip the internal per-prompt buffers from the public payload.
    if t_res is not None:
        t_res.pop("_per_prompt", None)
    if s_res is not None:
        s_res.pop("_per_prompt", None)

    return {
        "concept_shared": shared,
        "mode": mode,
        "alpha": float(alpha),
        "hook_point": hook_point,
        "teacher": t_res,
        "student": s_res,
        "delta_corr": delta_corr,
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation of two equal-length sequences.

    Args:
        xs: first sequence.
        ys: second sequence.

    Returns:
        Pearson r in ``[-1, 1]``; ``0.0`` if either sequence has zero variance.
    """
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom == 0:
        return 0.0
    return float(cov / denom)


def write_matches(matches: Mapping[str, Any], path: str) -> None:
    """Persist a matches mapping to JSON.

    Uses :func:`know_trans.utils.save_json` when available so output formatting
    and directory creation stay consistent with the rest of the pipeline; falls
    back to the stdlib so this module never hard-fails if utils is unavailable.

    Args:
        matches: the mapping returned by :func:`run_matching`.
        path: destination ``.json`` path. Parent directories are created.
    """
    try:
        from know_trans.utils import ensure_dir, save_json  # type: ignore
        import os

        ensure_dir(os.path.dirname(path) or ".")
        save_json(matches, path)
        return
    except Exception:
        pass

    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(matches, fh, indent=2, sort_keys=True)
        fh.write("\n")
