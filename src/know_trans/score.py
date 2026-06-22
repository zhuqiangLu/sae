"""score.py — anchor SAE features to external concept labels via detector AUC.

This module implements the **scoring** stage of the ``know_trans`` pipeline:

    capture -> train SAEs -> build battery -> [SCORE] -> match + validate -> distill

The science (see ``docs/DESIGN_NARRATIVE.md``): we break the *circularity trap*
of correlation-based feature matching by anchoring each SAE feature to an
*external* concept label. For a labelled concept (e.g. "harmful request", with
AdvBench/HarmBench behaviours as positives and benign MMLU questions as
negatives) we treat **each SAE feature as a binary concept detector** and
measure how well its (pooled, per-example) dense activation separates positives
from negatives using ``sklearn.metrics.roc_auc_score``.

Because tokenizers differ across model families, all alignment is at the
**example / span level**: we run the model, capture the chosen ``hook_point``
(MLP output) per layer, pool token activations down to one vector per example
(via :func:`know_trans.utils.pool_examples`), then push that through the SAE's
``encode_dense`` to obtain a ``[E, d_hidden]`` code matrix per layer. Each
column (feature) is scored against the concept labels.

Two public functions are provided, matching the interface contract exactly:

* :func:`score_features` — produce the per-(layer, feature, concept) AUC table
  and persist it as parquet with columns
  ``[layer, feature, concept, auc, n_pos, n_neg]``.
* :func:`concept_feature_sets` — collapse that table into, per concept, the SET
  of above-threshold features (feature *splitting* means a concept is detected
  by several features), discovering the single best layer for each concept.

Nothing heavy runs at import time. The model is only touched inside
:func:`score_features` when it is actually called.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from know_trans.capture import MLPHook, _get_hook_module
from know_trans.utils import (
    batched,
    ensure_dir,
    get_device,
    get_dtype,
    get_logger,
    pool_examples,
)

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from know_trans.concepts import Concept
    from know_trans.sae import SAEBundle, TopKSAE

__all__ = ["score_features", "concept_feature_sets"]

_LOG = get_logger(__name__)

# Columns of the persisted feature-score table, in canonical order.
SCORE_COLUMNS: list[str] = ["layer", "feature", "concept", "auc", "n_pos", "n_neg"]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _resolve_layers(
    layers: Sequence[int] | None,
    sae_bundle: "SAEBundle",
) -> list[int]:
    """Resolve the list of layers to score.

    Parameters
    ----------
    layers:
        Explicit layer indices requested by the caller, or ``None`` to use every
        layer for which the SAE bundle has a trained SAE.
    sae_bundle:
        The bundle of per-layer SAEs; ``sae_bundle.layers`` enumerates available
        layers.

    Returns
    -------
    list[int]
        Sorted, de-duplicated layers that exist in ``sae_bundle``. Requested
        layers without a corresponding SAE are dropped with a warning.
    """
    available = list(sae_bundle.layers)
    if layers is None:
        return sorted(available)

    available_set = set(available)
    resolved: list[int] = []
    for layer in layers:
        if layer in available_set:
            resolved.append(int(layer))
        else:
            _LOG.warning(
                "Requested layer %d has no SAE in bundle (available: %s); skipping.",
                layer,
                sorted(available_set),
            )
    return sorted(dict.fromkeys(resolved))


def _build_example_texts(
    battery: Sequence["Concept"],
) -> tuple[list[str], dict[str, tuple[np.ndarray, int, int]]]:
    """Flatten the battery into a single de-duplicated list of texts.

    Many concepts share the same negative pool (e.g. benign MMLU questions reused
    across safety concepts), so we encode each unique string once and remember,
    per concept, which global example indices are positives vs negatives.

    Parameters
    ----------
    battery:
        Concepts to score. Each :class:`~know_trans.concepts.Concept` carries
        ``positives`` and ``hard_negatives`` lists of raw strings.

    Returns
    -------
    texts:
        De-duplicated list of every text referenced by any concept. Index into
        this list is the global ``example_id`` used during capture/pooling.
    concept_spans:
        Mapping ``concept_name -> (labels, n_pos, n_neg)`` where ``labels`` is an
        int array of shape ``[n_examples_for_concept]`` (1 = positive,
        0 = negative) aligned with ``example_ids`` recorded separately. We also
        return the parallel example-id array implicitly via ``_concept_index``.
    """
    # Map each unique text to a stable global id.
    text_to_id: dict[str, int] = {}
    texts: list[str] = []

    def _intern(text: str) -> int:
        idx = text_to_id.get(text)
        if idx is None:
            idx = len(texts)
            text_to_id[text] = idx
            texts.append(text)
        return idx

    concept_index: dict[str, tuple[list[int], list[int]]] = {}
    for concept in battery:
        pos_ids = [_intern(t) for t in concept.positives if t]
        neg_ids = [_intern(t) for t in concept.hard_negatives if t]
        concept_index[concept.name] = (pos_ids, neg_ids)

    # Pack into the lighter-weight structure the scorer consumes.
    concept_spans: dict[str, tuple[np.ndarray, int, int]] = {}
    for name, (pos_ids, neg_ids) in concept_index.items():
        example_ids = np.asarray(pos_ids + neg_ids, dtype=np.int64)
        labels = np.concatenate(
            [np.ones(len(pos_ids), dtype=np.int64), np.zeros(len(neg_ids), dtype=np.int64)]
        )
        # Stash example_ids alongside labels via structured tuple; downstream uses
        # both. We keep them separate to avoid an extra dataclass.
        concept_spans[name] = (example_ids, labels, (len(pos_ids), len(neg_ids)))  # type: ignore[assignment]
    return texts, concept_spans  # type: ignore[return-value]


@torch.no_grad()
def _encode_dense_per_example(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    texts: Sequence[str],
    layers: Sequence[int],
    *,
    max_len: int,
    batch_size: int,
    hook_point: str,
    pool: str,
    device: str,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Run the model and produce per-example dense SAE codes for each layer.

    For every text we capture the ``hook_point`` activation at each requested
    layer, pool the token-level activations down to a single vector per example
    using :func:`know_trans.utils.pool_examples`, then encode that pooled vector
    through the layer's SAE via ``encode_dense``.

    The model is run once over all texts; activations for every layer are
    captured in the same forward pass via :class:`know_trans.capture.MLPHook`.

    Parameters
    ----------
    model, tokenizer:
        A HuggingFace causal-LM and its tokenizer. The model is moved to
        ``device`` and put in eval mode (left to the caller's existing state if
        already there).
    sae_bundle:
        Bundle providing one :class:`~know_trans.sae.TopKSAE` per layer.
    texts:
        The de-duplicated example strings; row ``i`` is global ``example_id`` i.
    layers:
        Layers to capture & encode.
    max_len:
        Maximum tokenization length (truncation).
    batch_size:
        Number of texts per forward pass.
    hook_point:
        Sub-module to hook. Only ``"mlp"`` is supported by
        :class:`~know_trans.capture.MLPHook`; other values raise.
    pool:
        Pooling mode passed to :func:`pool_examples` (``"mean"``, ``"max"``,
        ``"last"`` …). Defaults to the contract's ``"mean"``.
    device, dtype:
        Compute device and model dtype.

    Returns
    -------
    dict[int, torch.Tensor]
        ``layer -> codes_dense`` float32 tensor of shape ``[E, d_hidden]`` on
        CPU, where ``E == len(texts)`` and rows are aligned with global
        ``example_id``.
    """
    if hook_point != "mlp":
        raise ValueError(
            f"score_features only supports hook_point='mlp' (got {hook_point!r}); "
            "MLPHook hooks model.model.layers[i].mlp."
        )

    model.to(device)
    model.eval()

    # Attach one hook per layer for the duration of the forward passes.
    hooks: dict[int, MLPHook] = {
        layer: MLPHook(_get_hook_module(model, layer, hook_point), layer, to_cpu=False)
        for layer in layers
    }

    # Accumulate pooled per-example codes per layer.
    codes_per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    try:
        for batch_idx, text_batch in enumerate(batched(list(texts), batch_size)):
            enc = tokenizer(
                list(text_batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            attention_mask = enc.get("attention_mask")

            model(**enc)

            # For each layer, pool token acts -> per-example vectors -> encode_dense.
            for layer in layers:
                acts = hooks[layer].pop()  # [B, S, d]; pop() clears the buffer
                acts = acts.float()  # pool/score in fp32 for numerical stability
                b, s, d = acts.shape

                # Flatten to [B*S, d] and build example ids per token, masking pads.
                if attention_mask is not None:
                    mask = attention_mask.bool().reshape(-1)  # [B*S]
                else:
                    mask = torch.ones(b * s, dtype=torch.bool, device=acts.device)

                flat_acts = acts.reshape(b * s, d)[mask]  # [T, d]
                # Local (within-batch) example ids 0..B-1; offset by global base.
                local_ids = (
                    torch.arange(b, device=acts.device)
                    .unsqueeze(1)
                    .expand(b, s)
                    .reshape(-1)[mask]
                )

                pooled = pool_examples(flat_acts, local_ids, mode=pool)  # [B, d]
                # pool_examples returns one row per *present* example id, sorted.
                # All B examples in the batch have >=1 unpadded token, so rows == B.

                sae: "TopKSAE" = sae_bundle[layer]
                sae_param = next(sae.parameters())
                # Cast to the SAE's own param dtype (not the model dtype): the SAE
                # is trained/stored in fp32 while the model may be fp16/bf16, and
                # F.linear requires matching dtypes.
                dense = sae.encode_dense(
                    pooled.to(device=sae_param.device, dtype=sae_param.dtype)
                )
                codes_per_layer[layer].append(dense.float().cpu())
    finally:
        for h in hooks.values():
            h.remove()

    return {layer: torch.cat(chunks, dim=0) for layer, chunks in codes_per_layer.items()}


def _auc_per_feature(
    codes: torch.Tensor,
    example_ids: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute per-feature ROC-AUC of a concept detector.

    Each feature (column of ``codes``) is treated as the score of a binary
    detector; we measure how well it ranks the concept's positives above its
    negatives via :func:`sklearn.metrics.roc_auc_score`.

    Parameters
    ----------
    codes:
        ``[E, d_hidden]`` dense SAE codes, rows aligned with global example ids.
    example_ids:
        ``[n]`` global example ids participating in this concept (positives then
        negatives), indexing rows of ``codes``.
    labels:
        ``[n]`` binary labels aligned with ``example_ids`` (1 positive, 0 neg).

    Returns
    -------
    np.ndarray
        ``[d_hidden]`` float array of AUCs; features with zero variance across
        the concept's examples (e.g. never active) get ``0.5`` (chance), since
        an undefined ranking is uninformative rather than a detector.
    """
    from sklearn.metrics import roc_auc_score  # local import: optional heavy dep

    x = codes.numpy()[example_ids]  # [n, d_hidden]
    n, d_hidden = x.shape
    aucs = np.full(d_hidden, 0.5, dtype=np.float64)

    # Degenerate label vector (all positives or all negatives) -> AUC undefined.
    n_pos = int(labels.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return aucs

    # Identify constant columns up front; roc_auc_score handles them (AUC 0.5
    # for a constant score), but skipping avoids needless work / warnings.
    col_min = x.min(axis=0)
    col_max = x.max(axis=0)
    active = np.nonzero(col_max > col_min)[0]

    for j in active:
        aucs[j] = float(roc_auc_score(labels, x[:, j]))
    return aucs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def score_features(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    battery: Sequence["Concept"],
    layers: Sequence[int] | None,
    out_path: str,
    max_len: int = 512,
    batch_size: int = 16,
    *,
    hook_point: str = "mlp",
    pool: str = "mean",
    device: str | None = None,
    dtype: str | torch.dtype = "float16",
) -> pd.DataFrame:
    """Score every SAE feature as a detector for every concept, per layer.

    Pipeline:

    1. Flatten the battery into a de-duplicated list of example texts and record,
       per concept, which example ids are positive vs negative.
    2. Run ``model`` over all texts, capturing the ``hook_point`` activation at
       each requested layer, pool token activations to one vector per example,
       and ``encode_dense`` through that layer's SAE -> ``[E, d_hidden]`` codes.
    3. For each (layer, concept), compute the ROC-AUC of every feature column
       against the concept's labels using :func:`sklearn.metrics.roc_auc_score`.
    4. Assemble a tidy table and persist it as parquet.

    Parameters
    ----------
    model, tokenizer:
        HuggingFace causal-LM and tokenizer (e.g. the teacher or the student).
    sae_bundle:
        :class:`~know_trans.sae.SAEBundle` with one trained SAE per layer; must
        match the ``model``'s family/d_model (the SAE's ``d_in`` equals the
        captured activation width).
    battery:
        Concept battery to score against (positives + hard negatives per concept).
    layers:
        Layers to score. ``None`` ⇒ every layer present in ``sae_bundle``.
        Requested layers absent from the bundle are skipped with a warning.
    out_path:
        Destination ``.parquet`` path for the score table. Parent dirs created.
    max_len:
        Tokenizer truncation length.
    batch_size:
        Texts per forward pass.
    hook_point:
        Activation hook point; only ``"mlp"`` is supported (see
        :class:`~know_trans.capture.MLPHook`).
    pool:
        Span/example pooling mode forwarded to
        :func:`know_trans.utils.pool_examples` (contract default ``"mean"``).
    device:
        Compute device; defaults to :func:`know_trans.utils.get_device`.
    dtype:
        Model/SAE compute dtype (name or ``torch.dtype``); default ``"float16"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``[layer, feature, concept, auc, n_pos, n_neg]``, one row per
        (layer, feature, concept). The same frame is written to ``out_path``.

    Notes
    -----
    * Layer correspondence is **discovered downstream**: this function scores
      every (layer, concept) pair so that :func:`concept_feature_sets` can pick
      the best layer per concept. Nothing is hardcoded.
    * ``d_hidden`` (and hence ``feature`` range) is inferred from the SAE codes,
      never assumed.
    """
    device = device or get_device()
    torch_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype

    resolved_layers = _resolve_layers(layers, sae_bundle)
    if not resolved_layers:
        raise ValueError(
            "No layers to score: requested layers do not overlap with the SAE "
            f"bundle (bundle layers: {list(sae_bundle.layers)})."
        )

    if len(battery) == 0:
        raise ValueError("Empty battery: nothing to score.")

    texts, concept_spans = _build_example_texts(battery)
    if not texts:
        raise ValueError("Battery contained no non-empty texts.")

    _LOG.info(
        "Scoring features across %d layers for %d concepts over %d unique texts "
        "(device=%s, dtype=%s, pool=%s).",
        len(resolved_layers),
        len(battery),
        len(texts),
        device,
        torch_dtype,
        pool,
    )

    codes_per_layer = _encode_dense_per_example(
        model,
        tokenizer,
        sae_bundle,
        texts,
        resolved_layers,
        max_len=max_len,
        batch_size=batch_size,
        hook_point=hook_point,
        pool=pool,
        device=device,
        dtype=torch_dtype,
    )

    rows: list[dict[str, Any]] = []
    for layer in resolved_layers:
        codes = codes_per_layer[layer]  # [E, d_hidden] float32 cpu
        d_hidden = codes.shape[1]
        feature_ids = np.arange(d_hidden, dtype=np.int64)

        for concept in battery:
            example_ids, labels, (n_pos, n_neg) = concept_spans[concept.name]  # type: ignore[misc]
            if n_pos == 0 or n_neg == 0:
                _LOG.warning(
                    "Concept %r has n_pos=%d n_neg=%d at layer %d; AUC undefined, "
                    "writing chance (0.5).",
                    concept.name,
                    n_pos,
                    n_neg,
                    layer,
                )
            aucs = _auc_per_feature(codes, example_ids, labels)

            layer_block = pd.DataFrame(
                {
                    "layer": np.full(d_hidden, layer, dtype=np.int64),
                    "feature": feature_ids,
                    "concept": concept.name,
                    "auc": aucs.astype(np.float64),
                    "n_pos": np.full(d_hidden, n_pos, dtype=np.int64),
                    "n_neg": np.full(d_hidden, n_neg, dtype=np.int64),
                }
            )
            rows.append(layer_block)

        _LOG.info("Layer %d: scored %d features x %d concepts.", layer, d_hidden, len(battery))

    df = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=SCORE_COLUMNS)
    )
    df = df[SCORE_COLUMNS]

    ensure_dir(_parent_dir(out_path))
    df.to_parquet(out_path, index=False)
    _LOG.info("Wrote %d score rows to %s", len(df), out_path)
    return df


def concept_feature_sets(
    df: pd.DataFrame,
    auc_threshold: float = 0.8,
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """Collapse the score table into a per-concept SET of detector features.

    Feature *splitting* means a single concept is typically detected by several
    SAE features rather than one — so we return a SET per concept, not a single
    best feature. We also **discover the best layer per concept**: the layer
    whose above-threshold features collectively give the strongest signal
    (highest single-feature AUC, tie-broken by count of above-threshold
    features). Only features on that winning layer are returned, keeping each
    concept anchored to one layer for clean cross-model matching downstream.

    Parameters
    ----------
    df:
        Score table as produced by :func:`score_features` with columns
        ``[layer, feature, concept, auc, n_pos, n_neg]``.
    auc_threshold:
        Minimum AUC for a feature to count as a detector for the concept.
    top_k:
        Maximum number of features to keep per concept (the strongest by AUC),
        after the best layer has been selected.

    Returns
    -------
    dict[str, list[dict]]
        ``concept_name -> [{"layer": int, "feature": int, "auc": float}, ...]``
        sorted by descending AUC. Concepts with no feature above ``auc_threshold``
        on any layer map to an empty list (still present as keys so callers can
        distinguish "covered by nothing" from "absent from battery").

    Notes
    -----
    The returned structure is exactly what :mod:`know_trans.match` consumes for
    label-anchored matching, and what :mod:`know_trans.distill` consumes to build
    each model's concept-activation space.
    """
    required = set(SCORE_COLUMNS[:4])  # layer, feature, concept, auc
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Score DataFrame missing required columns: {sorted(missing)}; "
            f"expected at least {sorted(required)}."
        )

    sets: dict[str, list[dict]] = {}

    for concept_name, c_df in df.groupby("concept", sort=True):
        above = c_df[c_df["auc"] >= auc_threshold]
        if above.empty:
            sets[str(concept_name)] = []
            continue

        # Discover the best layer for this concept.
        # Primary key: max AUC achieved on the layer (peak detector strength).
        # Tie-break: number of above-threshold features (richer, more robust set).
        layer_stats = (
            above.groupby("layer")["auc"]
            .agg(peak="max", count="count")
            .reset_index()
            .sort_values(["peak", "count"], ascending=[False, False])
        )
        best_layer = int(layer_stats.iloc[0]["layer"])

        best = above[above["layer"] == best_layer].sort_values(
            "auc", ascending=False
        )
        best = best.head(top_k)

        sets[str(concept_name)] = [
            {
                "layer": int(r.layer),
                "feature": int(r.feature),
                "auc": float(r.auc),
            }
            for r in best.itertuples(index=False)
        ]

    return sets


# --------------------------------------------------------------------------- #
# Small path helper (kept local to avoid widening utils' public surface)
# --------------------------------------------------------------------------- #
def _parent_dir(path: str) -> str:
    """Return the parent directory of ``path`` (``"."`` if it has none)."""
    import os

    parent = os.path.dirname(os.path.abspath(path))
    return parent or "."
