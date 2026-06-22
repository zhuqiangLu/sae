"""wfs.py — Weighted Frequency Score (WFS) feature selection, after TraceRouter.

This module implements the **WFS** sensitive-feature selector from TraceRouter
(arXiv 2601.21900, §3.1), as an *alternative* anchoring statistic to the
detector-AUROC scorer in :mod:`know_trans.score`. Both answer the same question
("which SAE features encode this knowledge item?") but with different
machinery, so we can cross-check them.

The paper's definitions (their Eqs. 1–4 and the Figure 2 legend):

    WFS(m)  = f(m) · μ(m)                         # per SAE neuron m
        f(m) = P(a_m > 0)                         # activation frequency
        μ(m) = E[a_m | a_m > 0]                   # *conditional* mean magnitude
    ΔWFS(m) = WFS_sens(m) − WFS_non-sens(m)       # differential selector

WFS is computed once over the **sensitive** samples (a knowledge item's
positives) and once over the **non-sensitive** samples (its hard negatives);
the difference ΔWFS is the selection statistic, and the Top-K features by ΔWFS
are taken as that item's sensitive feature set.

Two notes on fidelity:

* For TopK-SAE codes (which are non-negative), ``f · μ_cond`` is *algebraically*
  the plain mean activation ``E[a_m]``. We still compute ``f`` and ``μ``
  separately and persist both, because the decomposition into "how often" (f)
  vs "how strongly when it fires" (μ) is exactly what the paper trades on, and
  it lets us inspect *why* a feature is selected.
* The original WFS is computed at the **onset layer** ``l*`` only (located by
  their attention-divergence Sensitive Score, §3.1). We do not have an onset
  detector for our text-only causal-LM setting, so — exactly as the AUROC
  scorer does — we compute WFS at every requested layer and let the best layer
  be *discovered* downstream (:func:`wfs_feature_sets`).

The granularity (token-level vs pooled example-level) is decided by the caller:
:func:`delta_wfs` operates on whatever ``[N, d_hidden]`` code matrix it is
handed. :func:`wfs_score_features` is an example-level driver that mirrors
:func:`know_trans.score.score_features` so the two statistics are directly
comparable on the same pooled codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import torch

from know_trans.capture import MLPHook, _get_hook_module
from know_trans.score import _build_example_texts, _parent_dir, _resolve_layers
from know_trans.utils import batched, ensure_dir, get_device, get_dtype, get_logger

if TYPE_CHECKING:  # pragma: no cover
    from know_trans.concepts import Concept
    from know_trans.sae import SAEBundle

__all__ = [
    "weighted_frequency_score",
    "delta_wfs",
    "wfs_score_features",
    "wfs_feature_sets",
    "WFS_COLUMNS",
]

_LOG = get_logger(__name__)

# Columns of the persisted WFS table, in canonical order.
WFS_COLUMNS: list[str] = [
    "layer",
    "feature",
    "concept",
    "f_sens",
    "mu_sens",
    "wfs_sens",
    "f_nonsens",
    "mu_nonsens",
    "wfs_nonsens",
    "delta_wfs",
    "n_pos",
    "n_neg",
]


# --------------------------------------------------------------------------- #
# Core math (granularity-agnostic: rows may be tokens or pooled examples)
# --------------------------------------------------------------------------- #
def weighted_frequency_score(
    codes: torch.Tensor,
    *,
    active_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature WFS over a set of samples.

    Parameters
    ----------
    codes:
        ``[N, d_hidden]`` non-negative SAE code matrix (rows = samples, which may
        be tokens or pooled examples; columns = features). Computed in float.
    active_eps:
        A feature is "active" on a sample when ``a_m > active_eps``. The paper
        uses ``> 0``; expose a tiny threshold only to guard against denormal
        noise if ever needed (default keeps the exact paper definition).

    Returns
    -------
    (f, mu, wfs):
        Each ``[d_hidden]`` float64 arrays — activation frequency ``f(m)``,
        conditional mean magnitude ``μ(m) = E[a_m | a_m > eps]`` (0 where the
        feature never fires), and ``WFS(m) = f(m) · μ(m)``.
    """
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D [N, d_hidden], got shape {tuple(codes.shape)}")
    x = codes.to(torch.float64)
    n = x.shape[0]
    if n == 0:
        d = x.shape[1]
        z = np.zeros(d, dtype=np.float64)
        return z, z.copy(), z.copy()

    active = x > active_eps  # [N, H] bool
    count = active.sum(dim=0).to(torch.float64)  # [H] number of firing samples
    f = (count / n).cpu().numpy()  # P(a_m > eps)

    masked_sum = torch.where(active, x, torch.zeros_like(x)).sum(dim=0)  # [H]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (masked_sum / count.clamp_min(1.0)).cpu().numpy()  # conditional mean
    mu[count.cpu().numpy() == 0] = 0.0  # never-active features contribute nothing
    wfs = f * mu
    return f, mu, wfs


def delta_wfs(
    codes: torch.Tensor,
    example_ids: np.ndarray,
    labels: np.ndarray,
    *,
    active_eps: float = 0.0,
) -> dict[str, np.ndarray]:
    """Differential WFS (ΔWFS) for one knowledge item.

    Splits the participating rows of ``codes`` into sensitive (label==1) and
    non-sensitive (label==0) sets, computes WFS for each, and returns the
    difference — TraceRouter's sensitive-neuron selection statistic.

    Parameters
    ----------
    codes:
        ``[E, d_hidden]`` code matrix, rows aligned with global sample ids.
    example_ids:
        ``[n]`` global ids participating in this item (positives then negatives),
        indexing rows of ``codes``.
    labels:
        ``[n]`` binary labels aligned with ``example_ids`` (1 = sensitive).

    Returns
    -------
    dict with keys ``f_sens, mu_sens, wfs_sens, f_nonsens, mu_nonsens,
    wfs_nonsens, delta_wfs`` — each a ``[d_hidden]`` float64 array.
    """
    rows = codes[example_ids]  # [n, d_hidden]
    lab = np.asarray(labels)
    sens = rows[torch.as_tensor(lab == 1)]
    nons = rows[torch.as_tensor(lab == 0)]

    f_s, mu_s, wfs_s = weighted_frequency_score(sens, active_eps=active_eps)
    f_n, mu_n, wfs_n = weighted_frequency_score(nons, active_eps=active_eps)
    return {
        "f_sens": f_s,
        "mu_sens": mu_s,
        "wfs_sens": wfs_s,
        "f_nonsens": f_n,
        "mu_nonsens": mu_n,
        "wfs_nonsens": wfs_n,
        "delta_wfs": wfs_s - wfs_n,
    }


# --------------------------------------------------------------------------- #
# Token-level driver (faithful to TraceRouter §3.1; f(m)=P(a_m>0) over tokens)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def wfs_score_features(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    battery: Sequence["Concept"],
    layers: Sequence[int] | None,
    out_path: str,
    max_len: int = 256,
    batch_size: int = 16,
    *,
    hook_point: str = "mlp",
    device: str | None = None,
    dtype: str | torch.dtype = "bfloat16",
    active_eps: float = 0.0,
) -> pd.DataFrame:
    """Token-level ΔWFS per (layer, feature, concept), persisted as parquet.

    Faithful to TraceRouter §3.1: ``f(m)=P(a_m>0)`` and ``μ(m)=E[a_m|a_m>0]``
    are accumulated over the **tokens** of each knowledge item's positive
    ("sensitive") prompts vs its negative ("non-sensitive") prompts. The model
    is run once over all battery texts (all layers hooked in one forward pass);
    per batch/layer the dense codes are encoded and folded into sufficient
    statistics, so only one batch of codes is ever resident.

    Efficiency: TopK-SAE codes are exactly 0 or positive, so the per-feature
    active count and activation sum over a token subset are just matmuls of the
    code (and code>0) matrix against a token→concept membership matrix.

    Output columns: :data:`WFS_COLUMNS`.
    """
    device = device or get_device()
    torch_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype

    resolved_layers = _resolve_layers(layers, sae_bundle)
    if not resolved_layers:
        raise ValueError(
            "No layers to score: requested layers do not overlap with the SAE "
            f"bundle (bundle layers: {list(sae_bundle.layers)})."
        )
    if hook_point != "mlp":
        raise ValueError(f"wfs_score_features supports hook_point='mlp' (got {hook_point!r}).")
    if len(battery) == 0:
        raise ValueError("Empty battery: nothing to score.")

    texts, concept_spans = _build_example_texts(battery)
    if not texts:
        raise ValueError("Battery contained no non-empty texts.")
    n_texts = len(texts)
    cnames = [c.name for c in battery]
    n_c = len(cnames)

    # Token→concept membership over the GLOBAL example id space [n_texts].
    # is_pos[t, c] = 1 if global text t is a positive of concept c (sensitive);
    # is_neg likewise for hard negatives (non-sensitive). Concepts may share
    # negatives, so a text can be non-sensitive for several concepts at once.
    is_pos = torch.zeros(n_texts, n_c)
    is_neg = torch.zeros(n_texts, n_c)
    n_pos = np.zeros(n_c, dtype=np.int64)
    n_neg = np.zeros(n_c, dtype=np.int64)
    for ci, name in enumerate(cnames):
        eids, labels, (np_, nn_) = concept_spans[name]  # type: ignore[misc]
        n_pos[ci], n_neg[ci] = np_, nn_
        for e, l in zip(eids, labels):
            (is_pos if l == 1 else is_neg)[int(e), ci] = 1.0
    is_pos = is_pos.to(device)
    is_neg = is_neg.to(device)

    _LOG.info(
        "Token-level WFS: %d layers, %d concepts, %d unique texts (device=%s, dtype=%s).",
        len(resolved_layers), n_c, n_texts, device, torch_dtype,
    )

    model.to(device)
    model.eval()
    hooks: dict[int, MLPHook] = {
        L: MLPHook(_get_hook_module(model, L, hook_point), L, to_cpu=False)
        for L in resolved_layers
    }

    # Sufficient statistics on device (float32). Per layer: [H, C] sums/counts
    # for sensitive and non-sensitive token pools. Token counts are layer-
    # independent (same tokens), accumulated once on the first layer.
    sum_s: dict[int, torch.Tensor] = {}
    cnt_s: dict[int, torch.Tensor] = {}
    sum_n: dict[int, torch.Tensor] = {}
    cnt_n: dict[int, torch.Tensor] = {}
    ntok_s = torch.zeros(n_c, device=device)
    ntok_n = torch.zeros(n_c, device=device)
    d_hidden: int | None = None

    base = 0
    try:
        for text_batch in batched(list(texts), batch_size):
            b = len(text_batch)
            enc = tokenizer(list(text_batch), return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            attn = enc.get("attention_mask")
            model(**enc)

            global_ids = torch.arange(base, base + b, device=device)
            base += b

            first_layer = True
            for L in resolved_layers:
                acts = hooks[L].pop().float()  # [b, s, d]
                bb, s, d = acts.shape
                if attn is not None:
                    mflat = attn.bool().reshape(-1)  # [b*s]
                else:
                    mflat = torch.ones(bb * s, dtype=torch.bool, device=device)
                flat = acts.reshape(bb * s, d)[mflat]  # [T, d]
                tok_local = (
                    torch.arange(bb, device=device).unsqueeze(1).expand(bb, s).reshape(-1)[mflat]
                )
                gid = global_ids[tok_local]  # [T] global example id per token

                sae = sae_bundle[L]
                p = next(sae.parameters())
                codes = sae.encode_dense(flat.to(device=p.device, dtype=p.dtype)).float()  # [T,H]
                if d_hidden is None:
                    d_hidden = codes.shape[1]
                active = (codes > active_eps).float()  # [T,H]

                Ms = is_pos[gid]  # [T, C] sensitive membership
                Mn = is_neg[gid]  # [T, C] non-sensitive membership

                ss = codes.t() @ Ms      # [H, C] sum of activations over sens tokens
                cs = active.t() @ Ms     # [H, C] active count over sens tokens
                sn = codes.t() @ Mn
                cn = active.t() @ Mn

                if L not in sum_s:
                    sum_s[L] = ss; cnt_s[L] = cs; sum_n[L] = sn; cnt_n[L] = cn
                else:
                    sum_s[L] += ss; cnt_s[L] += cs; sum_n[L] += sn; cnt_n[L] += cn

                if first_layer:
                    ntok_s += Ms.sum(0)
                    ntok_n += Mn.sum(0)
                    first_layer = False
                del acts, flat, codes, active, Ms, Mn, ss, cs, sn, cn
            torch.cuda.empty_cache()
    finally:
        for h in hooks.values():
            h.remove()

    assert d_hidden is not None
    feature_ids = np.arange(d_hidden, dtype=np.int64)
    ns = ntok_s.cpu().numpy()  # [C] sens token counts
    nn = ntok_n.cpu().numpy()  # [C] non-sens token counts

    rows: list[pd.DataFrame] = []
    for L in resolved_layers:
        SS = sum_s[L].cpu().numpy(); CS = cnt_s[L].cpu().numpy()
        SN = sum_n[L].cpu().numpy(); CN = cnt_n[L].cpu().numpy()
        for ci, name in enumerate(cnames):
            with np.errstate(invalid="ignore", divide="ignore"):
                f_s = CS[:, ci] / ns[ci] if ns[ci] > 0 else np.zeros(d_hidden)
                f_n = CN[:, ci] / nn[ci] if nn[ci] > 0 else np.zeros(d_hidden)
                mu_s = np.where(CS[:, ci] > 0, SS[:, ci] / np.maximum(CS[:, ci], 1.0), 0.0)
                mu_n = np.where(CN[:, ci] > 0, SN[:, ci] / np.maximum(CN[:, ci], 1.0), 0.0)
            # WFS = f*mu = sum/ntokens (identity for nonneg codes); use the stable form.
            wfs_s = SS[:, ci] / ns[ci] if ns[ci] > 0 else np.zeros(d_hidden)
            wfs_n = SN[:, ci] / nn[ci] if nn[ci] > 0 else np.zeros(d_hidden)
            rows.append(pd.DataFrame({
                "layer": np.full(d_hidden, L, dtype=np.int64),
                "feature": feature_ids,
                "concept": name,
                "f_sens": f_s, "mu_sens": mu_s, "wfs_sens": wfs_s,
                "f_nonsens": f_n, "mu_nonsens": mu_n, "wfs_nonsens": wfs_n,
                "delta_wfs": wfs_s - wfs_n,
                "n_pos": np.full(d_hidden, n_pos[ci], dtype=np.int64),
                "n_neg": np.full(d_hidden, n_neg[ci], dtype=np.int64),
            }))
        _LOG.info("Layer %d: WFS over %d features x %d concepts.", L, d_hidden, n_c)

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=WFS_COLUMNS)
    df = df[WFS_COLUMNS]
    ensure_dir(_parent_dir(out_path))
    df.to_parquet(out_path, index=False)
    _LOG.info("Wrote %d WFS rows to %s (sens tokens=%s, nonsens tokens=%s)",
              len(df), out_path, ns.astype(int).tolist(), nn.astype(int).tolist())
    return df


def wfs_feature_sets(
    df: pd.DataFrame,
    top_k: int = 10,
    min_delta: float = 0.0,
) -> dict[str, list[dict]]:
    """Collapse a WFS table into a per-concept Top-K sensitive feature set.

    Mirrors :func:`know_trans.score.concept_feature_sets` but ranks by ``ΔWFS``
    (the paper's selector) instead of AUC, and discovers the best layer per
    concept as the layer carrying the single highest ΔWFS (tie-broken by the
    count of features clearing ``min_delta``).

    Returns ``concept -> [{"layer", "feature", "delta_wfs", "wfs_sens",
    "wfs_nonsens"}, ...]`` sorted by descending ΔWFS.
    """
    required = {"layer", "feature", "concept", "delta_wfs"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"WFS DataFrame missing required columns: {sorted(missing)}")

    sets: dict[str, list[dict]] = {}
    for concept_name, c_df in df.groupby("concept", sort=True):
        pos = c_df[c_df["delta_wfs"] > min_delta]
        if pos.empty:
            sets[str(concept_name)] = []
            continue
        layer_stats = (
            pos.groupby("layer")["delta_wfs"]
            .agg(peak="max", count="count")
            .reset_index()
            .sort_values(["peak", "count"], ascending=[False, False])
        )
        best_layer = int(layer_stats.iloc[0]["layer"])
        best = pos[pos["layer"] == best_layer].sort_values("delta_wfs", ascending=False).head(top_k)
        sets[str(concept_name)] = [
            {
                "layer": int(r.layer),
                "feature": int(r.feature),
                "delta_wfs": float(r.delta_wfs),
                "wfs_sens": float(getattr(r, "wfs_sens", float("nan"))),
                "wfs_nonsens": float(getattr(r, "wfs_nonsens", float("nan"))),
            }
            for r in best.itertuples(index=False)
        ]
    return sets
