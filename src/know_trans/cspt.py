"""cspt.py — Causal Semantic Pathway Tracing (TraceRouter §3.2).

Given a knowledge item, trace the *causal* cross-layer pathway that propagates it,
following TraceRouter (arXiv 2601.21900, §3.2). Pipeline, per (model, knowledge):

1. **Onset layer ``l*``** — the earliest layer where the knowledge first emerges
   (first layer whose best detector AUROC crosses ``tau_onset``; falls back to the
   global-best AUROC layer). This is our text-LM stand-in for the paper's
   attention-divergence onset (their SS needs minimal pairs + token roles we lack;
   see the design discussion). AUROC *localizes*; WFS *selects* (next step).

2. **Source neurons ``S_src``** — select the Top-``k_feat`` sensitive SAE features
   at ``l*`` by ΔWFS (the WFS selector), then back-project through the decoder:

       Z_proj = W_dec[:, sel] @ w_sel            # dense direction, MLP-output space
       S_src  = Top-k_src indices of |Z_proj|    # paper Eq.: largest-magnitude dims

   ``w_sel`` weights each selected feature by its ΔWFS (its sensitivity).

3. **Zero-out intervention + FIS** — run the battery twice (clean / with the MLP
   output coords ``S_src`` forced to 0 at ``l*``). For every downstream dense
   neuron ``m`` (layers ``l > l*``), accumulate over the knowledge's *sensitive*
   (positive-prompt) tokens:

       f(m)   = P(a_m > 0)                        # activation frequency  (clean)
       μ(m)   = E[a_m | a_m > 0]                  # conditional mean mag  (clean)
       Δ(m)   = E[|a_m − â_m|]                    # zero-out activation shift
       FIS(m) = f(m) · μ(m) · Δ(m)               # TraceRouter Eq. 5

   Top-FIS neurons per downstream layer form the **pathway**.

4. **Validation** — re-score the knowledge's best detector at each downstream layer
   on clean vs intervened activations. A *real* pathway means zeroing ``S_src``
   collapses the downstream detectability (AUROC_clean → AUROC_intervened drops).

Everything is dense-neuron based (the faithful choice); the intervention is on the
MLP output because that is where the SAE — and thus ``W_dec`` — lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import torch

from know_trans.capture import MLPHook, _get_hook_module
from know_trans.score import _build_example_texts
from know_trans.utils import batched, ensure_dir, get_device, get_dtype, get_logger

if TYPE_CHECKING:  # pragma: no cover
    from know_trans.concepts import Concept
    from know_trans.sae import SAEBundle

__all__ = [
    "pick_onset_layer",
    "source_neurons",
    "ZeroOutHook",
    "trace_pathway",
    "trace_pathway_chain",
    "trace_pathway_greedy",
    "suppress_pathway_validation",
    "FIS_COLUMNS",
    "CHAIN_NODE_COLUMNS",
    "CHAIN_EDGE_COLUMNS",
    "GREEDY_NODE_COLUMNS",
]

_LOG = get_logger(__name__)

FIS_COLUMNS: list[str] = [
    "layer", "neuron", "concept", "f", "mu", "delta", "fis",
]

CHAIN_NODE_COLUMNS: list[str] = ["layer", "neuron", "concept", "f", "mu", "fmu"]
CHAIN_EDGE_COLUMNS: list[str] = [
    "src_layer", "src_neuron", "dst_layer", "dst_neuron", "concept", "delta", "fmu_dst",
]


# --------------------------------------------------------------------------- #
# Step 1: onset layer
# --------------------------------------------------------------------------- #
def pick_onset_layer(
    auroc_df: pd.DataFrame,
    concept: str,
    tau_onset: float = 0.70,
) -> tuple[int, float]:
    """Earliest layer whose best detector AUROC crosses ``tau_onset``.

    Falls back to the global-best AUROC layer if none crosses. Returns
    ``(l_star, peak_auc_at_l_star)``.
    """
    sub = auroc_df[auroc_df["concept"] == concept]
    if sub.empty:
        raise ValueError(f"concept {concept!r} absent from AUROC table")
    per_layer = sub.groupby("layer")["auc"].max().sort_index()
    crossed = per_layer[per_layer >= tau_onset]
    if len(crossed):
        l_star = int(crossed.index[0])
    else:
        l_star = int(per_layer.idxmax())
        _LOG.warning("concept %s: no layer crosses tau=%.2f; falling back to global-best L%d",
                     concept, tau_onset, l_star)
    return l_star, float(per_layer.loc[l_star])


# --------------------------------------------------------------------------- #
# Step 2: source neurons via WFS-selected features back-projected through W_dec
# --------------------------------------------------------------------------- #
def source_neurons(
    sae: Any,
    wfs_df: pd.DataFrame,
    concept: str,
    layer: int,
    k_feat: int = 10,
    k_src: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Back-project the Top-``k_feat`` ΔWFS features at ``layer`` to dense ``S_src``.

    Returns ``(src_idx, z_proj, sel_features)`` where ``src_idx`` are the Top-
    ``k_src`` dense coordinate indices of ``|Z_proj|`` (the source neurons), and
    ``z_proj`` is the full dense back-projection vector ``[d_in]``.
    """
    sub = wfs_df[(wfs_df["concept"] == concept) & (wfs_df["layer"] == layer)]
    if sub.empty:
        raise ValueError(f"no WFS rows for concept={concept!r} layer={layer}")
    sel = sub.nlargest(k_feat, "delta_wfs")
    feats = [int(f) for f in sel["feature"]]

    W_dec = sae.W_dec.detach().float().cpu()  # [d_in, d_hidden], unit-norm cols
    # Paper (TraceRouter §3.2): Z_proj = W_dec · m_sens — an UNWEIGHTED back-projection
    # of the selected sensitive features (m_sens = their indicator). ΔWFS selects feats
    # above but must NOT also weight the decoder columns (doing so shrank |Z_proj| ~24x
    # via the small ΔWFS magnitudes and picked different, weaker source coords).
    z_proj = W_dec[:, feats].sum(dim=1)  # [d_in] dense direction
    k = min(k_src, z_proj.numel())
    src_idx = torch.topk(z_proj.abs(), k).indices.sort().values.numpy().astype(np.int64)
    return src_idx, z_proj.numpy(), feats


# --------------------------------------------------------------------------- #
# Intervention hook: zero given MLP-output coordinates
# --------------------------------------------------------------------------- #
class ZeroOutHook:
    """Forward hook that forces selected coordinates of an MLP output to zero.

    ``register_forward_hook`` may return a replacement output; we clone the MLP
    output, zero the ``coords`` columns, and return it. ``enabled`` toggles the
    intervention so the same hook can serve clean and intervened passes.
    """

    def __init__(self, module: Any, coords: np.ndarray) -> None:
        self.set_coords(coords)
        self.enabled = False
        self._handle = module.register_forward_hook(self._hook)

    def set_coords(self, coords: np.ndarray) -> None:
        self.coords = torch.as_tensor(np.ascontiguousarray(coords), dtype=torch.long)

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        if not self.enabled:
            return output
        out = output[0] if isinstance(output, (tuple, list)) else output
        out = out.clone()
        out[..., self.coords.to(out.device)] = 0.0
        if isinstance(output, (tuple, list)):
            return (out, *output[1:])
        return out

    def remove(self) -> None:
        self._handle.remove()


# --------------------------------------------------------------------------- #
# Step 3 + 4: trace FIS and validate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def trace_pathway(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    concept_obj: "Concept",
    l_star: int,
    src_idx: np.ndarray,
    *,
    auroc_df: pd.DataFrame | None = None,
    max_len: int = 256,
    batch_size: int = 16,
    hook_point: str = "mlp",
    device: str | None = None,
    dtype: str | torch.dtype = "bfloat16",
    active_eps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace the FIS pathway for one knowledge item and validate it.

    Runs the concept's positives (sensitive) + hard negatives twice — clean and
    with ``src_idx`` zeroed at ``l_star`` — capturing downstream MLP activations
    (layers ``> l_star`` that have an SAE). Accumulates FIS sufficient statistics
    over *sensitive* tokens, and (if ``auroc_df`` given) re-scores the knowledge's
    best downstream detector clean-vs-intervened for validation.

    Returns ``(fis_df, valid_df)``:
      * ``fis_df`` columns :data:`FIS_COLUMNS` (one row per downstream dense neuron)
      * ``valid_df`` columns ``[layer, feature, auc_clean, auc_intervened, drop]``
    """
    device = device or get_device()
    torch_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype

    all_layers = sorted(int(l) for l in sae_bundle.layers)
    down = [l for l in all_layers if l > l_star]
    if not down:
        raise ValueError(f"onset l*={l_star} has no downstream SAE layers (max={max(all_layers)})")

    # battery texts for THIS concept only (positives + hard negatives)
    from know_trans.concepts import Concept  # noqa: F401  (typing only)
    texts, spans = _build_example_texts([concept_obj])
    example_ids, labels, (n_pos, n_neg) = spans[concept_obj.name]  # type: ignore[misc]
    # global text id -> sensitive? (positive prompt). texts are this concept's only.
    is_sens = np.zeros(len(texts), dtype=bool)
    for e, l in zip(example_ids, labels):
        if l == 1:
            is_sens[int(e)] = True
    is_sens_t = torch.tensor(is_sens, device=device)

    model.to(device)
    model.eval()

    read_hooks = {l: MLPHook(_get_hook_module(model, l, hook_point), l, to_cpu=False) for l in down}
    zero_hook = ZeroOutHook(_get_hook_module(model, l_star, hook_point), src_idx)

    # FIS sufficient stats per downstream layer (over SENSITIVE tokens), dense neurons.
    d_in = {l: sae_bundle[l].W_dec.shape[0] for l in down}
    cnt = {l: torch.zeros(d_in[l], device=device) for l in down}   # active count (clean)
    ssum = {l: torch.zeros(d_in[l], device=device) for l in down}  # sum a (clean)
    dsum = {l: torch.zeros(d_in[l], device=device) for l in down}  # sum |a - â|
    n_sens_tok = {l: 0.0 for l in down}

    # Validation: best detector feature per downstream layer for this concept.
    best_feat = {}
    if auroc_df is not None:
        for l in down:
            s = auroc_df[(auroc_df["concept"] == concept_obj.name) & (auroc_df["layer"] == l)]
            if len(s):
                best_feat[l] = int(s.loc[s["auc"].idxmax(), "feature"])
    # pooled per-example detector scores (clean / intervened) + labels, for AUROC.
    val_clean = {l: [] for l in down}
    val_interv = {l: [] for l in down}
    val_label: list[int] = []

    def _run(enc, enabled):
        zero_hook.enabled = enabled
        model(**enc)
        return {l: read_hooks[l].pop().float() for l in down}  # layer -> [B,S,d]

    base = 0
    for tb in batched(list(texts), batch_size):
        b = len(tb)
        enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        attn = enc.get("attention_mask")
        gids = torch.arange(base, base + b, device=device)
        base += b

        clean = _run(enc, False)
        interv = _run(enc, True)

        for l in down:
            a = clean[l]; ah = interv[l]  # [B,S,d]
            bb, s, d = a.shape
            mflat = attn.bool().reshape(-1) if attn is not None else torch.ones(bb * s, dtype=torch.bool, device=device)
            af = a.reshape(bb * s, d)[mflat]      # [T,d]
            ahf = ah.reshape(bb * s, d)[mflat]    # [T,d]
            tok_local = torch.arange(bb, device=device).unsqueeze(1).expand(bb, s).reshape(-1)[mflat]
            tok_sens = is_sens_t[gids[tok_local]]  # [T] bool sensitive token?

            sa = af[tok_sens]      # [Ts,d] clean sensitive tokens
            sah = ahf[tok_sens]    # [Ts,d] intervened sensitive tokens
            active = (sa > active_eps).float()
            cnt[l] += active.sum(0)
            ssum[l] += (sa * active).sum(0)  # sum POSITIVE acts only: mu = E[a|a>0] (dense acts are signed)
            dsum[l] += (sa - sah).abs().sum(0)
            n_sens_tok[l] += float(sa.shape[0])

            # validation: pooled-per-example detector score for best feature
            if l in best_feat:
                sae = sae_bundle[l]; p = next(sae.parameters())
                # mean-pool tokens per example (this batch), clean & intervened
                pooled_c = _mean_pool(af, tok_local, bb)   # [B,d]
                pooled_i = _mean_pool(ahf, tok_local, bb)
                fc = sae.encode_dense(pooled_c.to(p.device, p.dtype)).float()[:, best_feat[l]]
                fi = sae.encode_dense(pooled_i.to(p.device, p.dtype)).float()[:, best_feat[l]]
                val_clean[l].append(fc.cpu().numpy())
                val_interv[l].append(fi.cpu().numpy())
        # labels for this batch's examples (pos prompts -> 1)
        val_label.extend(int(is_sens[g]) for g in range(base - b, base))
        torch.cuda.empty_cache()

    for h in read_hooks.values():
        h.remove()
    zero_hook.remove()

    # assemble FIS table
    rows = []
    for l in down:
        c = cnt[l].cpu().numpy(); sm = ssum[l].cpu().numpy(); dm = dsum[l].cpu().numpy()
        n = max(n_sens_tok[l], 1.0)
        f = c / n
        mu = np.where(c > 0, sm / np.maximum(c, 1.0), 0.0)
        delta = dm / n
        fis = f * mu * delta
        rows.append(pd.DataFrame({
            "layer": np.full(len(f), l, dtype=np.int64),
            "neuron": np.arange(len(f), dtype=np.int64),
            "concept": concept_obj.name,
            "f": f, "mu": mu, "delta": delta, "fis": fis,
        }))
    fis_df = pd.concat(rows, ignore_index=True)[FIS_COLUMNS]

    # assemble validation table
    from sklearn.metrics import roc_auc_score
    y = np.asarray(val_label)
    vrows = []
    for l in down:
        if l not in best_feat or not val_clean[l]:
            continue
        xc = np.concatenate(val_clean[l]); xi = np.concatenate(val_interv[l])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            ac = float(roc_auc_score(y, xc)); ai = float(roc_auc_score(y, xi))
        except ValueError:
            continue
        vrows.append({"layer": l, "feature": best_feat[l],
                      "auc_clean": round(ac, 3), "auc_intervened": round(ai, 3),
                      "drop": round(ac - ai, 3)})
    valid_df = pd.DataFrame(vrows, columns=["layer", "feature", "auc_clean", "auc_intervened", "drop"])
    return fis_df, valid_df


# --------------------------------------------------------------------------- #
# Greedy causal+differential chain (de-contaminated; no magnitude selection)
# --------------------------------------------------------------------------- #
GREEDY_NODE_COLUMNS: list[str] = [
    "layer", "neuron", "concept", "f", "mu", "fmu", "fmu_benign", "diff", "score", "l_star",
]


@torch.no_grad()
def trace_pathway_greedy(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    wfs_df: pd.DataFrame,
    auroc_df: pd.DataFrame,
    concept_obj: "Concept",
    *,
    tau_onset: float = 0.70,
    k_src: int | str = "elbow",
    kmin: int = 2,
    kmax: int = 16,
    kmax_feat: int = 32,
    n_cap: int | None = 128,
    max_len: int = 256,
    batch_size: int = 16,
    hook_point: str = "mlp",
    device: str | None = None,
    active_eps: float = 0.0,
) -> tuple[pd.DataFrame, int]:
    """Greedy prior-layer pathway: SAE-space WFS source + CAUSAL x DIFFERENTIAL probe.

    Faithful source (TraceRouter Alg.1+2), then a dense downstream probe:

    * **Step 1 (onset l*)** from the AUROC table.
    * **Step 2 (SAE space)** at l*: rank SAE features by the contrast
      ``ΔWFS(m) = WFS_sens(m) - WFS_nonsens(m)``; keep the elbow-K hub features
      -> ``m_sens``.
    * **Step 3 (back-project -> dense)**: ``Z_proj = W_dec · m_sens`` (weighted by
      ΔWFS); ``S_{l*}`` = elbow-K dense neurons by ``|Z_proj|``. This is the root.
    * **Step 4+ (dense greedy probe)** for each layer L>l*:
        Delta(m) = E|a_m^clean - a_m^{zero S_{L-1} at L-1}|   over sensitive tokens
        diff(m)  = (f*mu)_sens(m) - (f*mu)_benign(m)
        score(m) = Delta(m) * max(diff(m), 0)                 # causal AND differential
        S_L      = elbow-K top score

    Downstream, a neuron survives only if zeroing the *prior* layer's pathway
    moves it (causal / on-route) AND it fires more on the subject than on benign
    text (differential). Returns ``(nodes_df, l_star)``.
    """
    device = device or get_device()
    cname = concept_obj.name
    n_layers = int(model.config.num_hidden_layers)
    l_star, _ = pick_onset_layer(auroc_df, cname, tau_onset)
    layers = [l for l in range(n_layers) if l >= l_star]
    if len(layers) < 2 and l_star > 0:           # guarantee at least one hop
        l_star -= 1
        layers = [l for l in range(n_layers) if l >= l_star]

    texts, spans = _build_example_texts([concept_obj])
    eids, labels, _ = spans[cname]  # type: ignore[misc]
    is_sens = np.zeros(len(texts), dtype=bool)
    for e, l in zip(eids, labels):
        if l == 1:
            is_sens[int(e)] = True
    pos = [texts[i] for i in range(len(texts)) if is_sens[i]]
    neg = [texts[i] for i in range(len(texts)) if not is_sens[i]]
    if n_cap:
        pos, neg = pos[:n_cap], neg[:n_cap]
    if not pos or not neg:
        raise ValueError(f"concept {cname!r} needs both positives and negatives")

    model.to(device); model.eval()

    def _flat(a, attn):
        bb, s, d = a.shape
        m = attn.bool().reshape(-1) if attn is not None else torch.ones(bb * s, dtype=torch.bool, device=a.device)
        return a.reshape(bb * s, d)[m]

    def clean_fmu(prompts):
        """Per-layer (f*mu, f, mu) over a prompt set (clean)."""
        reads = {L: MLPHook(_get_hook_module(model, L, hook_point), L, to_cpu=False) for L in layers}
        cnt = {L: None for L in layers}; ssum = {L: None for L in layers}; ntok = 0
        for tb in batched(list(prompts), batch_size):
            enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}; attn = enc.get("attention_mask")
            model(**enc); first = True
            for L in layers:
                af = _flat(reads[L].pop().float(), attn)
                act = (af > active_eps).float()
                c = act.sum(0); s = (af * act).sum(0)
                cnt[L] = c if cnt[L] is None else cnt[L] + c
                ssum[L] = s if ssum[L] is None else ssum[L] + s
                if first:
                    ntok += af.shape[0]; first = False
            torch.cuda.empty_cache()
        for h in reads.values():
            h.remove()
        out = {}
        for L in layers:
            c = cnt[L].cpu().numpy(); s = ssum[L].cpu().numpy()
            f = c / max(ntok, 1); mu = np.where(c > 0, s / np.maximum(c, 1.0), 0.0)
            out[L] = (f * mu, f, mu)
        return out

    fmu_sens = clean_fmu(pos)
    fmu_ben = clean_fmu(neg)
    diff = {L: fmu_sens[L][0] - fmu_ben[L][0] for L in layers}

    def pick(score_vec):
        order = np.argsort(score_vec)[::-1]
        k = elbow_k(score_vec[order], kmin=kmin, kmax=kmax) if k_src == "elbow" else int(k_src)
        return np.sort(order[:k]).astype(np.int64)

    # ---- Source S_{l*}: SAE-space WFS -> decoder back-projection (Alg.1+2) ----
    # Step 2 (SAE space): rank features by deltaWFS, keep elbow-K hub features.
    wsub = (wfs_df[(wfs_df["concept"] == cname) & (wfs_df["layer"] == l_star)]
            .sort_values("delta_wfs", ascending=False))
    if wsub.empty:
        raise ValueError(f"no WFS rows for {cname!r} at l*={l_star}")
    dwv = wsub["delta_wfs"].to_numpy()
    kf = elbow_k(dwv, kmin=kmin, kmax=kmax_feat) if k_src == "elbow" else int(k_src)
    feats = wsub["feature"].to_numpy()[:kf].astype(np.int64)
    # Step 3 (dense): back-project m_sens (the selected sensitive features) through the
    # decoder UNWEIGHTED, per TraceRouter §3.2: Z_proj = W_dec · m_sens. ΔWFS selects
    # feats (above) but must NOT also weight the columns (that shrank |Z_proj| ~24x).
    W_dec = sae_bundle[l_star].W_dec.detach().float().cpu().numpy()   # [d_in, H], unit-norm cols
    z_proj = W_dec[:, feats].sum(axis=1)                              # [d_in]
    zabs = np.abs(z_proj)
    order = np.argsort(zabs)[::-1]
    ks = elbow_k(zabs[order], kmin=kmin, kmax=kmax) if k_src == "elbow" else int(k_src)
    nodes = {l_star: np.sort(order[:ks]).astype(np.int64)}
    score_by_layer = {l_star: zabs}                              # source 'score' = |Z_proj|
    _LOG.info("[%s] source l*=%d: %d deltaWFS feats -> %d dense S_src (|z| max %.3f)",
              cname, l_star, kf, len(nodes[l_star]), float(zabs.max()))
    for idx in range(1, len(layers)):
        Lp, L = layers[idx - 1], layers[idx]
        read = MLPHook(_get_hook_module(model, L, hook_point), L, to_cpu=False)
        zero = ZeroOutHook(_get_hook_module(model, Lp, hook_point), nodes[Lp])
        dsum = None; ntok = 0
        for tb in batched(list(pos), batch_size):
            enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}; attn = enc.get("attention_mask")
            zero.enabled = False; model(**enc); clean = _flat(read.pop().float(), attn)
            zero.enabled = True;  model(**enc); abl = _flat(read.pop().float(), attn)
            d = (clean - abl).abs().sum(0)
            dsum = d if dsum is None else dsum + d
            ntok += clean.shape[0]
            torch.cuda.empty_cache()
        read.remove(); zero.remove()
        delta = (dsum / max(ntok, 1)).cpu().numpy()
        score = delta * np.maximum(diff[L], 0.0)    # causal x differential
        score_by_layer[L] = score
        nodes[L] = pick(score)
        _LOG.info("[%s] greedy %d->%d: |prev|=%d -> |S_%d|=%d (max score %.4f)",
                  cname, Lp, L, len(nodes[Lp]), L, len(nodes[L]), float(score.max()))

    rows = []
    for L in layers:
        fmu_s, f_s, mu_s = fmu_sens[L]
        for n in nodes[L]:
            rows.append({"layer": L, "neuron": int(n), "concept": cname,
                         "f": float(f_s[n]), "mu": float(mu_s[n]), "fmu": float(fmu_s[n]),
                         "fmu_benign": float(fmu_ben[L][0][n]), "diff": float(diff[L][n]),
                         "score": float(score_by_layer[L][n]), "l_star": int(l_star)})
    nodes_df = pd.DataFrame(rows, columns=GREEDY_NODE_COLUMNS)
    return nodes_df, l_star


# --------------------------------------------------------------------------- #
# Chained (layer-to-layer) pathway: per-layer WFS nodes + per-neuron edges
# --------------------------------------------------------------------------- #
@torch.no_grad()
def trace_pathway_chain(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    wfs_df: pd.DataFrame,
    concept_obj: "Concept",
    *,
    k_feat: int = 10,
    k_src: int | str = 8,
    n_cap: int | None = 128,
    max_len: int = 256,
    batch_size: int = 16,
    hook_point: str = "mlp",
    device: str | None = None,
    dtype: str | torch.dtype = "bfloat16",
    active_eps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace a *connected* pathway with real layer-to-layer (one-hop) causality.

    Unlike :func:`trace_pathway` (a star: every downstream neuron measured vs a
    single source ablation at the onset), this builds the paper's connected
    circuit:

    * **Nodes** — at *every* layer L, the per-layer WFS sensitive set: the
      Top-``k_feat`` ΔWFS features at L back-projected through ``W_dec`` to the
      Top-``k_src`` dense MLP-output coordinates (i.e. :func:`source_neurons`
      applied at L, not just the onset).
    * **Edges** — for each adjacent pair (L-1 → L) and each source node ``i`` at
      L-1: zero **only ``i``** at L-1 and read L; the edge weight to each
      destination node ``j`` at L is ``E|a_j^clean − a_j^{zero i}|`` over the
      knowledge's *sensitive* (positive-prompt) tokens. Reading at L immediately
      after cutting at L-1 makes this a pure one-hop dependence — "neuron j at L
      depends on neuron i in its immediately prior layer."

    Cost is ``O(Σ_L k_src)`` ablation passes over the positive battery (one
    forward per source node per layer), plus one clean pass for node statistics.

    Returns ``(nodes_df, edges_df)`` with columns :data:`CHAIN_NODE_COLUMNS` and
    :data:`CHAIN_EDGE_COLUMNS`.
    """
    device = device or get_device()
    layers = sorted(int(l) for l in sae_bundle.layers)
    cname = concept_obj.name

    # ---- 1. per-layer sensitive nodes (WFS features -> dense coords) ----
    # k_src is a fixed int OR "elbow": the per-layer Kneedle elbow on the sorted
    # |z_proj| curve (paper Fig. 11 K-selection), so concentrated layers keep few
    # nodes and diffuse layers more.
    nodes: dict[int, np.ndarray] = {}
    for L in layers:
        idx, z_proj, _feats = source_neurons(sae_bundle[L], wfs_df, cname, L,
                                             k_feat=k_feat, k_src=(8 if k_src == "elbow" else int(k_src)))
        if k_src == "elbow":
            zabs = np.abs(np.asarray(z_proj))
            order = np.argsort(zabs)[::-1]
            # Kneedle on the HEAD only (paper Fig.11 inspects the top ranks); the
            # full 4096-long tail otherwise swamps the chord and pins k at kmax.
            k = elbow_k(zabs[order][:48], kmin=2, kmax=16)
            nodes[L] = np.sort(order[:k]).astype(np.int64)
        else:
            nodes[L] = np.asarray(idx, dtype=np.int64)

    # ---- 2. positive (sensitive) prompts only (all their tokens are sensitive) ----
    texts, spans = _build_example_texts([concept_obj])
    example_ids, labels, _ = spans[cname]  # type: ignore[misc]
    is_sens = np.zeros(len(texts), dtype=bool)
    for e, l in zip(example_ids, labels):
        if l == 1:
            is_sens[int(e)] = True
    pos_texts = [texts[i] for i in range(len(texts)) if is_sens[i]]
    if n_cap is not None and len(pos_texts) > n_cap:
        pos_texts = pos_texts[:n_cap]
    if not pos_texts:
        raise ValueError(f"concept {cname!r} has no positive prompts to trace")

    model.to(device)
    model.eval()

    def _flat(acts: torch.Tensor, attn: torch.Tensor | None) -> torch.Tensor:
        bb, s, d = acts.shape
        if attn is not None:
            mflat = attn.bool().reshape(-1)
        else:
            mflat = torch.ones(bb * s, dtype=torch.bool, device=acts.device)
        return acts.reshape(bb * s, d)[mflat]  # [T, d]

    # ---- 3. clean node stats (f, mu) over sensitive tokens, all layers ----
    read_all = {L: MLPHook(_get_hook_module(model, L, hook_point), L, to_cpu=False) for L in layers}
    cnt = {L: torch.zeros(len(nodes[L]), device=device) for L in layers}
    ssum = {L: torch.zeros(len(nodes[L]), device=device) for L in layers}
    ntok = 0.0
    for tb in batched(list(pos_texts), batch_size):
        enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        attn = enc.get("attention_mask")
        model(**enc)
        first = True
        for L in layers:
            af = _flat(read_all[L].pop().float(), attn)  # [T,d]
            sub = af[:, torch.as_tensor(nodes[L], device=device)]  # [T,k]
            act = (sub > active_eps).float()
            cnt[L] += act.sum(0)
            ssum[L] += (sub * act).sum(0)
            if first:
                ntok += float(sub.shape[0]); first = False
        torch.cuda.empty_cache()
    for h in read_all.values():
        h.remove()

    node_f = {L: (cnt[L] / max(ntok, 1.0)).cpu().numpy() for L in layers}
    node_mu = {L: np.where(cnt[L].cpu().numpy() > 0,
                           ssum[L].cpu().numpy() / np.maximum(cnt[L].cpu().numpy(), 1.0), 0.0)
               for L in layers}
    node_fmu = {L: node_f[L] * node_mu[L] for L in layers}

    node_rows = []
    for L in layers:
        for j, n in enumerate(nodes[L]):
            node_rows.append({"layer": L, "neuron": int(n), "concept": cname,
                              "f": float(node_f[L][j]), "mu": float(node_mu[L][j]),
                              "fmu": float(node_fmu[L][j])})
    nodes_df = pd.DataFrame(node_rows, columns=CHAIN_NODE_COLUMNS)

    # ---- 4. per-neuron one-hop edges (L-1 -> L) ----
    edge_rows = []
    for p in range(len(layers) - 1):
        Ls, Ld = layers[p], layers[p + 1]
        src, dst = nodes[Ls], nodes[Ld]
        if len(src) == 0 or len(dst) == 0:
            continue
        dst_t = torch.as_tensor(dst, device=device)
        read = MLPHook(_get_hook_module(model, Ld, hook_point), Ld, to_cpu=False)
        zero = ZeroOutHook(_get_hook_module(model, Ls, hook_point), src[:1])
        dsum = torch.zeros(len(src), len(dst), device=device)  # [k_src, k_dst]
        npair = 0.0
        for tb in batched(list(pos_texts), batch_size):
            enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            attn = enc.get("attention_mask")
            zero.enabled = False
            model(**enc)
            clean = _flat(read.pop().float(), attn)[:, dst_t]  # [T,k_dst]
            npair += float(clean.shape[0])
            for ii, i in enumerate(src):
                zero.set_coords(np.asarray([i], dtype=np.int64))
                zero.enabled = True
                model(**enc)
                ab = _flat(read.pop().float(), attn)[:, dst_t]  # [T,k_dst]
                dsum[ii] += (clean - ab).abs().sum(0)
            torch.cuda.empty_cache()
        read.remove(); zero.remove()
        delta = (dsum / max(npair, 1.0)).cpu().numpy()  # [k_src,k_dst]
        for ii, i in enumerate(src):
            for jj, j in enumerate(dst):
                edge_rows.append({"src_layer": Ls, "src_neuron": int(i),
                                  "dst_layer": Ld, "dst_neuron": int(j),
                                  "concept": cname, "delta": float(delta[ii, jj]),
                                  "fmu_dst": float(node_fmu[Ld][jj])})
        _LOG.info("[%s] edges %d->%d: %d src x %d dst (max delta %.4f)",
                  cname, Ls, Ld, len(src), len(dst), float(delta.max()) if delta.size else 0.0)

    edges_df = pd.DataFrame(edge_rows, columns=CHAIN_EDGE_COLUMNS)
    return nodes_df, edges_df


def elbow_k(vals_desc: np.ndarray, kmin: int = 3, kmax: int = 256) -> int:
    """Kneedle-style elbow on a DESCENDING score curve (paper's Fig. 11 method).

    Finds the rank of maximum perpendicular distance from the chord joining the
    first and last points of the sorted-descending curve — the "elbow" before the
    long tail. Clamped to [kmin, kmax]. Returns how many top neurons to keep.
    """
    v = np.asarray(vals_desc, dtype=np.float64)
    n = len(v)
    if n <= kmin:
        return n
    x = np.arange(n) / (n - 1)
    rng = (v.max() - v.min())
    y = (v - v.min()) / (rng if rng > 0 else 1.0)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    k = int(np.argmax(dist)) + 1
    return int(max(kmin, min(k, kmax, n)))


@torch.no_grad()
def suppress_pathway_validation(
    model: Any,
    tokenizer: Any,
    sae_bundle: "SAEBundle",
    concept_obj: "Concept",
    l_star: int,
    src_idx: np.ndarray,
    fis_df: pd.DataFrame,
    auroc_df: pd.DataFrame,
    *,
    k_down: int | str = 10,
    light: bool = False,
    max_len: int = 256,
    batch_size: int = 16,
    hook_point: str = "mlp",
    device: str | None = None,
    dtype: str | torch.dtype = "bfloat16",
) -> pd.DataFrame:
    """Factorial intervention test — ABSOLUTE downstream detector AUC per condition.

    Two intervention sites, each with a real and a magnitude-matched random option:
      * onset l*:   none / src (64 |Z_proj|) / randS (64 random, mag-matched at l*)
      * downstream: none / path (Top-k_down FIS/layer) / randD (k_down random, mag-matched)

    Six reported conditions (mean downstream AUC each):
      * **clean**        (none, none)          — original performance
      * **src**          (src,  none)          — zero source only
      * **path**         (none, path)          — zero downstream path only
      * **src+path**     (src,  path)          — full pathway
      * **src+randD**    (src,  randD)         — control: is the DOWNSTREAM selection special?
      * **randS+path**   (randS, path)         — control: is the ONSET selection special?

    Magnitude-matched randomization (paper's random-suppression control) rules out
    "the drop is just from removing high-variance neurons." A real source: src < randS+...;
    a real path: src+path < src+randD. Partial circularity remains where a detector
    reads a zeroed layer — hence both random controls.
    """
    device = device or get_device()
    torch_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype

    all_layers = sorted(int(l) for l in sae_bundle.layers)
    down = [l for l in all_layers if l > l_star]
    if not down:
        raise ValueError(f"onset l*={l_star} has no downstream layers")
    src_idx = np.asarray(src_idx, dtype=np.int64)

    rng = np.random.default_rng(0)

    # per-layer downstream k: fixed int, or "elbow" (Kneedle on each layer's FIS)
    kper = {}
    for l in down:
        sub = fis_df[fis_df["layer"] == l]
        if isinstance(k_down, str) and k_down == "elbow":
            kper[int(l)] = elbow_k(np.sort(sub["fis"].to_numpy())[::-1])
        else:
            kper[int(l)] = int(k_down)

    # downstream path + magnitude-matched random per layer (from FIS f*mu band)
    path, randD = {}, {}
    for l in down:
        sub = fis_df[fis_df["layer"] == l]
        k = kper[int(l)]
        neur = sub["neuron"].to_numpy()
        mag = (sub["f"] * sub["mu"]).to_numpy()  # activation magnitude per neuron
        p = sub.nlargest(k, "fis")["neuron"].to_numpy()
        path[int(l)] = p
        pset = set(int(x) for x in p)
        thr = float(np.min(mag[np.isin(neur, list(pset))])) if len(pset) else 0.0
        pool = neur[(mag >= thr) & (~np.isin(neur, list(pset)))]
        if len(pool) < k:
            pool = neur[~np.isin(neur, list(pset))]
        randD[int(l)] = rng.choice(pool, size=min(k, len(pool)), replace=False)
    self_kper = kper  # exposed via df.attrs below

    best_feat = {}
    for l in down:
        s = auroc_df[(auroc_df["concept"] == concept_obj.name) & (auroc_df["layer"] == l)]
        if len(s):
            best_feat[l] = int(s.loc[s["auc"].idxmax(), "feature"])

    texts, spans = _build_example_texts([concept_obj])
    example_ids, labels, _ = spans[concept_obj.name]  # type: ignore[misc]
    is_sens = np.zeros(len(texts), dtype=bool)
    for e, l in zip(example_ids, labels):
        if l == 1:
            is_sens[int(e)] = True
    sens_t = torch.tensor(is_sens, device=device)

    model.to(device); model.eval()

    # ---- pre-pass: clean activation magnitude at l* (for mag-matched randS) ----
    # Skipped in light mode (the randS+path control is dropped to halve the cost,
    # used for the k_down sweep where we only need clean/src/path/src+randD).
    randS = None
    if not light:
        rd_lstar = MLPHook(_get_hook_module(model, l_star, hook_point), l_star, to_cpu=False)
        d_lstar = sae_bundle[l_star].W_dec.shape[0]
        cnt_l = torch.zeros(d_lstar, device=device); sum_l = torch.zeros(d_lstar, device=device); nt = 0.0
        base = 0
        for tb in batched(list(texts), batch_size):
            b = len(tb)
            enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}; attn = enc.get("attention_mask")
            gids = torch.arange(base, base + b, device=device); base += b
            model(**enc)
            a = rd_lstar.pop().float(); bb, s, d = a.shape
            mflat = attn.bool().reshape(-1) if attn is not None else torch.ones(bb*s, dtype=torch.bool, device=device)
            af = a.reshape(bb*s, d)[mflat]
            tl = torch.arange(bb, device=device).unsqueeze(1).expand(bb, s).reshape(-1)[mflat]
            ts = sens_t[gids[tl]]
            sa = af[ts]; act = (sa > 0).float()
            cnt_l += act.sum(0); sum_l += (sa * act).sum(0); nt += float(sa.shape[0])
        rd_lstar.remove()
        mag_l = ((sum_l / max(nt, 1.0)).cpu().numpy())  # f*mu at l* = WFS_dense
        thr_s = float(np.min(mag_l[src_idx])) if len(src_idx) else 0.0
        alln = np.arange(d_lstar)
        poolS = alln[(mag_l >= thr_s) & (~np.isin(alln, src_idx))]
        if len(poolS) < len(src_idx):
            poolS = alln[~np.isin(alln, src_idx)]
        randS = rng.choice(poolS, size=min(len(src_idx), len(poolS)), replace=False)

    # ---- main pass ----
    zero_hooks: dict[int, ZeroOutHook] = {l_star: ZeroOutHook(_get_hook_module(model, l_star, hook_point), src_idx)}
    for l in down:
        zero_hooks[l] = ZeroOutHook(_get_hook_module(model, l, hook_point), path[l])
    read_hooks = {l: MLPHook(_get_hook_module(model, l, hook_point), l, to_cpu=False) for l in down}

    coords_by_cond = {
        "clean":     {},
        "src":       {l_star: src_idx},
        "path":      {**path},
        "src+path":  {l_star: src_idx, **path},
        "src+randD": {l_star: src_idx, **randD},
    }
    if not light:
        coords_by_cond["randS+path"] = {l_star: randS, **path}
    scores = {c: {l: [] for l in down} for c in coords_by_cond}
    ys: list[int] = []

    base = 0
    for tb in batched(list(texts), batch_size):
        b = len(tb)
        enc = tokenizer(list(tb), return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}; attn = enc.get("attention_mask")
        ys.extend(int(is_sens[g]) for g in range(base, base + b)); base += b
        for cname, cmap in coords_by_cond.items():
            for l, h in zero_hooks.items():
                if l in cmap:
                    h.set_coords(cmap[l]); h.enabled = True
                else:
                    h.enabled = False
            model(**enc)
            for l in down:
                a = read_hooks[l].pop().float(); bb, s, d = a.shape
                mflat = attn.bool().reshape(-1) if attn is not None else torch.ones(bb*s, dtype=torch.bool, device=device)
                af = a.reshape(bb*s, d)[mflat]
                tl = torch.arange(bb, device=device).unsqueeze(1).expand(bb, s).reshape(-1)[mflat]
                pooled = _mean_pool(af, tl, bb)
                if l in best_feat:
                    sae = sae_bundle[l]; p = next(sae.parameters())
                    fc = sae.encode_dense(pooled.to(p.device, p.dtype)).float()[:, best_feat[l]]
                    scores[cname][l].append(fc.cpu().numpy())
        torch.cuda.empty_cache()

    for h in zero_hooks.values(): h.remove()
    for h in read_hooks.values(): h.remove()

    from sklearn.metrics import roc_auc_score
    y = np.asarray(ys)
    conds = list(coords_by_cond.keys())
    cols = ["layer"] + [f"auc_{c}" for c in conds]
    mean_k = float(np.mean(list(self_kper.values()))) if self_kper else 0.0
    rows = []
    if y.sum() == 0 or y.sum() == len(y):
        out = pd.DataFrame(columns=cols); out.attrs["mean_k"] = mean_k; return out
    for l in down:
        if l not in best_feat:
            continue
        try:
            row = {"layer": l}
            for c in conds:
                row[f"auc_{c}"] = round(float(roc_auc_score(y, np.concatenate(scores[c][l]))), 3)
        except ValueError:
            continue
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["mean_k"] = mean_k
    out.attrs["kper"] = self_kper
    return out


def _mean_pool(flat: torch.Tensor, tok_local: torch.Tensor, n_ex: int) -> torch.Tensor:
    """Mean-pool token rows ``flat`` [T,d] into [n_ex,d] by local example id."""
    d = flat.shape[1]
    out = torch.zeros(n_ex, d, device=flat.device, dtype=flat.dtype)
    cnt = torch.zeros(n_ex, device=flat.device)
    out.index_add_(0, tok_local, flat)
    cnt.index_add_(0, tok_local, torch.ones_like(tok_local, dtype=flat.dtype))
    return out / cnt.clamp_min(1.0).unsqueeze(1)
