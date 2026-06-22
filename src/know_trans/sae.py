"""OpenAI-style TopK Sparse Autoencoder (SAE) for know_trans.

This module implements a TopK SAE in the style of OpenAI's
"Scaling and evaluating sparse autoencoders" (Gao et al., 2024):

- The encoder is a single affine map followed by a hard ``TopK`` activation:
  only the ``k`` largest pre-activations per example survive, the rest are
  zeroed. This gives an exact, tunable L0 (= ``k``) instead of relying on an
  L1 penalty.
- The decoder is a single linear map whose **columns are kept unit-norm**.
  Unit-norm decoder directions make feature magnitudes comparable and keep the
  optimisation well-conditioned (a feature cannot shrink its decoder column to
  cheat the reconstruction loss).
- **AuxK** dead-feature revival: features that have not fired for a while are
  "dead". An auxiliary reconstruction term forces the top dead features to
  explain the residual error, which revives them instead of letting capacity
  rot.
- We track the **dead-feature count** and the realised **L0** for logging.

The module is pure PyTorch (plus ``safetensors`` for checkpointing). It never
runs training or downloads at import time — heavy work lives behind functions.

Public API (see ``docs/INTERFACE_SPEC.md``):

- ``class TopKSAE(nn.Module)``
- ``def train_sae(reader, layer, cfg, seed, out_path) -> TopKSAE``
- ``class SAEBundle``
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

from .config import SAECfg
from .utils import ensure_dir, get_device, get_logger, set_seed

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids hard import cycle
    from .capture import ActivationReader


__all__ = ["TopKSAE", "train_sae", "SAEBundle"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _topk_activation(pre_acts: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the TopK activation to a batch of pre-activations.

    Keeps the ``k`` largest values per row (after a ReLU so that only positive
    activations can be selected, matching the OpenAI formulation) and returns
    them together with their feature indices.

    Args:
        pre_acts: Pre-activation tensor of shape ``[B, d_hidden]``.
        k: Number of features to keep per example.

    Returns:
        A tuple ``(values, indices)`` each of shape ``[B, k]``. ``values`` are
        the (ReLU'd) top-k activations; ``indices`` are the corresponding
        feature ids into ``[0, d_hidden)``.
    """
    # ReLU before selecting so negative pre-acts never enter the sparse code.
    relu_acts = F.relu(pre_acts)
    k = min(k, relu_acts.shape[-1])
    values, indices = torch.topk(relu_acts, k=k, dim=-1, sorted=False)
    return values, indices


def _scatter_dense(
    values: torch.Tensor, indices: torch.Tensor, d_hidden: int
) -> torch.Tensor:
    """Scatter sparse ``(values, indices)`` back into a dense ``[B, d_hidden]`` tensor.

    Args:
        values: Sparse activation values, shape ``[B, k]``.
        indices: Feature indices, shape ``[B, k]``.
        d_hidden: Width of the dense feature dimension.

    Returns:
        Dense code tensor of shape ``[B, d_hidden]`` with zeros everywhere
        except the selected features.
    """
    dense = values.new_zeros(values.shape[0], d_hidden)
    dense.scatter_(dim=-1, index=indices, src=values)
    return dense


# --------------------------------------------------------------------------- #
# TopK SAE
# --------------------------------------------------------------------------- #
class TopKSAE(nn.Module):
    """OpenAI-style TopK sparse autoencoder.

    The forward computation is::

        z          = x - b_pre                      # center the input
        pre_acts   = W_enc @ z + b_enc              # encoder pre-activations
        values, ix = TopK(ReLU(pre_acts), k)        # sparse code
        recon      = W_dec @ sparse(values, ix) + b_pre

    where ``W_dec`` has unit-norm columns (one column per feature). Subtracting
    a learnable pre-bias ``b_pre`` before encoding and adding it back after
    decoding ties the input/output offsets, which empirically improves
    reconstruction.

    Attributes:
        d_in: Input/activation dimensionality.
        d_hidden: Number of dictionary features (``expansion * d_in``).
        k: Number of active features per example (the L0 of every code).
        W_enc: Encoder weight, shape ``[d_hidden, d_in]``.
        b_enc: Encoder bias, shape ``[d_hidden]``.
        W_dec: Decoder weight, shape ``[d_in, d_hidden]`` (unit-norm columns).
        b_pre: Pre-encoder / post-decoder bias, shape ``[d_in]``.
    """

    def __init__(self, d_in: int, d_hidden: int, k: int) -> None:
        """Initialise a TopK SAE.

        The decoder is initialised as the transpose of the encoder (a common,
        well-behaved init) and its columns are then normalised to unit norm.
        ``b_pre`` is initialised to zero; callers may overwrite it with the
        data mean before training (see :func:`train_sae`).

        Args:
            d_in: Activation dimensionality of the data being modelled.
            d_hidden: Dictionary size (number of features).
            k: Number of nonzero entries per code.

        Raises:
            ValueError: If any dimension is non-positive or ``k > d_hidden``.
        """
        super().__init__()
        if d_in <= 0 or d_hidden <= 0 or k <= 0:
            raise ValueError(
                f"d_in, d_hidden, k must be positive; got {d_in}, {d_hidden}, {k}"
            )
        if k > d_hidden:
            raise ValueError(f"k ({k}) cannot exceed d_hidden ({d_hidden})")

        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.k = int(k)

        # Encoder: pre_acts = W_enc @ (x - b_pre) + b_enc
        self.W_enc = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        # Decoder: recon = W_dec @ code + b_pre   (columns are features)
        self.W_dec = nn.Parameter(torch.empty(d_in, d_hidden))
        # Shared pre-bias subtracted before encode and added after decode.
        self.b_pre = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialisation / invariants
    # ------------------------------------------------------------------ #
    def _reset_parameters(self) -> None:
        """Initialise weights and enforce unit-norm decoder columns."""
        # Kaiming-style init for the encoder.
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        # Tie the decoder to the encoder transpose, then normalise columns.
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t().contiguous())
            self.set_decoder_unit_norm()

    @torch.no_grad()
    def set_decoder_unit_norm(self, eps: float = 1e-8) -> None:
        """Rescale each decoder column (feature direction) to unit L2 norm.

        Call this after every optimiser step so the decoder columns stay on the
        unit sphere. Operating in-place on the parameter keeps it a leaf so the
        optimiser state remains valid.

        Args:
            eps: Numerical floor to avoid division by zero for dead columns.
        """
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(eps)
        self.W_dec.div_(norms)

    @torch.no_grad()
    def remove_parallel_decoder_grad(self) -> None:
        """Project out the component of ``W_dec.grad`` parallel to its columns.

        Renormalising the decoder after a step (see
        :meth:`set_decoder_unit_norm`) silently discards any gradient component
        that points along a column. Removing that component *before* the step
        keeps Adam's running moments consistent with the constrained geometry,
        as recommended in the OpenAI SAE work. Safe to call only when grads
        exist.
        """
        if self.W_dec.grad is None:
            return
        # Component of the gradient along each (unit) column.
        parallel = (self.W_dec.grad * self.W_dec).sum(dim=0, keepdim=True)
        self.W_dec.grad.sub_(parallel * self.W_dec)

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def preacts(self, x: torch.Tensor) -> torch.Tensor:
        """Compute encoder pre-activations (before TopK).

        Args:
            x: Input activations, shape ``[B, d_in]``.

        Returns:
            Pre-activation tensor of shape ``[B, d_hidden]``.
        """
        return F.linear(x - self.b_pre, self.W_enc, self.b_enc)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch into its sparse TopK code.

        Args:
            x: Input activations, shape ``[B, d_in]``.

        Returns:
            ``(values, indices)`` each of shape ``[B, k]``: the nonzero code
            values and the feature indices they belong to.
        """
        pre = self.preacts(x)
        return _topk_activation(pre, self.k)

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch into a dense (mostly zero) code tensor.

        Convenience wrapper used by downstream feature-scoring code that wants
        a ``[B, d_hidden]`` matrix rather than the sparse representation.

        Args:
            x: Input activations, shape ``[B, d_in]``.

        Returns:
            Dense code tensor of shape ``[B, d_hidden]`` (only ``k`` nonzero
            entries per row).
        """
        values, indices = self.encode(x)
        return _scatter_dense(values, indices, self.d_hidden)

    def decode(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Decode a sparse code back into the input space.

        Args:
            values: Sparse code values, shape ``[B, k]``.
            indices: Feature indices, shape ``[B, k]``.

        Returns:
            Reconstruction tensor of shape ``[B, d_in]``.
        """
        # Gather the active decoder columns: W_dec[:, indices] -> [B, d_in, k].
        # einsum keeps this differentiable and memory-light.
        dec_cols = self.W_dec[:, indices]  # [d_in, B, k]
        recon = torch.einsum("dbk,bk->bd", dec_cols, values)
        return recon + self.b_pre

    def decode_dense(self, codes_dense: torch.Tensor) -> torch.Tensor:
        """Decode a dense code tensor back into the input space.

        Args:
            codes_dense: Dense codes, shape ``[B, d_hidden]``.

        Returns:
            Reconstruction tensor of shape ``[B, d_in]``.
        """
        return F.linear(codes_dense, self.W_dec) + self.b_pre

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the full autoencoder.

        Args:
            x: Input activations, shape ``[B, d_in]``.

        Returns:
            Dict with keys:

            - ``recon``: reconstruction ``[B, d_in]``
            - ``codes_dense``: dense code ``[B, d_hidden]``
            - ``values``: sparse code values ``[B, k]``
            - ``indices``: sparse code indices ``[B, k]``
        """
        values, indices = self.encode(x)
        recon = self.decode(values, indices)
        codes_dense = _scatter_dense(values, indices, self.d_hidden)
        return {
            "recon": recon,
            "codes_dense": codes_dense,
            "values": values,
            "indices": indices,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _config_dict(self) -> Dict[str, int]:
        """Return the constructor hyper-parameters as a JSON-serialisable dict."""
        return {"d_in": self.d_in, "d_hidden": self.d_hidden, "k": self.k}

    def save(self, path: str) -> None:
        """Serialise the SAE to ``path`` (safetensors) plus a JSON sidecar.

        Two files are written:

        - ``path`` — a ``.safetensors`` file holding the parameters.
        - ``<path>.config.json`` — the constructor hyper-parameters, so
          :meth:`load` can rebuild the module without external metadata.

        The architecture config is also embedded in the safetensors metadata
        header for redundancy.

        Args:
            path: Destination ``.safetensors`` path. Parent dirs are created.
        """
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        cfg = self._config_dict()
        # Store all tensors on CPU in their training dtype.
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        metadata = {k: str(v) for k, v in cfg.items()}
        safetensors_save(state, path, metadata=metadata)
        with open(f"{path}.config.json", "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TopKSAE":
        """Load a SAE previously written by :meth:`save`.

        The architecture is recovered from the JSON sidecar if present,
        otherwise from the safetensors metadata header.

        Args:
            path: Path to the ``.safetensors`` file.
            device: Optional device string to map the parameters onto. Defaults
                to the value returned by :func:`know_trans.utils.get_device`.

        Returns:
            A :class:`TopKSAE` with the loaded weights, in eval mode.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the architecture config cannot be recovered.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SAE checkpoint not found: {path}")
        device = device or get_device()

        cfg: Optional[Dict[str, Any]] = None
        sidecar = f"{path}.config.json"
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)

        state = safetensors_load(path, device="cpu")
        if cfg is None:
            # Fall back to inferring dims from the tensor shapes.
            if "W_dec" not in state:
                raise ValueError(
                    f"Cannot recover SAE architecture from {path}: no config sidecar "
                    "and missing W_dec tensor."
                )
            d_in, d_hidden = state["W_dec"].shape
            # k is not derivable from weights; require the sidecar/metadata.
            raise ValueError(
                f"Missing config sidecar {sidecar}; cannot recover k "
                f"(inferred d_in={d_in}, d_hidden={d_hidden})."
            )

        sae = cls(d_in=int(cfg["d_in"]), d_hidden=int(cfg["d_hidden"]), k=int(cfg["k"]))
        sae.load_state_dict(state)
        sae.to(device)
        sae.eval()
        return sae


# --------------------------------------------------------------------------- #
# Dead-feature tracking
# --------------------------------------------------------------------------- #
class _DeadFeatureTracker:
    """Tracks how many training steps each feature has gone without firing.

    A feature is considered "dead" once it has not appeared in any TopK code
    for at least ``dead_steps`` consecutive optimisation steps. The tracker is
    a plain counter buffer kept on the model's device; it is not part of the
    saved checkpoint.
    """

    def __init__(self, d_hidden: int, dead_steps: int, device: torch.device) -> None:
        """Initialise the tracker.

        Args:
            d_hidden: Number of features to track.
            dead_steps: Steps without firing before a feature counts as dead.
            device: Device on which to keep the counter.
        """
        self.dead_steps = int(dead_steps)
        # steps_since_fire[i] = number of steps since feature i last fired.
        self.steps_since_fire = torch.zeros(d_hidden, dtype=torch.long, device=device)

    def update(self, indices: torch.Tensor) -> None:
        """Update counters given the active feature indices of a batch.

        Args:
            indices: Active feature indices for the current batch, any shape;
                values index into ``[0, d_hidden)``.
        """
        self.steps_since_fire += 1
        fired = torch.unique(indices)
        self.steps_since_fire[fired] = 0

    def dead_mask(self) -> torch.Tensor:
        """Return a boolean mask of currently-dead features (shape ``[d_hidden]``)."""
        return self.steps_since_fire >= self.dead_steps

    def num_dead(self) -> int:
        """Return the current dead-feature count."""
        return int(self.dead_mask().sum().item())


# --------------------------------------------------------------------------- #
# Streaming batch iterator over an ActivationReader
# --------------------------------------------------------------------------- #
def _stream_activation_batches(
    reader: "ActivationReader",
    layer: int,
    batch_size: int,
    steps: int,
    device: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Iterator[torch.Tensor]:
    """Yield shuffled mini-batches of activations for one layer.

    The layer's activations are read once (the reader is responsible for not
    holding *all* layers in RAM at once; we hold a single layer's rows). We
    then sample ``batch_size`` rows with a reshuffled permutation each epoch,
    yielding exactly ``steps`` batches in total. This keeps the optimiser fed
    regardless of how the data shards are laid out on disk.

    Args:
        reader: An ``ActivationReader`` pointing at a capture directory.
        layer: Layer index to train on.
        batch_size: Rows per yielded batch.
        steps: Total number of batches to yield.
        device: Device to move batches onto.
        seed: RNG seed for the shuffling permutation.
        dtype: Dtype to cast the (typically float16) activations to for
            training. Float32 is recommended for stable SAE optimisation.

    Yields:
        Float tensors of shape ``[batch_size, d_in]`` on ``device``.
    """
    acts, _index = reader.read(layer)  # acts: Tensor[T, d_in] (likely float16, CPU)
    n = acts.shape[0]
    if n == 0:
        raise ValueError(f"No activations found for layer {layer}.")

    # Loader fix: keep the WHOLE layer tensor resident on the GPU so each step's
    # minibatch is an on-device gather (no CPU gather + host->device copy per
    # step, which was the ~4 it/s bottleneck). Falls back to CPU residence when
    # the tensor would not comfortably fit (leaving headroom for SAE + batches).
    dev = torch.device(device)
    if dev.type == "cuda":
        need = acts.numel() * acts.element_size()
        try:
            free, _total = torch.cuda.mem_get_info(dev)
        except Exception:
            free = 0
        if 0 < need < free * 0.6:
            acts = acts.to(dev)

    gen = torch.Generator()
    gen.manual_seed(int(seed))

    perm = torch.randperm(n, generator=gen).to(acts.device)
    cursor = 0
    for _ in range(steps):
        if cursor + batch_size > n:
            # Reshuffle for the next "epoch" and restart the cursor.
            perm = torch.randperm(n, generator=gen).to(acts.device)
            cursor = 0
        idx = perm[cursor : cursor + batch_size]
        cursor += batch_size
        batch = acts[idx].to(dtype=dtype)        # gather on acts.device
        if batch.device != dev:                  # CPU-resident fallback
            batch = batch.to(dev)
        yield batch


def _estimate_pre_bias(
    reader: "ActivationReader",
    layer: int,
    device: str,
    dtype: torch.dtype,
    max_rows: int = 200_000,
) -> torch.Tensor:
    """Estimate the data mean to initialise ``b_pre``.

    Initialising the pre-bias to the activation mean makes the very first
    reconstruction sensible (predict the mean) and speeds up convergence.

    Args:
        reader: Activation reader.
        layer: Layer index.
        device: Device for the returned tensor.
        dtype: Dtype for the returned tensor.
        max_rows: Cap on rows used for the estimate (keeps it cheap on huge
            captures).

    Returns:
        Mean activation vector of shape ``[d_in]``.
    """
    acts, _ = reader.read(layer)
    rows = min(acts.shape[0], max_rows)
    mean = acts[:rows].to(torch.float32).mean(dim=0)
    return mean.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_sae(
    reader: "ActivationReader",
    layer: int,
    cfg: SAECfg,
    seed: int,
    out_path: str,
) -> TopKSAE:
    """Train a TopK SAE on one layer's captured activations.

    The objective is the standard TopK-SAE loss::

        L = ||x - recon||^2  +  aux_coef * ||residual - aux_recon||^2

    where ``residual = x - recon`` and ``aux_recon`` reconstructs that residual
    using only the **top ``aux_k`` dead features** (AuxK). The AuxK term is what
    revives dead features: it gives them a gradient signal pointing at whatever
    the live features currently fail to explain. When no features are dead the
    AuxK term is zero and the loss reduces to plain reconstruction MSE.

    Decoder columns are projected to unit norm after every step (with the
    parallel gradient component removed beforehand). Dead-feature count and
    realised L0 are logged periodically.

    The data dimensionality ``d_in`` is **inferred** from the activation shards
    and ``d_hidden = cfg.expansion * d_in`` — nothing is hardcoded.

    Args:
        reader: An ``ActivationReader`` for the capture directory.
        layer: Layer index to train on (must be in ``reader.layers``).
        cfg: SAE hyper-parameters (expansion, k, lr, batch_size, steps,
            aux_k, aux_coef, activation).
        seed: Random seed for reproducibility.
        out_path: Path to write the trained SAE (``.safetensors``).

    Returns:
        The trained :class:`TopKSAE` (also persisted to ``out_path``).

    Raises:
        ValueError: If ``cfg.activation`` is not ``"topk"``, or the layer has
            no data.
    """
    if cfg.activation != "topk":
        raise ValueError(
            f"sae.train_sae only implements the TopK activation; "
            f"got cfg.activation={cfg.activation!r}."
        )

    logger = get_logger("know_trans.sae")
    set_seed(seed)
    device = get_device()
    train_dtype = torch.float32  # SAE training is most stable in fp32.

    # ---- Infer dimensions from the data ----------------------------------- #
    acts, _ = reader.read(layer)
    d_in = int(acts.shape[-1])
    n_tokens = int(acts.shape[0])
    # Absolute d_hidden (shared across models) takes precedence over expansion.
    d_hidden = (
        int(cfg.d_hidden) if getattr(cfg, "d_hidden", None)
        else int(cfg.expansion) * d_in
    )
    k = int(cfg.k)
    aux_k = int(cfg.aux_k)
    del acts  # release reference; the streamer re-reads as needed

    logger.info(
        "train_sae: layer=%d d_in=%d d_hidden=%d k=%d aux_k=%d "
        "n_tokens=%d steps=%d batch=%d lr=%g",
        layer, d_in, d_hidden, k, aux_k, n_tokens, cfg.steps, cfg.batch_size, cfg.lr,
    )

    # ---- Build model ------------------------------------------------------ #
    sae = TopKSAE(d_in=d_in, d_hidden=d_hidden, k=k).to(device=device, dtype=train_dtype)
    # Initialise pre-bias to the data mean for a sensible starting point.
    with torch.no_grad():
        sae.b_pre.copy_(_estimate_pre_bias(reader, layer, device, train_dtype))

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=6.25e-10)

    # Dead after ~10 epochs' worth of steps (capped) without firing.
    dead_steps = max(1, min(cfg.steps, 10_000))
    dead_tracker = _DeadFeatureTracker(d_hidden, dead_steps=dead_steps, device=torch.device(device))

    log_every = max(1, cfg.steps // 100)

    # ---- Training loop ---------------------------------------------------- #
    sae.train()
    running_mse = 0.0
    running_aux = 0.0
    running_l0 = 0.0
    t0 = time.time()

    batches = _stream_activation_batches(
        reader, layer, cfg.batch_size, cfg.steps, device, seed, dtype=train_dtype
    )
    for step, x in enumerate(batches):
        out = sae(x)
        recon = out["recon"]

        # Primary reconstruction MSE (mean over batch and features).
        mse = F.mse_loss(recon, x)

        # ---- AuxK dead-feature revival ----------------------------------- #
        residual = (x - recon).detach()  # what live features failed to explain
        dead_mask = dead_tracker.dead_mask()
        aux_loss = recon.new_zeros(())
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0 and aux_k > 0 and cfg.aux_coef > 0:
            pre = sae.preacts(x)  # [B, d_hidden]
            # Restrict to dead features only.
            dead_pre = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
            # ReLU then pick the top (aux_k) dead features per example.
            dead_relu = F.relu(dead_pre)
            kk = min(aux_k, n_dead)
            aux_vals, aux_idx = torch.topk(dead_relu, k=kk, dim=-1, sorted=False)
            # Guard: -inf entries became 0 after ReLU; that is fine (no grad).
            aux_recon = sae.decode(aux_vals, aux_idx) - sae.b_pre  # model the residual
            aux_loss = F.mse_loss(aux_recon, residual)

        loss = mse + float(cfg.aux_coef) * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Keep decoder on the unit sphere: drop parallel grad, step, renorm.
        sae.remove_parallel_decoder_grad()
        optimizer.step()
        sae.set_decoder_unit_norm()

        # ---- Tracking ---------------------------------------------------- #
        dead_tracker.update(out["indices"])
        # Realised L0 = mean number of strictly-positive code entries per row.
        l0 = (out["values"] > 0).float().sum(dim=-1).mean().item()

        running_mse += mse.item()
        running_aux += float(aux_loss.item())
        running_l0 += l0

        if (step + 1) % log_every == 0 or step == cfg.steps - 1:
            denom = log_every if (step + 1) % log_every == 0 else ((step % log_every) + 1)
            elapsed = time.time() - t0
            logger.info(
                "step %6d/%d | mse=%.5f aux=%.5f L0=%.1f dead=%d/%d (%.1f%%) | %.1f it/s",
                step + 1, cfg.steps,
                running_mse / denom, running_aux / denom, running_l0 / denom,
                dead_tracker.num_dead(), d_hidden,
                100.0 * dead_tracker.num_dead() / d_hidden,
                (step + 1) / max(elapsed, 1e-9),
            )
            running_mse = running_aux = running_l0 = 0.0

    # ---- Persist ---------------------------------------------------------- #
    sae.eval()
    sae.save(out_path)

    # Sidecar training summary (small json, alongside the checkpoint).
    summary = {
        "layer": int(layer),
        "d_in": d_in,
        "d_hidden": d_hidden,
        "k": k,
        "aux_k": aux_k,
        "aux_coef": float(cfg.aux_coef),
        "steps": int(cfg.steps),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "seed": int(seed),
        "n_tokens": n_tokens,
        "final_dead": dead_tracker.num_dead(),
        "final_dead_frac": dead_tracker.num_dead() / d_hidden,
        "train_seconds": time.time() - t0,
    }
    with open(f"{out_path}.train.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        "train_sae done: saved %s | final dead=%d/%d (%.1f%%)",
        out_path, summary["final_dead"], d_hidden, 100.0 * summary["final_dead_frac"],
    )
    return sae


# --------------------------------------------------------------------------- #
# SAE bundle (a directory of per-layer SAEs)
# --------------------------------------------------------------------------- #
class SAEBundle:
    """A collection of trained SAEs, one per layer, loaded from a directory.

    The bundle expects checkpoints named ``layer{L}.safetensors`` inside the
    directory (the convention written by the CLI / training scripts). SAEs are
    loaded lazily on first access and cached.

    Example::

        bundle = SAEBundle.load("data/saes/teacher")
        sae = bundle[12]            # TopKSAE for layer 12
        print(bundle.layers)        # [0, 4, 8, 12, ...]

    Attributes:
        dir: Source directory.
        layers: Sorted list of available layer indices.
    """

    _FILE_GLOB = "layer*.safetensors"

    def __init__(self, directory: str, paths: Dict[int, str], device: Optional[str] = None) -> None:
        """Construct a bundle (prefer :meth:`load`).

        Args:
            directory: Directory holding the per-layer checkpoints.
            paths: Mapping of layer index -> checkpoint path.
            device: Device to load SAEs onto (defaults to :func:`get_device`).
        """
        self.dir = directory
        self._paths: Dict[int, str] = dict(paths)
        self._device = device or get_device()
        self._cache: Dict[int, TopKSAE] = {}

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "SAEBundle":
        """Discover and index per-layer SAE checkpoints in ``directory``.

        Does **not** load the weights yet — individual SAEs are materialised on
        first ``__getitem__`` access. Filenames must match
        ``layer{L}.safetensors``.

        Args:
            directory: Directory to scan.
            device: Device for subsequently-loaded SAEs.

        Returns:
            A :class:`SAEBundle`.

        Raises:
            FileNotFoundError: If the directory does not exist or contains no
                matching checkpoints.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"SAE bundle directory not found: {directory}")
        paths: Dict[int, str] = {}
        for fp in sorted(glob.glob(os.path.join(directory, cls._FILE_GLOB))):
            base = os.path.basename(fp)
            # Strip 'layer' prefix and '.safetensors' suffix -> integer layer id.
            stem = base[len("layer") : -len(".safetensors")]
            try:
                layer = int(stem)
            except ValueError:
                continue  # skip files that don't follow the convention
            paths[layer] = fp
        if not paths:
            raise FileNotFoundError(
                f"No '{cls._FILE_GLOB}' checkpoints found in {directory}."
            )
        return cls(directory, paths, device=device)

    @property
    def layers(self) -> List[int]:
        """Sorted list of layer indices available in the bundle."""
        return sorted(self._paths.keys())

    def __contains__(self, layer: int) -> bool:
        """Return whether ``layer`` has a checkpoint in the bundle."""
        return int(layer) in self._paths

    def __len__(self) -> int:
        """Return the number of layers in the bundle."""
        return len(self._paths)

    def __iter__(self) -> Iterator[int]:
        """Iterate over available layer indices in sorted order."""
        return iter(self.layers)

    def __getitem__(self, layer: int) -> TopKSAE:
        """Return the (lazily loaded, cached) SAE for ``layer``.

        Args:
            layer: Layer index.

        Returns:
            The :class:`TopKSAE` for that layer.

        Raises:
            KeyError: If ``layer`` is not present in the bundle.
        """
        layer = int(layer)
        if layer not in self._paths:
            raise KeyError(
                f"Layer {layer} not in bundle (available: {self.layers})."
            )
        if layer not in self._cache:
            self._cache[layer] = TopKSAE.load(self._paths[layer], device=self._device)
        return self._cache[layer]

    def items(self) -> Iterator[Tuple[int, TopKSAE]]:
        """Iterate over ``(layer, TopKSAE)`` pairs, loading each on demand."""
        for layer in self.layers:
            yield layer, self[layer]
