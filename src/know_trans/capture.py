"""Activation capture for know_trans.

This module records per-layer MLP activations from a causal language model and
persists them to disk in a streaming fashion (we never hold the full activation
tensor for the whole corpus in RAM).  The artifacts written for a corpus are::

    out_dir/layer{L}.safetensors   # key "acts": float16 [T, d_model]
    out_dir/index.parquet          # cols: row, example_id, tok_idx, char_start, char_end
    out_dir/meta.json              # {model, d_model, layers, n_examples, n_tokens, hook_point}

The ``index.parquet`` rows align 1:1 with the rows of every ``layer{L}.safetensors``
tensor (same ``T`` and ordering for all captured layers).  Each row corresponds to a
single *token* of a single *example* (text).  Because tokenizers differ across model
families, downstream code must never align by token index across models — instead it
pools tokens to the example/span level using ``example_id`` and the ``char_start`` /
``char_end`` character spans recorded here (filled via ``return_offsets_mapping=True``).

Public API (see ``docs/INTERFACE_SPEC.md``):
    - ``MLPHook``               forward hook on ``model.model.layers[i].mlp`` output.
    - ``capture_activations``   run the model over texts and write the artifacts.
    - ``ActivationReader``      read the artifacts back (``.layers``, ``.meta``, ``.read``).
"""

from __future__ import annotations

import glob
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

# Heavy/optional deps are imported lazily inside functions where practical so that
# ``import know_trans.capture`` stays cheap and side-effect free.  We import the
# project helpers at module scope because they are lightweight and define the
# canonical names (get_logger, ensure_dir, batched, save_json, load_json).
from .utils import batched, ensure_dir, get_logger, load_json, save_json

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing transformers eagerly
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Forward hook
# ---------------------------------------------------------------------------
class MLPHook:
    """Forward hook that captures the output of one layer's MLP block.

    The hook is attached to ``model.model.layers[i].mlp`` and stores the most
    recent forward output in :attr:`activation`.  It does **not** detach to CPU
    automatically beyond what the caller requests; :meth:`pop` returns the stored
    tensor (moved to CPU / cast as configured) and clears the buffer so a single
    hook instance can be reused across batches without leaking GPU memory.

    Parameters
    ----------
    module:
        The ``nn.Module`` to hook (typically ``model.model.layers[i].mlp``).
    layer:
        Index of the layer this hook corresponds to (used only for bookkeeping).
    to_cpu:
        If ``True`` (default) the captured activation is moved to CPU immediately
        in the hook callback, keeping GPU memory bounded during long captures.
    dtype:
        Optional torch dtype to cast the captured activation to (e.g.
        ``torch.float16``).  If ``None`` the model's native dtype is kept.
    """

    def __init__(
        self,
        module: "torch.nn.Module",
        layer: int,
        to_cpu: bool = True,
        dtype: Optional[torch.dtype] = None,
        capture_input: bool = False,
    ) -> None:
        self.layer = layer
        self.to_cpu = to_cpu
        self.dtype = dtype
        # When True, capture the module's INPUT (inputs[0]) instead of its output.
        # Used to hook down_proj's input = the 14336-dim SwiGLU intermediate
        # (the MLP "neuron" activations), rather than the 4096-dim MLP output.
        self.capture_input = capture_input
        self.activation: Optional[torch.Tensor] = None
        # register_forward_hook returns a handle we keep so we can remove it later.
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module: "torch.nn.Module", inputs: Any, output: Any) -> None:
        """Forward-hook callback. Captures the module output, or its input when
        ``capture_input`` is set (e.g. down_proj's input = MLP intermediate)."""
        if self.capture_input:
            src = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        elif isinstance(output, (tuple, list)):
            src = output[0]
        else:
            src = output
        out = src.detach()
        if self.dtype is not None:
            out = out.to(self.dtype)
        if self.to_cpu:
            out = out.to("cpu")
        self.activation = out

    def pop(self) -> torch.Tensor:
        """Return the captured activation and clear the internal buffer.

        Returns
        -------
        torch.Tensor
            The activation of shape ``[batch, seq_len, d_model]`` from the most
            recent forward pass.

        Raises
        ------
        RuntimeError
            If no activation has been captured since the last :meth:`pop`.
        """
        if self.activation is None:
            raise RuntimeError(
                f"MLPHook(layer={self.layer}) has no captured activation; "
                "did a forward pass run since the last pop()?"
            )
        act = self.activation
        self.activation = None
        return act

    def remove(self) -> None:
        """Remove the underlying forward hook (idempotent)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "MLPHook":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_layers(model: "PreTrainedModel", layers: Sequence[int] | str) -> List[int]:
    """Resolve a ``layers`` spec (``"all"`` or explicit indices) to a sorted list.

    Parameters
    ----------
    model:
        The loaded model; used to discover the number of decoder layers.
    layers:
        Either the string ``"all"`` or an iterable of integer layer indices.

    Returns
    -------
    list[int]
        Sorted, de-duplicated, validated layer indices.
    """
    decoder = _get_decoder_layers(model)
    n_layers = len(decoder)
    if isinstance(layers, str):
        if layers.strip().lower() != "all":
            raise ValueError(f"layers string must be 'all', got {layers!r}")
        return list(range(n_layers))
    resolved = sorted({int(i) for i in layers})
    for i in resolved:
        if i < 0 or i >= n_layers:
            raise ValueError(
                f"layer index {i} out of range for model with {n_layers} layers"
            )
    return resolved


def _get_decoder_layers(model: "PreTrainedModel") -> "torch.nn.ModuleList":
    """Return the ``ModuleList`` of decoder blocks (``model.model.layers``).

    Handles the common HF causal-LM layout where the backbone is under
    ``model.model`` (Llama, Qwen, Mistral, ...).  Falls back to ``model.layers``
    if the model *is* the backbone.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        "Could not locate decoder layers; expected model.model.layers or model.layers"
    )


def _get_hook_module(
    model: "PreTrainedModel", layer_idx: int, hook_point: str
) -> "torch.nn.Module":
    """Return the submodule to hook for ``layer_idx`` and ``hook_point``.

    Currently only ``hook_point == "mlp"`` is supported (the spec default), which
    hooks ``model.model.layers[i].mlp``.  The argument is kept explicit so the
    function can be extended (e.g. to ``"attn"`` or the residual stream) without
    changing the public signature of :func:`capture_activations`.
    """
    block = _get_decoder_layers(model)[layer_idx]
    if hook_point == "mlp":
        if not hasattr(block, "mlp"):
            raise AttributeError(
                f"layer {layer_idx} has no .mlp submodule for hook_point='mlp'"
            )
        return block.mlp
    if hook_point == "down_proj_in":
        # Hook the SECOND FF matrix; we capture its INPUT = the post-SwiGLU
        # intermediate activation (intermediate_size dim = the MLP "neurons"/
        # key-value memory coefficients). See MLPHook(capture_input=True).
        if not (hasattr(block, "mlp") and hasattr(block.mlp, "down_proj")):
            raise AttributeError(
                f"layer {layer_idx} has no .mlp.down_proj for hook_point='down_proj_in'"
            )
        return block.mlp.down_proj
    raise ValueError(
        f"Unsupported hook_point {hook_point!r}; expected 'mlp' or 'down_proj_in'."
    )


# Hook points where MLPHook should capture the module INPUT rather than output.
_INPUT_HOOK_POINTS = {"down_proj_in"}


def _infer_capture_dim(model: "PreTrainedModel", hook_point: str) -> int:
    """Width of the activation captured at ``hook_point`` (for meta.json)."""
    if hook_point == "down_proj_in":
        cfg = getattr(model, "config", None)
        for attr in ("intermediate_size", "ffn_dim", "n_inner"):
            val = getattr(cfg, attr, None) if cfg is not None else None
            if val is not None:
                return int(val)
    return _infer_d_model(model)


def _infer_d_model(model: "PreTrainedModel") -> int:
    """Best-effort inference of ``d_model`` from the model config (never hardcoded)."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
    raise AttributeError("Could not infer d_model from model.config")


def _model_name(model: "PreTrainedModel") -> str:
    """Return a human-readable model identifier for ``meta.json``."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        if name:
            return str(name)
        mt = getattr(cfg, "model_type", None)
        if mt:
            return str(mt)
    return type(model).__name__


def _shard_path(out_dir: str, layer: int, shard: int) -> str:
    """Path for a temporary per-batch shard of a given layer."""
    return os.path.join(out_dir, f".layer{layer}.shard{shard:06d}.safetensors")


def _final_layer_path(out_dir: str, layer: int) -> str:
    """Final consolidated safetensors path for a layer."""
    return os.path.join(out_dir, f"layer{layer}.safetensors")


def _concat_shards_to_final(out_dir: str, layer: int, n_shards: int) -> int:
    """Concatenate per-batch shards for ``layer`` into one ``layer{L}.safetensors``.

    Shards are streamed one at a time so peak memory is bounded by the largest
    single shard plus the growing output buffer.  We assemble the final tensor by
    loading each shard, appending to a list, and concatenating once — this still
    materializes the per-layer tensor in RAM at the end, which is acceptable for a
    single layer (a single layer's activations for the whole corpus are far smaller
    than holding *all* layers simultaneously).  Temporary shards are removed after a
    successful write.

    Returns
    -------
    int
        Total number of rows (tokens) written for this layer.
    """
    chunks: List[torch.Tensor] = []
    total_rows = 0
    for s in range(n_shards):
        path = _shard_path(out_dir, layer, s)
        tensor = load_file(path)["acts"]
        total_rows += int(tensor.shape[0])
        chunks.append(tensor)
    if chunks:
        acts = torch.cat(chunks, dim=0).contiguous()
    else:
        acts = torch.empty((0, 0), dtype=torch.float16)
    save_file({"acts": acts}, _final_layer_path(out_dir, layer))
    # Clean up shards only after the final file is safely written.
    for s in range(n_shards):
        try:
            os.remove(_shard_path(out_dir, layer, s))
        except OSError:  # pragma: no cover - best effort cleanup
            logger.warning("could not remove shard %s", _shard_path(out_dir, layer, s))
    return total_rows


def _cleanup_partial(out_dir: str, layers: Sequence[int]) -> None:
    """Remove any leftover temporary shard files (e.g. after a failure)."""
    for path in glob.glob(os.path.join(out_dir, ".layer*.shard*.safetensors")):
        try:
            os.remove(path)
        except OSError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Capture entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_activations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    texts: Sequence[str],
    layers: Sequence[int] | str,
    out_dir: str,
    batch_size: int = 16,
    max_len: int = 512,
    hook_point: str = "mlp",
    dtype: str = "float16",
) -> None:
    """Run ``model`` over ``texts`` and stream MLP activations to ``out_dir``.

    For every requested layer this writes one ``layer{L}.safetensors`` file with a
    single key ``"acts"`` of shape ``[T, d_model]`` (``float16``), where ``T`` is the
    total number of non-padding tokens across all texts.  A shared ``index.parquet``
    maps each row to its source example and character span, and ``meta.json`` records
    capture metadata.  All ``layer{L}.safetensors`` files share the same row ordering
    and row count, aligned with ``index.parquet``.

    Streaming behaviour
    -------------------
    Activations are processed batch-by-batch.  After each batch, every layer's
    (token-filtered) activation chunk is appended to a per-layer temporary shard on
    disk and dropped from memory.  Once all batches are processed, per-layer shards
    are concatenated into the final ``layer{L}.safetensors`` files.  At no point are
    activations for the entire corpus across all layers held in RAM simultaneously.

    Parameters
    ----------
    model:
        A loaded causal LM in eval mode.  (This function calls ``model.eval()``.)
    tokenizer:
        The matching fast tokenizer.  Must support ``return_offsets_mapping=True``
        (i.e. a "fast" tokenizer) so character spans can be recorded.
    texts:
        The corpus; one string per example.  ``example_id`` is the index into this
        sequence.
    layers:
        ``"all"`` or an iterable of decoder-layer indices to capture.
    out_dir:
        Output directory (created if needed).
    batch_size:
        Number of texts per forward pass.
    max_len:
        Max tokenized length per text (truncation applied).
    hook_point:
        Which submodule to hook.  Only ``"mlp"`` is currently implemented.
    dtype:
        On-disk activation dtype name (default ``"float16"``); cast at capture time.

    Returns
    -------
    None
        Artifacts are written to ``out_dir``.

    Raises
    ------
    ValueError
        If the tokenizer is not a fast tokenizer (offset mapping unavailable) or an
        unsupported ``hook_point`` / ``layers`` spec is given.
    """
    out_dir = ensure_dir(out_dir)
    texts = list(texts)
    n_examples = len(texts)

    # Resolve dtype for on-disk storage.
    storage_dtype = _resolve_storage_dtype(dtype)

    # Validate tokenizer supports offset mapping (required for char spans).
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "capture_activations requires a fast tokenizer "
            "(return_offsets_mapping=True is unavailable on slow tokenizers)."
        )

    layer_indices = _resolve_layers(model, layers)
    d_model = _infer_capture_dim(model, hook_point)
    device = next(model.parameters()).device

    model.eval()

    # Pad on the right so that token positions and offset mappings line up simply.
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        # Many causal LMs ship without a pad token; reuse EOS purely for batching.
        tokenizer.pad_token = tokenizer.eos_token

    # Attach hooks for the requested layers.
    hooks: Dict[int, MLPHook] = {}
    capture_input = hook_point in _INPUT_HOOK_POINTS
    for li in layer_indices:
        module = _get_hook_module(model, li, hook_point)
        hooks[li] = MLPHook(module, layer=li, to_cpu=True, dtype=storage_dtype,
                            capture_input=capture_input)

    # Per-layer shard counters and an accumulating index of token metadata.
    n_shards = 0
    index_records: List[Dict[str, int]] = []
    global_row = 0  # running row offset across the whole corpus
    n_tokens = 0

    try:
        for batch_texts, base_example_id in _iter_batches(texts, batch_size):
            enc = tokenizer(
                list(batch_texts),
                return_offsets_mapping=True,
                return_attention_mask=True,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            attn = enc["attention_mask"]  # [B, S] (1 for real tokens, 0 for pad)
            offsets = enc["offset_mapping"]  # [B, S, 2]
            input_ids = enc["input_ids"].to(device)
            attn_dev = attn.to(device)

            # Forward pass; hooks capture per-layer activations to CPU.
            model(input_ids=input_ids, attention_mask=attn_dev, use_cache=False)

            # Build a flat boolean mask of real (non-pad) tokens for this batch.
            # We also (defensively) drop "special" tokens whose offset span is
            # (0, 0) AND that are flagged by the tokenizer, since those carry no
            # character span. Padding always has attention_mask == 0.
            keep_mask = attn.bool()  # [B, S]

            # Record index rows for kept tokens (row, example_id, tok_idx, span).
            batch_kept = _append_index_records(
                index_records=index_records,
                attn=attn,
                offsets=offsets,
                base_example_id=base_example_id,
                start_row=global_row,
            )

            # For each layer, gather kept tokens and append a shard.
            flat_keep = keep_mask.reshape(-1)  # [B*S]
            for li in layer_indices:
                act = hooks[li].pop()  # [B, S, d_model] on CPU, storage_dtype
                flat = act.reshape(-1, act.shape[-1])  # [B*S, d_model]
                kept = flat[flat_keep].contiguous()  # [k, d_model]
                save_file({"acts": kept}, _shard_path(out_dir, li, n_shards))

            global_row += batch_kept
            n_tokens += batch_kept
            n_shards += 1
            logger.info(
                "captured batch %d (examples %d-%d): %d tokens, total %d",
                n_shards,
                base_example_id,
                base_example_id + len(batch_texts) - 1,
                batch_kept,
                n_tokens,
            )

        # Consolidate per-layer shards into final safetensors files.
        for li in layer_indices:
            rows = _concat_shards_to_final(out_dir, li, n_shards)
            if rows != n_tokens:  # pragma: no cover - invariant check
                raise RuntimeError(
                    f"row count mismatch for layer {li}: {rows} != {n_tokens}"
                )
    except Exception:
        _cleanup_partial(out_dir, layer_indices)
        raise
    finally:
        for hk in hooks.values():
            hk.remove()
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side

    # Write the shared token index.
    index_df = pd.DataFrame.from_records(
        index_records,
        columns=["row", "example_id", "tok_idx", "char_start", "char_end"],
    )
    index_path = os.path.join(out_dir, "index.parquet")
    index_df.to_parquet(index_path, index=False)

    # Write metadata.
    meta = {
        "model": _model_name(model),
        "d_model": int(d_model),
        "layers": [int(li) for li in layer_indices],
        "n_examples": int(n_examples),
        "n_tokens": int(n_tokens),
        "hook_point": hook_point,
    }
    save_json(meta, os.path.join(out_dir, "meta.json"))
    logger.info(
        "capture complete: %d examples, %d tokens, %d layers -> %s",
        n_examples,
        n_tokens,
        len(layer_indices),
        out_dir,
    )


def _resolve_storage_dtype(dtype: str) -> torch.dtype:
    """Map a dtype name to a torch dtype for on-disk storage.

    The spec stores activations as ``float16``; ``bfloat16`` and ``float32`` are
    also accepted for flexibility.  safetensors supports all three.
    """
    name = str(dtype).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation dtype {dtype!r}")
    return mapping[name]


def _iter_batches(
    texts: Sequence[str], batch_size: int
) -> Iterable[Tuple[List[str], int]]:
    """Yield ``(batch_texts, base_example_id)`` tuples over ``texts``.

    ``base_example_id`` is the index of the first text in the batch within the
    full corpus, which lets us recover global ``example_id`` values without holding
    extra state.  Uses :func:`know_trans.utils.batched` for the chunking.
    """
    base = 0
    for chunk in batched(texts, batch_size):
        chunk_list = list(chunk)
        yield chunk_list, base
        base += len(chunk_list)


def _append_index_records(
    index_records: List[Dict[str, int]],
    attn: torch.Tensor,
    offsets: torch.Tensor,
    base_example_id: int,
    start_row: int,
) -> int:
    """Append index rows for the kept (non-pad) tokens of one batch.

    Iterates in the same (row-major, ``[B, S]``) order used to flatten and filter
    the activation tensor, guaranteeing that ``index.parquet`` row ``r`` describes
    the activation stored at row ``r`` of every ``layer{L}.safetensors``.

    Parameters
    ----------
    index_records:
        The growing list of index dicts (mutated in place).
    attn:
        Attention mask ``[B, S]`` (CPU); ``1`` marks real tokens.
    offsets:
        Offset mapping ``[B, S, 2]`` (CPU) with ``(char_start, char_end)`` per token.
    base_example_id:
        Global example id of the first row in this batch.
    start_row:
        Global row index to assign to the first kept token of this batch.

    Returns
    -------
    int
        Number of kept tokens recorded for this batch.
    """
    attn_list = attn.tolist()
    offsets_list = offsets.tolist()
    row = start_row
    kept = 0
    for b, mask_row in enumerate(attn_list):
        example_id = base_example_id + b
        off_row = offsets_list[b]
        for tok_idx, m in enumerate(mask_row):
            if not m:
                continue
            char_start, char_end = off_row[tok_idx]
            index_records.append(
                {
                    "row": int(row),
                    "example_id": int(example_id),
                    "tok_idx": int(tok_idx),
                    "char_start": int(char_start),
                    "char_end": int(char_end),
                }
            )
            row += 1
            kept += 1
    return kept


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class ActivationReader:
    """Read activation artifacts written by :func:`capture_activations`.

    Parameters
    ----------
    model_dir:
        Directory containing ``layer{L}.safetensors``, ``index.parquet`` and
        ``meta.json`` (i.e. the ``out_dir`` passed to :func:`capture_activations`).

    Attributes
    ----------
    model_dir : str
        The directory being read.
    meta : dict
        Parsed ``meta.json`` contents.
    layers : list[int]
        Sorted list of available layer indices (from ``meta.json``, validated
        against on-disk ``layer{L}.safetensors`` files).
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {model_dir!r}")
        self.meta: Dict[str, Any] = load_json(meta_path)

        index_path = os.path.join(model_dir, "index.parquet")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index.parquet not found in {model_dir!r}")
        self._index_path = index_path
        self._index_cache: Optional[pd.DataFrame] = None

        # Determine available layers: prefer meta, but only keep ones present on disk.
        declared = [int(x) for x in self.meta.get("layers", [])]
        present = []
        for li in declared:
            if os.path.exists(_final_layer_path(model_dir, li)):
                present.append(li)
        if not present:
            # Fall back to discovering from filenames if meta is missing/stale.
            present = self._discover_layers(model_dir)
        self.layers: List[int] = sorted(present)

    @staticmethod
    def _discover_layers(model_dir: str) -> List[int]:
        """Discover layer indices from ``layer{L}.safetensors`` filenames."""
        found: List[int] = []
        for path in glob.glob(os.path.join(model_dir, "layer*.safetensors")):
            name = os.path.basename(path)
            stem = name[len("layer") : -len(".safetensors")]
            if stem.isdigit():
                found.append(int(stem))
        return sorted(found)

    @property
    def index(self) -> pd.DataFrame:
        """The full token index as a ``DataFrame`` (cached after first access)."""
        if self._index_cache is None:
            self._index_cache = pd.read_parquet(self._index_path)
        return self._index_cache

    @property
    def d_model(self) -> int:
        """Model hidden size recorded in ``meta.json``."""
        return int(self.meta["d_model"])

    @property
    def n_tokens(self) -> int:
        """Total number of token rows recorded in ``meta.json``."""
        return int(self.meta.get("n_tokens", len(self.index)))

    @property
    def n_examples(self) -> int:
        """Number of examples (texts) recorded in ``meta.json``."""
        return int(self.meta.get("n_examples", 0))

    def read(self, layer: int) -> Tuple[torch.Tensor, pd.DataFrame]:
        """Read one layer's activations and the aligned token index.

        Parameters
        ----------
        layer:
            The layer index to read (must be in :attr:`layers`).

        Returns
        -------
        tuple[torch.Tensor, pandas.DataFrame]
            ``(acts, index)`` where ``acts`` has shape ``[T, d_model]`` and ``index``
            is the full token index whose row ``r`` describes ``acts[r]``.

        Raises
        ------
        ValueError
            If ``layer`` is not available.
        FileNotFoundError
            If the expected safetensors file is missing.
        """
        if layer not in self.layers:
            raise ValueError(
                f"layer {layer} not available; have layers {self.layers}"
            )
        path = _final_layer_path(self.model_dir, layer)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing activation file: {path}")
        acts = load_file(path)["acts"]
        return acts, self.index

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ActivationReader(model_dir={self.model_dir!r}, "
            f"layers={self.layers}, d_model={self.d_model}, "
            f"n_tokens={self.n_tokens}, n_examples={self.n_examples})"
        )
