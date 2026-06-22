"""Shared helpers for know_trans: device/dtype, logging, IO, pooling.

Every other module imports its primitives from here so that names and
behaviours stay consistent. Nothing in this module performs heavy work at import
time; model loading is guarded behind :func:`load_model_and_tokenizer`.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Iterable, Iterator

import torch
from torch import Tensor

# --------------------------------------------------------------------------- #
# Reproducibility / hardware
# --------------------------------------------------------------------------- #
_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if present) and PyTorch RNGs for reproducibility.

    Parameters
    ----------
    seed:
        The integer seed to apply across all RNGs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # NumPy is optional at this layer.
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return the best available compute device as a string.

    Returns
    -------
    str
        ``"cuda"`` if a CUDA device is visible, ``"mps"`` on Apple silicon,
        otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(name: str) -> torch.dtype:
    """Map a dtype name string to a :class:`torch.dtype`.

    Parameters
    ----------
    name:
        One of ``float16``/``bfloat16``/``float32`` (or common aliases).

    Returns
    -------
    torch.dtype
        The corresponding torch dtype.

    Raises
    ------
    KeyError
        If ``name`` is not a recognised dtype.
    """
    key = name.lower().strip()
    if key not in _DTYPES:
        raise KeyError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[key]


def load_model_and_tokenizer(
    path: str,
    dtype: str = "bfloat16",
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal-LM checkpoint and its tokenizer from ``path``.

    Heavy imports (``transformers``) happen *inside* this function so that
    importing :mod:`know_trans.utils` stays cheap. The model is placed in eval
    mode on ``device``.

    Parameters
    ----------
    path:
        Filesystem path to a HuggingFace checkpoint directory.
    dtype:
        Compute dtype name (see :func:`get_dtype`).
    device:
        Target device; defaults to :func:`get_device` when ``None``.

    Returns
    -------
    tuple
        ``(model, tokenizer)``. The tokenizer's ``pad_token`` is set to
        ``eos_token`` when missing so batched encoding works.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    torch_dtype = get_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler.

    Idempotent: repeated calls with the same ``name`` reuse the existing
    handler rather than stacking duplicates.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__``).

    Returns
    -------
    logging.Logger
        Configured logger at ``INFO`` level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Filesystem / serialization
# --------------------------------------------------------------------------- #
def ensure_dir(p: str) -> str:
    """Create directory ``p`` (and parents) if needed and return it.

    Parameters
    ----------
    p:
        Directory path.

    Returns
    -------
    str
        The same path, now guaranteed to exist.
    """
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj: Any, path: str) -> None:
    """Serialize ``obj`` to ``path`` as UTF-8 JSON (parent dirs auto-created).

    Parameters
    ----------
    obj:
        Any JSON-serializable object.
    path:
        Destination file path.
    """
    parent = os.path.dirname(os.path.abspath(path))
    ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load and return a JSON object from ``path``.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    Any
        The deserialized object.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def batched(seq: Iterable[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive lists of up to ``n`` items from ``seq``.

    Works on any iterable (not just sized sequences), so it can stream over
    generators without materialising them.

    Parameters
    ----------
    seq:
        Source iterable.
    n:
        Maximum batch size; must be positive.

    Yields
    ------
    list
        Chunks of length ``n`` (the final chunk may be shorter).

    Raises
    ------
    ValueError
        If ``n <= 0``.
    """
    if n <= 0:
        raise ValueError(f"batch size must be positive, got {n}")
    chunk: list[Any] = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
# Pooling (the tokenizer-mismatch workaround)
# --------------------------------------------------------------------------- #
def pool_examples(
    token_acts: Tensor,
    example_ids: Tensor,
    mode: str = "mean",
) -> Tensor:
    """Pool per-token activations up to per-example vectors.

    Because the teacher and student use *different tokenizers*, token indices
    are not comparable across families. We therefore always pool token-level
    activations to the example (or span) level before any cross-model
    comparison. Rows are grouped by ``example_ids`` (which must be contiguous in
    the first appearance order); the output is ordered by first appearance.

    Parameters
    ----------
    token_acts:
        Float tensor of shape ``[T, d]`` (one row per token).
    example_ids:
        Long tensor of shape ``[T]`` mapping each token row to an example id
        (or span id). Ids need not be 0..E-1 nor sorted.
    mode:
        Pooling mode: ``"mean"`` (default), ``"max"``, ``"sum"`` or ``"last"``.

    Returns
    -------
    Tensor
        Tensor of shape ``[E, d]`` with one pooled vector per unique example id,
        ordered by the id's first appearance in ``example_ids``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``mode`` is unknown.
    """
    if token_acts.dim() != 2:
        raise ValueError(f"token_acts must be 2-D [T, d], got shape {tuple(token_acts.shape)}")
    if example_ids.dim() != 1 or example_ids.shape[0] != token_acts.shape[0]:
        raise ValueError("example_ids must be 1-D with length matching token_acts rows")

    # Preserve first-appearance order of example ids.
    unique_ids, inverse = torch.unique(example_ids, sorted=False, return_inverse=True)
    # ``torch.unique`` does not guarantee first-appearance order; rebuild it.
    order = _first_appearance_order(example_ids)
    n_examples = unique_ids.numel()
    d = token_acts.shape[1]
    work_dtype = torch.float32 if token_acts.dtype in (torch.float16, torch.bfloat16) else token_acts.dtype
    acts = token_acts.to(work_dtype)

    if mode == "last":
        out = torch.zeros(n_examples, d, dtype=work_dtype, device=token_acts.device)
        # The last token row for each example wins.
        for row in range(acts.shape[0]):
            out[inverse[row]] = acts[row]
        return _reorder(out, unique_ids, order).to(token_acts.dtype)

    if mode in ("mean", "sum"):
        out = torch.zeros(n_examples, d, dtype=work_dtype, device=token_acts.device)
        out.index_add_(0, inverse, acts)
        if mode == "mean":
            counts = torch.zeros(n_examples, dtype=work_dtype, device=token_acts.device)
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=work_dtype))
            out = out / counts.clamp_min(1.0).unsqueeze(1)
        return _reorder(out, unique_ids, order).to(token_acts.dtype)

    if mode == "max":
        out = torch.full((n_examples, d), float("-inf"), dtype=work_dtype, device=token_acts.device)
        for row in range(acts.shape[0]):
            idx = int(inverse[row])
            out[idx] = torch.maximum(out[idx], acts[row])
        return _reorder(out, unique_ids, order).to(token_acts.dtype)

    raise ValueError(f"Unknown pooling mode {mode!r}; expected mean/max/sum/last")


def _first_appearance_order(example_ids: Tensor) -> Tensor:
    """Return the unique ids of ``example_ids`` in first-appearance order."""
    seen: dict[int, None] = {}
    for value in example_ids.tolist():
        if value not in seen:
            seen[value] = None
    return torch.tensor(list(seen.keys()), device=example_ids.device)


def _reorder(out: Tensor, unique_ids: Tensor, desired_ids: Tensor) -> Tensor:
    """Reorder ``out`` (indexed by ``unique_ids``) to match ``desired_ids``."""
    id_to_row = {int(v): i for i, v in enumerate(unique_ids.tolist())}
    perm = torch.tensor([id_to_row[int(v)] for v in desired_ids.tolist()], device=out.device)
    return out.index_select(0, perm)
