"""know_trans — SAE-based cross-family knowledge distillation.

This package implements the pipeline described in ``docs/INTERFACE_SPEC.md`` and
``docs/DESIGN_NARRATIVE.md``: transfer knowledge from a stronger teacher to a
weaker student of a *different model family* by aligning them in an
interpretable **concept space** built from sparse autoencoder (SAE) features,
rather than in raw-activation space.

The public surface is intentionally small and stable so the individual modules
import cleanly from one another. See ``docs/INTERFACE_SPEC.md`` for the
authoritative contract.

Sub-modules
-----------
config
    Dataclasses describing a run + a YAML loader.
utils
    Device / dtype / logging / IO helpers + example-level pooling.
capture
    Span-aware activation capture and a streaming reader.
sae
    OpenAI-style TopK sparse autoencoder + trainer + bundle.
concepts
    Labeled concept battery built from the real benchmark files.
score
    Score every SAE feature as a concept detector (ROC-AUC).
match
    Label-anchored matching + validation gates (stability, null, causal).
distill
    Concept-activation-space distillation loss + trainer.
cli
    argparse entry points for each stage.

Importing this package never triggers heavy work (no model loads, no training,
no downloads); all such work is guarded behind functions / CLI entry points.
"""

from __future__ import annotations

# Eagerly export the public symbols of the two leaf modules that are guaranteed
# import-safe (they only pull in stdlib + torch/yaml and never trigger heavy
# work). The remaining stage modules are referenced by name in ``__all__`` so
# ``import know_trans`` always succeeds even before they are implemented; import
# them explicitly (e.g. ``from know_trans import capture``) to use them.
from . import config as config
from . import utils as utils
from .config import (
    CaptureCfg,
    Config,
    DistillCfg,
    ModelCfg,
    Paths,
    SAECfg,
    load_config,
)
from .utils import (
    batched,
    ensure_dir,
    get_device,
    get_dtype,
    get_logger,
    load_json,
    load_model_and_tokenizer,
    pool_examples,
    save_json,
    set_seed,
)

__all__ = [
    # sub-modules
    "config",
    "utils",
    "capture",
    "sae",
    "concepts",
    "score",
    "match",
    "distill",
    "cli",
    # config public symbols
    "ModelCfg",
    "SAECfg",
    "CaptureCfg",
    "DistillCfg",
    "Paths",
    "Config",
    "load_config",
    # utils public symbols
    "set_seed",
    "get_device",
    "get_dtype",
    "load_model_and_tokenizer",
    "get_logger",
    "ensure_dir",
    "save_json",
    "load_json",
    "batched",
    "pool_examples",
]

__version__ = "0.1.0"
