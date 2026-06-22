"""Concept-space knowledge distillation (Reading A: the SAE *is* the mechanism).

This module implements the final stage of the ``know_trans`` pipeline: transferring
knowledge from a **stronger teacher** to a **weaker student of a different family** by
aligning them in interpretable **concept-activation space** rather than raw-activation
space.

The central trick (see ``docs/DESIGN_NARRATIVE.md`` §8) sidesteps two mismatches at once:

* **d_hidden mismatch** — the teacher and student SAEs have different hidden widths.
* **tokenizer mismatch** — the two families tokenize text differently, so we can *never*
  align by token index.

We resolve both by projecting each model into a shared, low-dimensional
**concept-activation space** whose axes are the *shared concepts* discovered by the
matching stage::

    concept_act_c(example) = mean over that model's concept-c feature set of encode_dense

Because the projection is per-model (each model uses its *own* SAE and its *own* feature
set for concept ``c``), the two models land in the *same* ``C``-dimensional space even
though their raw activations and SAE codebooks are incomparable. Distillation then simply
pulls the student's concept activations toward the (frozen) teacher's.

Gradient flow (read carefully — this is the whole point of the loss design):

* The **teacher** model is frozen and run under ``torch.no_grad`` — it only *produces a
  target*; no gradient ever flows into it.
* **Both SAEs are frozen** when ``cfg.distill.freeze_saes`` is ``True`` (the default and
  the contractually-required behaviour for this anchor experiment). The student SAE still
  participates in the forward pass that produces the student's concept activations, but
  ``requires_grad`` is off on its parameters, so it acts as a fixed, differentiable
  *measuring instrument*. Gradients pass *through* the frozen student SAE's encoder back
  into the student model's hidden states, then into the student model's weights.
* The **student** model is the *only* thing that learns. Its parameters carry gradient
  from both the language-modeling loss and the concept-distillation loss.

Pooling is always at the **example level** (``cfg.distill.pool``), never the token level,
because the tokenizers differ across families.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Sibling modules — import the *exact* names defined in docs/INTERFACE_SPEC.md so imports
# line up across the package. These are intentionally module-level: distill.py is part of
# the know_trans package and is only imported once the rest of the package exists.
from know_trans.config import Config, DistillCfg
from know_trans.sae import SAEBundle, TopKSAE
from know_trans.utils import (
    get_device,
    get_dtype,
    get_logger,
    load_json,
    load_model_and_tokenizer,
    pool_examples,
    set_seed,
)

__all__ = [
    "concept_activations",
    "concept_distill_loss",
    "MatchedFeatureDistiller",
    "train_distill",
]

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Concept-set parsing helpers
# ---------------------------------------------------------------------------
#
# The matching stage (match.run_matching / write_matches) emits, per concept, the feature
# set for each model at the *discovered best layer*. The on-disk schema (see match.py) is::
#
#     {
#       "<concept>": {
#         "teacher": {"layer": <int>, "features": [<int>, ...]},
#         "student": {"layer": <int>, "features": [<int>, ...]},
#         "shared":  <bool>,
#       },
#       ...
#     }
#
# ``concept_activations`` accepts a *single model's* view of this — a mapping from concept
# name to that model's ``{"layer": ..., "features": [...]}`` block (or a bare list of
# feature indices). We normalise both shapes here so callers can pass whatever they have.


def _concept_feature_indices(entry: Any) -> list[int]:
    """Extract the flat list of feature indices from one concept's set entry.

    Accepts any of the shapes produced upstream:

    * ``{"layer": L, "features": [f0, f1, ...]}``  (per-model match block)
    * ``[{"layer": L, "feature": f, "auc": a}, ...]``  (concept_feature_sets output)
    * ``[f0, f1, ...]``  (a bare index list)

    Args:
        entry: One concept's feature-set descriptor in any of the supported shapes.

    Returns:
        A list of integer feature indices (possibly empty).
    """
    if entry is None:
        return []
    # Dict form: {"layer": ..., "features": [...]}
    if isinstance(entry, Mapping):
        feats = entry.get("features", [])
        return [int(f) for f in feats]
    # Sequence form: either [int, ...] or [{"feature": int, ...}, ...]
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        out: list[int] = []
        for f in entry:
            if isinstance(f, Mapping):
                out.append(int(f["feature"]))
            else:
                out.append(int(f))
        return out
    raise TypeError(f"Unrecognised concept feature-set entry of type {type(entry)!r}")


def _concept_layer(entry: Any) -> int | None:
    """Return the SAE layer associated with a concept feature-set entry, if present.

    Args:
        entry: One concept's feature-set descriptor.

    Returns:
        The integer layer index, or ``None`` if the entry does not carry one.
    """
    if isinstance(entry, Mapping) and "layer" in entry:
        return int(entry["layer"])
    if (
        isinstance(entry, Sequence)
        and not isinstance(entry, (str, bytes))
        and entry
        and isinstance(entry[0], Mapping)
        and "layer" in entry[0]
    ):
        return int(entry[0]["layer"])
    return None


def _ordered_concepts(concept_sets: Mapping[str, Any]) -> list[str]:
    """Return concept names in a deterministic order.

    A stable ordering matters because the columns of the concept-activation tensors of the
    teacher and student must line up index-for-index. We sort alphabetically so that two
    independently-built mappings over the *same* concept names always agree.

    Args:
        concept_sets: Mapping from concept name to feature-set descriptor.

    Returns:
        Sorted list of concept names.
    """
    return sorted(concept_sets.keys())


# ---------------------------------------------------------------------------
# Concept-activation projection
# ---------------------------------------------------------------------------


def concept_activations(
    codes_dense: Tensor,
    concept_sets: Mapping[str, Any],
    *,
    concept_order: Sequence[str] | None = None,
) -> Tensor:
    """Project dense SAE codes into shared concept-activation space.

    For each concept ``c`` we average the dense SAE activations over that model's concept-c
    feature set::

        concept_act[:, c] = mean_{f in features(c)} codes_dense[:, f]

    This maps a model's ``[E, d_hidden]`` dense codes (already pooled to the **example**
    level) into a ``[E, C]`` tensor whose axes are concepts shared with the other model.
    Because each model supplies its *own* ``concept_sets`` (its own feature indices), two
    models with different ``d_hidden`` produce tensors in the *same* ``C``-dimensional
    space — which is exactly what makes cross-family distillation well-defined.

    Gradient flow: this function is a plain differentiable reduction over ``codes_dense``.
    When ``codes_dense`` carries gradient (i.e. it is the *student's* code), gradients flow
    straight through the averaging back into ``codes_dense`` and hence into whatever
    produced it (the frozen student SAE encoder, then the student model).

    Args:
        codes_dense: Example-pooled dense SAE codes, shape ``[E, d_hidden]``.
        concept_sets: Mapping from concept name to this model's feature-set descriptor.
            Each value may be ``{"layer", "features"}``, a list of ``{"feature", ...}``
            dicts, or a bare list of indices (see :func:`_concept_feature_indices`).
        concept_order: Optional explicit concept ordering. When given, the output column
            ``c`` corresponds to ``concept_order[c]``. This MUST be identical for the
            teacher and student so their columns align. When ``None`` the concepts are
            sorted alphabetically (deterministic, and identical for matching key-sets).

    Returns:
        Concept activations, shape ``[E, C]`` where ``C == len(concept_order)`` (or the
        number of concepts in ``concept_sets``). A concept whose feature set is empty
        contributes an all-zeros column.

    Raises:
        ValueError: If ``codes_dense`` is not 2-D.
    """
    if codes_dense.dim() != 2:
        raise ValueError(
            f"codes_dense must be [E, d_hidden]; got shape {tuple(codes_dense.shape)}"
        )

    concepts = list(concept_order) if concept_order is not None else _ordered_concepts(
        concept_sets
    )
    E = codes_dense.shape[0]
    d_hidden = codes_dense.shape[1]
    C = len(concepts)

    out = codes_dense.new_zeros((E, C))
    for ci, name in enumerate(concepts):
        feats = _concept_feature_indices(concept_sets.get(name))
        # Drop any out-of-range indices defensively (e.g. if a feature set references a
        # different/larger SAE than the one that produced these codes).
        feats = [f for f in feats if 0 <= f < d_hidden]
        if not feats:
            # Empty / unusable concept set -> zero column. Keeps tensor shapes aligned
            # between teacher and student without injecting spurious signal.
            continue
        idx = torch.as_tensor(feats, dtype=torch.long, device=codes_dense.device)
        # index_select keeps the op differentiable wrt codes_dense; mean over the feature
        # axis collapses the set into one concept-activation scalar per example.
        out[:, ci] = codes_dense.index_select(1, idx).mean(dim=1)
    return out


# ---------------------------------------------------------------------------
# Concept-space distillation loss
# ---------------------------------------------------------------------------


def concept_distill_loss(
    student_concept_acts: Tensor,
    teacher_concept_acts: Tensor,
    *,
    loss_type: str = "smooth_l1",
    per_concept_norm: bool = True,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Distance between student and teacher concept activations.

    The teacher concept activations are treated as a fixed regression target; the student
    is pulled toward them. Gradient flows **only** into ``student_concept_acts`` — callers
    must supply a detached teacher target (see :class:`MatchedFeatureDistiller`), and this
    function additionally detaches the teacher defensively.

    Args:
        student_concept_acts: Student concept activations, shape ``[E, C]`` (carries grad).
        teacher_concept_acts: Teacher concept activations, shape ``[E, C]`` (target).
        loss_type: ``"smooth_l1"`` (Huber, robust to the occasional large SAE activation)
            or ``"mse"``.
        per_concept_norm: When ``True``, standardise each concept column by the teacher's
            per-concept scale before comparing, so concepts that happen to fire with large
            magnitudes do not dominate the gradient. Uses the teacher's per-column standard
            deviation as the scale (computed without grad).
        beta: Transition point of the smooth-L1 (Huber) loss. Ignored for ``"mse"``.
        eps: Numerical floor for the per-concept scale.

    Returns:
        A scalar loss tensor.

    Raises:
        ValueError: If the two inputs have mismatched shapes or ``loss_type`` is unknown.
    """
    if student_concept_acts.shape != teacher_concept_acts.shape:
        raise ValueError(
            "student/teacher concept activations must match in shape; got "
            f"{tuple(student_concept_acts.shape)} vs {tuple(teacher_concept_acts.shape)}"
        )

    # The teacher is a *target*: never let gradient leak into it (it is already produced
    # under no_grad in the distiller, but we detach again for safety and for callers that
    # use this function standalone).
    target = teacher_concept_acts.detach()
    pred = student_concept_acts

    if per_concept_norm:
        # Scale each concept column by the teacher's per-concept std. Computed without grad
        # so it acts as a fixed normaliser rather than something the student can game.
        scale = target.std(dim=0, keepdim=True).clamp_min(eps)
        scale = scale.detach()
        pred = pred / scale
        target = target / scale

    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=beta)
    raise ValueError(f"Unknown loss_type {loss_type!r}; expected 'mse' or 'smooth_l1'")


# ---------------------------------------------------------------------------
# Matched-feature distiller
# ---------------------------------------------------------------------------


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set ``requires_grad`` on every parameter of a module in place.

    Args:
        module: The module to (un)freeze.
        flag: ``True`` to enable gradients, ``False`` to freeze.
    """
    for p in module.parameters():
        p.requires_grad_(flag)


def _split_matches_by_model(
    matches: Mapping[str, Any],
    *,
    shared_only: bool = True,
) -> tuple[dict[str, dict], dict[str, dict], list[str]]:
    """Split a matches mapping into per-model concept-set views.

    Args:
        matches: The matches mapping (see module docstring / match.py schema).
        shared_only: When ``True`` (the default — and the only sensible choice for the
            distillation loss, which lives in the *shared* concept space), keep only
            concepts present in both models. Teacher-only concepts are the headline
            *finding* but cannot define a shared axis, so they are excluded from the loss.

    Returns:
        ``(teacher_sets, student_sets, concept_order)`` where the first two are mappings
        concept -> ``{"layer", "features"}`` and ``concept_order`` is the shared,
        deterministically-sorted list of concept names that defines the column layout.
    """
    teacher_sets: dict[str, dict] = {}
    student_sets: dict[str, dict] = {}
    for concept, blk in matches.items():
        if not isinstance(blk, Mapping):
            continue
        if shared_only and not blk.get("shared", False):
            continue
        t = blk.get("teacher")
        s = blk.get("student")
        if t is None or s is None:
            continue
        teacher_sets[concept] = dict(t)
        student_sets[concept] = dict(s)
    concept_order = sorted(set(teacher_sets) & set(student_sets))
    return teacher_sets, student_sets, concept_order


def _unique_layer(
    concept_sets: Mapping[str, Any], concept_order: Sequence[str]
) -> int | None:
    """Return the single SAE layer used across a concept-set view, if unique.

    The distiller captures one hidden layer per model. If every concept in the shared set
    points at the same discovered layer (the common case for an anchor experiment), we use
    it. If concepts disagree on the layer we return ``None`` and the caller must specify
    the capture layer explicitly via the config.

    Args:
        concept_sets: Per-model concept -> feature-set mapping.
        concept_order: Concepts to consider.

    Returns:
        The shared layer index, or ``None`` if concepts disagree / none is recorded.
    """
    layers = {
        _concept_layer(concept_sets[c])
        for c in concept_order
        if concept_sets.get(c) is not None
    }
    layers.discard(None)
    if len(layers) == 1:
        return next(iter(layers))
    return None


class _ResidualCapture:
    """Forward-hook helper that captures the output of one decoder block's MLP.

    We register a forward hook on ``model.model.layers[i].mlp`` (matching capture.MLPHook's
    hook point) and stash the output tensor on each forward pass. Unlike capture.MLPHook —
    which detaches and moves activations to CPU for *storage* — this hook keeps the tensor
    **attached to the graph** so gradient can flow back through the captured activations
    into the model. That is essential for the student; for the teacher the surrounding
    ``no_grad`` context makes the distinction moot.

    Attributes:
        activations: The most recently captured activation tensor (``[B, T, d_model]``),
            or ``None`` before the first forward pass.
    """

    def __init__(self, model: nn.Module, layer: int) -> None:
        """Register the hook.

        Args:
            model: A causal-LM whose blocks live at ``model.model.layers[i].mlp``.
            layer: Index of the decoder block whose MLP output to capture.
        """
        self.activations: Tensor | None = None
        self._layer = layer
        block = model.model.layers[layer].mlp
        self._handle = block.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inp: Any, output: Any) -> None:
        """Store the MLP output (kept on the autograd graph)."""
        # HF MLP modules return a plain tensor; guard for the tuple case just in case.
        self.activations = output[0] if isinstance(output, tuple) else output

    def remove(self) -> None:
        """Detach the forward hook."""
        self._handle.remove()


class MatchedFeatureDistiller:
    """Distill teacher -> student in shared concept-activation space.

    The distiller wires together: a **frozen teacher** LM + its frozen SAE (the source of
    concept targets), and a **trainable student** LM + its (by default frozen) SAE (the
    measuring instrument). For each batch it computes::

        loss = lm_loss + cfg.lambda_concept * concept_loss

    where ``lm_loss`` is the student's own next-token cross-entropy (keeps the student a
    competent language model) and ``concept_loss`` pulls the student's concept activations
    toward the teacher's (the knowledge-transfer term).

    Gradient bookkeeping (see also the module docstring):

    * Teacher LM + teacher SAE: frozen, run under ``torch.no_grad``. They emit a detached
      target only.
    * Student SAE: parameters frozen iff ``cfg.distill.freeze_saes`` (default ``True``),
      but always run *with* grad enabled so gradients pass *through* it into the student
      LM. When ``freeze_saes`` is ``False`` the student SAE parameters also learn.
    * Student LM: fully trainable — the sole recipient of the LM-loss gradient.

    Attributes:
        teacher: Frozen teacher causal-LM.
        student: Trainable student causal-LM.
        t_sae: Frozen teacher :class:`~know_trans.sae.TopKSAE`.
        s_sae: Student :class:`~know_trans.sae.TopKSAE` (frozen params by default).
        cfg: The :class:`~know_trans.config.DistillCfg` controlling the run.
        concept_order: Shared concept names defining the concept-space column layout.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        t_sae: TopKSAE,
        s_sae: TopKSAE,
        matches: Mapping[str, Any],
        cfg: DistillCfg,
    ) -> None:
        """Build the distiller and freeze everything except the student LM.

        Args:
            teacher: The (stronger) teacher causal-LM.
            student: The (weaker, different-family) student causal-LM to train.
            t_sae: SAE trained on the teacher's activations (frozen).
            s_sae: SAE trained on the student's activations.
            matches: Output of ``match.run_matching`` (concept -> per-model feature sets).
            cfg: Distillation hyper-parameters.

        Raises:
            ValueError: If ``matches`` contains no concepts shared by both models, or if a
                single capture layer per model cannot be determined from the matches.
        """
        self.teacher = teacher
        self.student = student
        self.t_sae = t_sae
        self.s_sae = s_sae
        self.cfg = cfg
        self.pool: str = cfg.pool

        # --- Partition matches into the shared concept space -----------------------
        self._teacher_sets, self._student_sets, self.concept_order = (
            _split_matches_by_model(matches, shared_only=True)
        )
        if not self.concept_order:
            raise ValueError(
                "No shared concepts in `matches` — nothing to distill. "
                "Teacher-only concepts cannot define a shared concept axis."
            )

        # --- Resolve the single hidden layer to hook per model ---------------------
        self._t_layer = _unique_layer(self._teacher_sets, self.concept_order)
        self._s_layer = _unique_layer(self._student_sets, self.concept_order)
        if self._t_layer is None or self._s_layer is None:
            raise ValueError(
                "Could not determine a unique capture layer per model from the matches. "
                "The anchor distiller hooks one layer per model; ensure the shared "
                "concepts agree on a layer, or pre-filter the matches before distilling."
            )

        # --- Freeze the teacher LM (no grad ever) ----------------------------------
        self.teacher.eval()
        _set_requires_grad(self.teacher, False)

        # --- Freeze the teacher SAE (always) ---------------------------------------
        self.t_sae.eval()
        _set_requires_grad(self.t_sae, False)

        # --- Student SAE: freeze params per cfg, but it stays in the forward graph --
        if cfg.freeze_saes:
            self.s_sae.eval()
            _set_requires_grad(self.s_sae, False)
        else:
            # Allow the student SAE to co-adapt with the student LM.
            self.s_sae.train()
            _set_requires_grad(self.s_sae, True)

        # --- Student LM: the only thing that learns from the LM loss ---------------
        self.student.train()
        _set_requires_grad(self.student, True)

        # --- Install MLP-output capture hooks --------------------------------------
        self._t_cap = _ResidualCapture(self.teacher, self._t_layer)
        self._s_cap = _ResidualCapture(self.student, self._s_layer)

        _LOG.info(
            "MatchedFeatureDistiller ready: %d shared concepts; teacher layer %d, "
            "student layer %d; freeze_saes=%s",
            len(self.concept_order),
            self._t_layer,
            self._s_layer,
            cfg.freeze_saes,
        )

    # ------------------------------------------------------------------ helpers ----

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the parameters an optimizer should update.

        Always the student LM; additionally the student SAE when ``freeze_saes`` is False.

        Returns:
            List of parameters with ``requires_grad=True``.
        """
        params = [p for p in self.student.parameters() if p.requires_grad]
        if not self.cfg.freeze_saes:
            params += [p for p in self.s_sae.parameters() if p.requires_grad]
        return params

    def _pool_to_examples(
        self, token_acts: Tensor, example_ids: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        """Pool ``[B, T, d]`` token activations to ``[E, d]`` at the example level.

        We flatten the batch/time dims, drop padding (via ``attention_mask``), and delegate
        to :func:`know_trans.utils.pool_examples`, which groups by ``example_id``. Pooling
        is differentiable (mean / sum reductions), so gradient flows back into
        ``token_acts``.

        Args:
            token_acts: Activations, shape ``[B, T, d]``.
            example_ids: Per-token example id, shape ``[B, T]`` (long).
            attention_mask: Optional ``[B, T]`` mask; non-positive entries are dropped.

        Returns:
            Example-pooled activations, shape ``[E, d]`` where ``E`` is the number of
            distinct example ids in the (unmasked) batch.
        """
        B, T, d = token_acts.shape
        flat_acts = token_acts.reshape(B * T, d)
        flat_ids = example_ids.reshape(B * T).to(torch.long)
        if attention_mask is not None:
            keep = attention_mask.reshape(B * T) > 0
            flat_acts = flat_acts[keep]
            flat_ids = flat_ids[keep]
        # pool_examples returns rows ordered by sorted unique example id; the teacher and
        # student share the same example_ids in a batch, so their [E, *] rows align.
        return pool_examples(flat_acts, flat_ids, mode=self.pool)

    @staticmethod
    def _encode_dense(sae: TopKSAE, acts: Tensor) -> Tensor:
        """Run an SAE encoder to dense codes, casting dtype to match the SAE.

        Args:
            sae: The SAE to run.
            acts: Pooled activations, shape ``[E, d_in]``.

        Returns:
            Dense codes, shape ``[E, d_hidden]``.
        """
        # SAEs are typically trained/stored in float32; cast the (bf16/fp16) model acts to
        # the SAE's parameter dtype so the matmul is well-typed. This cast is on-graph for
        # the student, so gradient flows back to ``acts`` (and thus the student LM).
        sae_dtype = next(sae.parameters()).dtype
        return sae.encode_dense(acts.to(sae_dtype))

    # -------------------------------------------------------------------- loss -----

    def compute_loss(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Compute the combined distillation loss for one batch.

        The ``batch`` must contain aligned teacher- and student-tokenized views of the
        *same* underlying examples, plus a shared ``example_id`` per token so that example
        pooling lines up across the two (differently-tokenized!) views. Expected keys:

        * ``teacher_input_ids`` / ``teacher_attention_mask`` / ``teacher_example_ids``
        * ``student_input_ids`` / ``student_attention_mask`` / ``student_example_ids``
        * optional ``labels`` (defaults to the student input ids for the LM loss)

        Flow and gradients:

        1. Teacher forward under ``no_grad`` -> capture MLP acts -> pool -> teacher SAE
           ``encode_dense`` -> :func:`concept_activations`. Everything detached: a target.
        2. Student forward (with grad) -> capture MLP acts -> pool -> student SAE
           ``encode_dense`` -> :func:`concept_activations`. Gradient flows through the
           (frozen-parameter) student SAE into the student LM.
        3. ``concept_loss`` = :func:`concept_distill_loss` (student vs detached teacher).
        4. ``lm_loss`` = the student forward's own cross-entropy.
        5. ``loss = lm_loss + lambda_concept * concept_loss``.

        Args:
            batch: Mapping of tensors as described above.

        Returns:
            ``{"loss", "lm_loss", "concept_loss"}`` — all scalar tensors. ``loss`` carries
            the autograd graph for ``.backward()``.

        Raises:
            KeyError: If required student/teacher fields are missing from ``batch``.
        """
        device = next(self.student.parameters()).device

        # ---- Required fields ------------------------------------------------------
        try:
            s_ids = batch["student_input_ids"].to(device)
            s_mask = batch["student_attention_mask"].to(device)
            s_eids = batch["student_example_ids"].to(device)
            t_ids = batch["teacher_input_ids"]
            t_mask = batch["teacher_attention_mask"]
            t_eids = batch["teacher_example_ids"]
        except KeyError as exc:  # surface a clear contract error
            raise KeyError(
                "compute_loss expects teacher_/student_ input_ids, attention_mask and "
                f"example_ids in `batch`; missing {exc}"
            ) from exc
        labels = batch.get("labels", s_ids).to(device)

        # =====================================================================
        # 1) TEACHER: frozen target. No gradient flows here at all.
        # =====================================================================
        t_device = next(self.teacher.parameters()).device
        with torch.no_grad():
            self.teacher(
                input_ids=t_ids.to(t_device),
                attention_mask=t_mask.to(t_device),
            )
            t_tok_acts = self._t_cap.activations  # [B, T_t, d_t], detached (no_grad)
            assert t_tok_acts is not None, "teacher hook did not fire"
            t_pooled = self._pool_to_examples(
                t_tok_acts, t_eids.to(t_device), t_mask.to(t_device)
            )  # [E, d_t]
            t_codes = self._encode_dense(self.t_sae, t_pooled)  # [E, d_hidden_t]
            teacher_concept = concept_activations(
                t_codes, self._teacher_sets, concept_order=self.concept_order
            )  # [E, C]
        # Move the target to the student's device and detach (belt and braces).
        teacher_concept = teacher_concept.to(device).detach()

        # =====================================================================
        # 2) STUDENT: trainable. Gradient flows from here back into student LM.
        #    grad path: lm_loss + concept_loss -> student hidden states ->
        #               (through frozen student SAE encoder) -> student weights.
        # =====================================================================
        out = self.student(
            input_ids=s_ids,
            attention_mask=s_mask,
            labels=labels,  # HF computes the shifted-CE LM loss for us
        )
        lm_loss = out.loss  # carries grad into the student LM

        s_tok_acts = self._s_cap.activations  # [B, T_s, d_s], ON graph
        assert s_tok_acts is not None, "student hook did not fire"
        s_pooled = self._pool_to_examples(s_tok_acts, s_eids, s_mask)  # [E, d_s]
        # encode_dense runs the (frozen-param) student SAE; grad passes through it.
        s_codes = self._encode_dense(self.s_sae, s_pooled)  # [E, d_hidden_s]
        student_concept = concept_activations(
            s_codes, self._student_sets, concept_order=self.concept_order
        ).to(lm_loss.dtype)  # [E, C]

        # =====================================================================
        # 3) Concept-space distillation loss (student pulled to teacher target).
        # =====================================================================
        concept_loss = concept_distill_loss(student_concept, teacher_concept)

        # =====================================================================
        # 4) Combine. lambda_concept trades off LM competence vs concept transfer.
        # =====================================================================
        total = lm_loss + self.cfg.lambda_concept * concept_loss.to(lm_loss.dtype)

        return {"loss": total, "lm_loss": lm_loss, "concept_loss": concept_loss}

    def close(self) -> None:
        """Remove the forward hooks. Safe to call multiple times."""
        self._t_cap.remove()
        self._s_cap.remove()

    def __enter__(self) -> "MatchedFeatureDistiller":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_distill(cfg: Config, matches_path: str) -> None:
    """Run concept-space distillation end to end.

    Loads the teacher and student models + their SAE bundles, reads the validated matches,
    builds a :class:`MatchedFeatureDistiller`, and runs ``cfg.distill.steps`` optimisation
    steps over batches produced by a data loader. Only the student LM (and, optionally, the
    student SAE) is updated; the teacher and teacher SAE never receive gradient.

    This function performs heavy work (model loading, optimisation) and is therefore only
    invoked from a CLI / script — never at import time, per the package contract.

    .. note::
        The batch source is deliberately pluggable. This module does not own the dataset /
        dual-tokenizer collation logic (that belongs with ``capture`` / a dedicated data
        module), so a missing batch builder raises a clear, actionable error rather than
        guessing. The contract for a batch is documented on
        :meth:`MatchedFeatureDistiller.compute_loss`.

    Args:
        cfg: The full pipeline :class:`~know_trans.config.Config`.
        matches_path: Path to the matches JSON written by ``match.write_matches``.

    Raises:
        FileNotFoundError: If ``matches_path`` does not exist.
        NotImplementedError: If no batch iterator is wired in for this run (the dual-
            tokenizer training loader is provided by the data/CLI layer, not by distill).
    """
    set_seed(cfg.seed)
    device = get_device()
    log = get_logger("know_trans.distill.train")

    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"matches file not found: {matches_path}")
    matches = load_json(matches_path)

    # --- Load models (frozen teacher, trainable student) ---------------------------
    log.info("Loading teacher from %s", cfg.teacher.path)
    teacher, _ = load_model_and_tokenizer(
        cfg.teacher.path, dtype=cfg.teacher.dtype, device=device
    )
    log.info("Loading student from %s", cfg.student.path)
    student, _ = load_model_and_tokenizer(
        cfg.student.path, dtype=cfg.student.dtype, device=device
    )

    # --- Load the per-model SAE bundles and pick the matched layers ----------------
    # SAEBundle.load returns a bundle keyed by layer; the distiller resolves the exact
    # layer per model from the matches, so here we just need a TopKSAE for each side.
    sae_root = cfg.paths.data
    t_bundle = SAEBundle.load(os.path.join(sae_root, "saes", cfg.teacher.name))
    s_bundle = SAEBundle.load(os.path.join(sae_root, "saes", cfg.student.name))

    # Determine the single shared layer per model from the matches (same logic the
    # distiller uses) so we can index the bundles.
    t_sets, s_sets, order = _split_matches_by_model(matches, shared_only=True)
    if not order:
        raise ValueError("matches contains no shared concepts; nothing to distill.")
    t_layer = _unique_layer(t_sets, order)
    s_layer = _unique_layer(s_sets, order)
    if t_layer is None or s_layer is None:
        raise ValueError(
            "matches do not agree on a single layer per model; cannot pick an SAE."
        )
    t_sae = t_bundle[t_layer].to(device)
    s_sae = s_bundle[s_layer].to(device)

    distiller = MatchedFeatureDistiller(
        teacher=teacher,
        student=student,
        t_sae=t_sae,
        s_sae=s_sae,
        matches=matches,
        cfg=cfg.distill,
    )

    optim = torch.optim.AdamW(distiller.trainable_parameters(), lr=cfg.distill.lr)

    # --- Batch source --------------------------------------------------------------
    # The dual-tokenizer training loader (teacher- and student-tokenized views of the
    # same examples, sharing example_ids) is owned by the data/CLI layer. We fetch it via
    # an optional hook so distill.py stays free of dataset/collation concerns.
    batch_iter = _resolve_batch_iter(cfg)
    if batch_iter is None:
        raise NotImplementedError(
            "train_distill needs a dual-tokenizer batch iterator yielding the fields "
            "documented on MatchedFeatureDistiller.compute_loss. Wire one in via a data "
            "module / the CLI (e.g. attach `cfg._distill_batch_iter`). It is intentionally "
            "not implemented here so distill.py owns only the loss, not data loading."
        )

    log.info("Starting distillation: %d steps, lr=%g, lambda_concept=%g",
             cfg.distill.steps, cfg.distill.lr, cfg.distill.lambda_concept)

    step = 0
    distiller.student.train()
    try:
        for batch in batch_iter:
            if step >= cfg.distill.steps:
                break
            optim.zero_grad(set_to_none=True)
            out = distiller.compute_loss(batch)
            out["loss"].backward()  # grad flows into student LM (+ student SAE if unfrozen)
            optim.step()
            if step % 50 == 0:
                log.info(
                    "step %d | loss %.4f | lm %.4f | concept %.4f",
                    step,
                    float(out["loss"].detach()),
                    float(out["lm_loss"].detach()),
                    float(out["concept_loss"].detach()),
                )
            step += 1
    finally:
        distiller.close()

    # --- Save the distilled student ------------------------------------------------
    out_dir = os.path.join(cfg.paths.data, "distilled", cfg.student.name)
    os.makedirs(out_dir, exist_ok=True)
    log.info("Saving distilled student to %s", out_dir)
    # save_pretrained is the canonical HF persistence path for the trained student LM.
    distiller.student.save_pretrained(out_dir)
    log.info("Distillation complete after %d steps.", step)


def _resolve_batch_iter(cfg: Config) -> Any | None:
    """Return the training batch iterator attached to ``cfg``, if any.

    Kept as a single, easily-mocked seam so that the CLI / a data module can inject a
    dual-tokenizer loader without distill.py depending on the dataset implementation.

    Args:
        cfg: The pipeline config (possibly carrying a private ``_distill_batch_iter``).

    Returns:
        An iterable of batches, or ``None`` if none is attached.
    """
    return getattr(cfg, "_distill_batch_iter", None)
