# know_trans

**SAE-based cross-family knowledge distillation.**

Transfer knowledge from a **stronger teacher** to a **weaker student of a
different model family** by aligning them in an interpretable **concept space**
built from sparse-autoencoder (SAE) features — not in raw-activation space.

The concrete default pair is **Llama-3.1-8B (teacher) → Qwen3-0.6B (student)**:
different families, very different sizes, and — critically — **different
tokenizers**, so we never align by token index. Everything is pooled to the
example/span level.

> The cross-family **concept-correspondence** premise is *ours to prove*.
> Primary baseline to beat: **Transport-and-Merge** (optimal transport on raw
> activations + weight merge, arXiv 2602.05495). Secondary baselines:
> hidden-state distillation (learned projector) and logit distillation.

See `docs/INTERFACE_SPEC.md` (authoritative API contract) and
`docs/DESIGN_NARRATIVE.md` (decision trail) for full detail.

---

## Why SAE concept space? (the bet)

Raw hidden states encode concepts in **superposition** inside family-specific
geometry. Matching raw activations forces the student to copy the teacher's
coordinate system — wasteful and possibly impossible across families. An SAE
gives a more canonical, **monosemantic** basis, letting us transfer *concepts*
without copying geometry.

**Falsifiable headline:** SAE-distill beats hidden-state-distill *across
families*, and the gap *widens with family distance*. **Honest risk:** SAE
reconstruction is lossy and sparse, so it may discard signal that distillation
needs.

---

## The 7-stage pipeline

```
capture → train-sae → build-concepts → score → match → (validate) → distill
```

1. **capture** — Span-aware activation capture. Run texts through each model,
   hook the per-layer MLP output, and write `layer{L}.safetensors` (`acts`
   `[T, d_model]`), an `index.parquet` (`row, example_id, tok_idx, char_start,
   char_end`), and `meta.json`. Char spans (from `return_offsets_mapping=True`)
   enable span pooling; shards are streamed, never all held in RAM.

2. **train-sae** — Train an OpenAI-style **TopK SAE** per layer (recon MSE +
   AuxK dead-feature revival, unit-norm decoder columns; track dead features and
   L0). `d_hidden = expansion * d_in`, with `d_in` inferred from the shards.
   Train **≥2 teacher seeds** so match stability can be measured.

3. **build-concepts** — Build a **labeled concept battery** from the real
   benchmark files (AdvBench, HarmBench, Global-MMLU-Lite). Groups: `safety`
   (harmful behaviors vs benign MMLU questions), `safety_semantic` (per HarmBench
   `SemanticCategory`), `topic_<subject>` (Global-MMLU-Lite subjects),
   `language_<xx>` (per-language dirs). One JSONL per concept.

4. **score** — Score **every SAE feature** as a concept *detector*. Pool
   `encode_dense` over spans to `[E, d_hidden]`, run positives + negatives, and
   compute per-feature `roc_auc_score`. Output a parquet of
   `[layer, feature, concept, auc, n_pos, n_neg]`. This **anchors features to
   external labels**, which is what breaks the circularity trap and is
   tokenizer-proof.

5. **match** — **Label-anchored** matching: a concept present in both models'
   above-threshold feature sets is `shared:true` with each side's best
   `(layer, features)`; **teacher-only** concepts (no student match) are
   `shared:false` and are the **headline finding**, logged separately.

6. **validate** (part of `match`) — Gates *before* any distillation:
   cross-seed **stability** (Jaccard), **null control** (shuffled-pair gap), and
   **causal steering** (ablate/steer a matched feature via its decoder direction
   and measure the behavioral delta — actually run).

7. **distill** — Map both models into a shared **concept-activation space**
   (dim = #shared concepts): `concept_act_c = mean over that model's concept-c
   feature set of encode_dense`. Align there with an MSE/smooth-L1 loss
   (`concept_distill_loss`) added to the student LM loss. Teacher frozen; both
   SAEs frozen by default; gradients flow into the **student only**. This single
   move sidesteps **both** the `d_hidden` mismatch and the tokenizer mismatch.

Layer correspondence is **discovered** in score/match (best `(t_layer, s_layer)`
per concept), never hardcoded.

---

## Installation

Python ≥ 3.10. Heavy work (model loading, training, downloads) is always behind
a function/CLI call — importing the package never triggers it.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs deps from pyproject.toml
# (CUDA torch should match your cluster; install the matching wheel if needed.)
```

This exposes a console entry point `know-trans`, equivalent to
`python -m know_trans.cli`.

---

## Running each CLI subcommand

Every subcommand takes `--config <yaml>` and calls into the stage modules. The
default config is `configs/pair_llama8b_qwen0p6b.yaml`.

```bash
CFG=configs/pair_llama8b_qwen0p6b.yaml

# 1) Capture span-aware MLP activations for teacher + student.
know-trans capture        --config $CFG

# 2) Train a TopK SAE per layer (run twice with different seeds for stability).
know-trans train-sae      --config $CFG

# 3) Build the labeled concept battery from the benchmark files.
know-trans build-concepts --config $CFG

# 4) Score every SAE feature as a concept detector (ROC-AUC).
know-trans score          --config $CFG

# 5) Match concepts across models + run the validation gates.
know-trans match          --config $CFG

# 6) Distill in concept-activation space (student only is trained).
know-trans distill        --config $CFG --matches data/matches/matches.json
```

(Equivalently: `python -m know_trans.cli <subcommand> --config $CFG`. Exact
flags per subcommand are defined in `know_trans.cli`; consult `--help`.)

### Outputs (all under `data/`, gitignored)

```
data/
  activations/   layer{L}.safetensors + index.parquet + meta.json
  saes/          per-layer SAE checkpoints (SAEBundle dir)
  concepts/      one jsonl per concept (the battery)
  feature_scores/  feature→concept AUC parquet
  matches/       matches.json (shared + teacher-only) + validation reports
```

---

## Decision-trail summary

A condensed version of `docs/DESIGN_NARRATIVE.md`:

- **Reading A.** The SAE is the *mechanism* of distillation, not just a
  microscope — concept alignment is used as a **training signal**.
- **The disentanglement bet.** SAE concept space is a more canonical,
  monosemantic basis than raw activations, so we transfer concepts without
  copying family-specific geometry.
- **Anchor baseline: Transport-and-Merge.** Our novelty is **concepts (not
  neurons) as the unit of correspondence**, used for distillation rather than a
  training-free weight merge.
- **Two independent SAEs + post-hoc matching** (not a shared crosscoder) — more
  flexible, but we must control three hazards: the **circularity trap**, **match
  instability** across SAE seeds, and **many-to-many / partial coverage**.
- **Matching signal = labeled concept battery.** Anchoring features to an
  *external* label via detector AUC **breaks circularity** (matches are not the
  quantity later minimized), is **tokenizer-proof**, and is interpretable.
- **Validation before distillation:** cross-seed stability (Jaccard), shuffled-
  pair null control, and causal steering. **Teacher-only concepts** are the
  headline finding, not noise.
- **Loss design:** project both models into a shared concept-activation space
  (dim = #shared concepts) and align there — one move that sidesteps both the
  `d_hidden` mismatch and the tokenizer mismatch.

---

## Repo layout

```
src/know_trans/{__init__,config,utils,capture,sae,concepts,score,match,distill,cli}.py
configs/pair_llama8b_qwen0p6b.yaml
scripts/*.py
report/design_report.html
docs/{INTERFACE_SPEC,DESIGN_NARRATIVE}.md
data/{activations,saes,concepts,feature_scores,matches}/   # outputs; gitignored
```

## Conventions

- **safetensors** for tensors, **parquet** for tables, **json** for small
  structured outputs.
- Never hardcode dims: infer `d_in` from activation shards;
  `d_hidden = expansion * d_in`.
- Tokenizer mismatch ⇒ pooling is always example/span level (via
  `index.parquet` spans / `example_id`).
- Layer correspondence is discovered, never hardcoded.
- Importing any module triggers no heavy work.
