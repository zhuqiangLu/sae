# know_trans — Design Narrative (decision trail)

The canonical rationale behind the pipeline, captured from the design review. The HTML report
must explain all of this in depth.

1. **Reading A chosen.** The SAE is the *mechanism* of distillation, not just a microscope
   (Reading B). We use SAE concept alignment as a training signal.

2. **Why SAE over the baselines — the disentanglement bet.** Raw hidden states encode concepts
   in *superposition* inside family-specific geometry; matching them forces the student to copy the
   teacher's coordinate system — wasteful and possibly impossible across families. SAE concept space
   is a more canonical, monosemantic basis → transfer *concepts* without copying geometry.
   **Falsifiable headline:** SAE-distill beats hidden-state-distill *across families*, and the gap
   *widens with family distance*. Secondary payoff: interpretability / steerability.
   **Honest risk:** SAE reconstruction is lossy and sparse → may discard signal distillation needs.

3. **Anchor / primary baseline — Transport-and-Merge (arXiv 2602.05495).** It aligns *polysemantic
   neurons* via optimal transport and *merges weights* (no training). Our novelty is **concepts as
   the unit of correspondence** (vs their neurons), used for distillation/SFT. Their downstream gains
   are weak evidence of true concept-level correspondence — so the premise is *ours to prove*.

4. **Correspondence method — two independent SAEs + post-hoc matching** (not a shared crosscoder).
   Buys flexibility, inherits three hazards we must control: (a) the **circularity trap**,
   (b) **match instability** across SAE seeds, (c) **many-to-many / partial coverage**.

5. **Matching signal — labeled concept battery.** Anchor each model's features to an *external*
   concept label via detector AUC; match by shared label. This **breaks circularity** (the matches
   are not the quantity later minimized), is **tokenizer-proof**, and gives per-concept interpretable
   results. Coverage is limited to the battery → complement with auto-interp later.
   *Worked example:* concept = "harmful request"; AdvBench/HarmBench = positives, benign instructions
   = negatives; teacher feature #4821 (AUC 0.97) ↔ student feature #1190 (AUC 0.93), matched by the
   shared label, not by activation correlation.

6. **Validation gates before any distillation:** cross-seed stability (Jaccard), shuffled-pair null
   control, causal steering. **Teacher-only concepts** (no student match) are the *headline finding*,
   not noise.

7. **Pipeline:** capture (span-aware) → train SAEs (≥2 teacher seeds) → build battery (AdvBench /
   HarmBench / Global-MMLU-Lite) → score features→concepts (AUC) → match + validate → distill in
   concept-activation space.

8. **Loss design.** Project both models into a shared **concept-activation space** (dim = #shared
   concepts) via each model's own SAE + concept feature sets; align there. Sidesteps the d_hidden
   mismatch and the tokenizer mismatch in one move.

9. **Open decisions to flag:** battery breadth (benchmark-only vs auto-interp); pooling granularity
   (sequence vs span); frozen vs jointly-trained student SAE in distill; which exact teacher/student
   pairs (and how to vary family distance for the headline ablation).
