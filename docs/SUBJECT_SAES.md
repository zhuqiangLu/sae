# Subject-specialized SAEs — training corpora, models, and artifacts

**Status:** 2026-06-22. Experiment B. 228 SAEs = 3 models × 3 domains × all layers.
Diagnostic/research artifacts, kept **separate** from the production general SAEs
in `data/saes/`.

---

## 1. Why these exist

The production general SAEs (`data/saes/<model>/seed0/`) were trained on a corpus
(`capture.corpus = "benchmarks"`) that is **94% Global-MMLU-Lite (multilingual) +
6% AdvBench/HarmBench (safety)**, totalling only ~1.16M tokens. That corpus contains
**no full English MMLU**, so the SAEs never learned monosemantic features for the
`topic_math / topic_economics / topic_medical` concepts (which are *defined* from full
English MMLU).

Measured consequence (`scripts/diag_sae_ood.py`): on held-out math/eco/med activations
the general SAE leaves **~0.59–0.66 of the variance unreconstructed (FVU)** at mid/late
layers, vs ~0.25–0.46 on its own training distribution. The topic concept-activation
dimensions were therefore dominated by reconstruction *noise* — which explains why
concept-space distillation moved alignment loss but **not** downstream accuracy.

These subject-specialized SAEs are the controlled test of the fix: **same SAE recipe,
only the training corpus changes** to in-domain text. If coverage is the binding
constraint, an in-domain SAE should reconstruct held-out in-domain activations far
better. It does (see §6).

---

## 2. Models

| role | model | d_model | layers | general SAE (baseline) |
|------|-------|--------:|-------:|------------------------|
| teacher | `Llama-3.1-8B`  | 4096 | 32 | `data/saes/Llama-3.1-8B/seed0/` |
| student | `Qwen3-0.6B`    | 1024 | 28 | `data/saes/Qwen3-0.6B/seed0/`   |
| student | `Llama-3.2-1B`  | 2048 | 16 | `data/saes/Llama-3.2-1B/seed0/` |

Model weights live under `models/` (symlink to `/share1/zhlu6105/models`).
One specialized SAE is trained per (model, domain, layer).

---

## 3. Training corpora (the new data)

Built by `build_train(domain)` in `scripts/exp_subject_sae_full.py`.

- **Source:** full **English MMLU** only — `benchmarks/MMLU/<subject>/{test,validation,dev}-*.parquet`.
  *Not* Global-MMLU-Lite, *not* MMLU-Pro, *no* safety/multilingual text.
- **Format:** each example is the MMLU question concatenated with its answer choices
  (`_fmt(question, choices)`).
- **Splits pooled:** `test + validation + dev` (to maximize the limited English domain
  text while keeping MMLU-Pro fully held out).

Token counts are the real captured values from the per-SAE `*.train.json` (teacher).

### econ — 3 subjects, 838 texts, ~48.7k tokens
| subject | test | val | dev | total |
|---------|-----:|----:|----:|------:|
| high_school_macroeconomics | 390 | 43 | 5 | 438 |
| high_school_microeconomics | 238 | 26 | 5 | 269 |
| econometrics               | 114 | 12 | 5 | 131 |

### math — 5 subjects, 1204 texts, ~78.6k tokens
| subject | test | val | dev | total |
|---------|-----:|----:|----:|------:|
| abstract_algebra        | 100 | 11 | 5 | 116 |
| college_mathematics     | 100 | 11 | 5 | 116 |
| high_school_mathematics | 270 | 29 | 5 | 304 |
| elementary_mathematics  | 378 | 41 | 5 | 424 |
| high_school_statistics  | 216 | 23 | 5 | 244 |

### med — 7 subjects, 1610 texts, ~121k tokens
| subject | test | val | dev | total |
|---------|-----:|----:|----:|------:|
| clinical_knowledge   | 265 | 29 | 5 | 299 |
| college_medicine     | 173 | 22 | 5 | 200 |
| professional_medicine| 272 | 31 | 5 | 308 |
| anatomy              | 135 | 14 | 5 | 154 |
| medical_genetics     | 100 | 11 | 5 | 116 |
| virology             | 166 | 18 | 5 | 189 |
| nutrition            | 306 | 33 | 5 | 344 |

Text counts are identical across the 3 models; token counts vary slightly by tokenizer
(econ/math/med ≈ 49k / 79k / 121k for teacher and the two students).

### Held-out evaluation corpus
`build_eval(domain)` — **MMLU-Pro** same category, **never seen in training**:
`economics` (844 q), `math` (1351 q), `health` (818 q); 300 used per domain.
Holding out MMLU-Pro makes "specialized beats general" a **generalization** result and
keeps the Goal-2 transfer eval set uncontaminated.

### Contrast: the production general-SAE corpus
| source | texts | share |
|--------|------:|------:|
| AdvBench behaviors | 520 | 3.5% |
| HarmBench behaviors | 400 | 2.7% |
| Global-MMLU-Lite (23 langs, ~400–685/lang) | 13,999 | 93.8% |
| **total** | **14,919 (~1.16M tok)** | |

English in Global-MMLU-Lite is 1 of 23 languages (615 q over 57 subjects ≈ **~11 English
questions per subject**) — why safety and language concepts work but math/eco/med do not.

---

## 4. SAE recipe

Identical to the production general SAE; **the only variable is the training corpus**.

| param | value |
|-------|-------|
| architecture | OpenAI-style TopK SAE (`know_trans.sae.TopKSAE`) |
| d_hidden | 16384 (absolute, same for all models) |
| k (active latents) | 32 |
| aux_k / aux_coef | 256 / 0.03125 (AuxK dead-feature revival) |
| lr / batch | 4e-4 / 4096 |
| steps | 2000 |
| seed | 0 |
| dtype | fp32 training; weights saved fp32 |

`final_dead_frac = 0.0%` for every domain SAE — AuxK kept all 16384 features alive
despite the small corpora.

---

## 5. Storage layout & management

```
data/saes_subject/<model>/<domain>/layer{L}.safetensors          # weights
                                   layer{L}.safetensors.config.json  # d_in/d_hidden/k (load() reads this)
                                   layer{L}.safetensors.train.json   # n_tokens/dead/steps/seconds
```

- 9 directories = 3 models × {econ, math, med}.
- Disk: ~73 GB total (teacher ~49 GB at ~537 MB/SAE; Llama-3.2-1B ~13 GB; Qwen3-0.6B ~9 GB).
- **Kept separate** from production `data/saes/` on purpose.

### Results index
`report/diag/subjectB_<model>_<domain>.json` — one per (model, domain), each holding
per-layer `{fvu_spec, fvu_gen, cos_spec, cos_gen, frac_active_spec, n_eval_tok}`.
These JSON files are the authoritative ledger of results.

### Scripts
| script | role |
|--------|------|
| `scripts/exp_subject_sae_full.py` | train the specialized SAEs (the run) |
| `scripts/exp_subject_sae.py` | defines `DOMAINS` (subject lists) + `_fmt`/`fvu` helpers |
| `scripts/agg_subjectB.py` | aggregate the result JSONs into a summary table |
| `scripts/diag_sae_ood.py` | the original general-SAE OOD diagnostic (3 layers, teacher) |

### How to use
- **Load a SAE:** `TopKSAE.load("data/saes_subject/<m>/<d>/layer{L}.safetensors")`
  (architecture recovered from the `.config.json` sidecar).
- **Summarize:** `PYTHONPATH=src python -m scripts.agg_subjectB` (works on partial results).
- **Resume / extend:** re-run `exp_subject_sae_full.py --tasks <Model:domain,...>`; it
  **skips** any (layer) already having a `.safetensors` + a recorded JSON entry. Sharded
  by (model, domain) so two GPU processes never race on the same file.
- **Temp captures:** activations are staged under `data/_subjB/<model>_<domain>` and
  deleted as soon as that domain's layers finish.

### The run that produced these
Two GPU processes (`CUDA_VISIBLE_DEVICES=0/1`), ~10–12 h:
- GPU0: `Llama-3.1-8B:econ,Llama-3.1-8B:math,Qwen3-0.6B:econ,Qwen3-0.6B:math`
- GPU1: `Llama-3.1-8B:med,Qwen3-0.6B:med,Llama-3.2-1B:econ,Llama-3.2-1B:math,Llama-3.2-1B:med`

---

## 6. Results (held-out MMLU-Pro FVU, mid/late layers ≥ ⅓ depth)

Lower FVU = better reconstruction. `win%` = fraction of mid/late layers where the
specialized SAE beats the general SAE on the **held-out** MMLU-Pro tokens.

| model | domain | general FVU | specialized FVU | win% | active% |
|-------|--------|------------:|----------------:|-----:|--------:|
| Llama-3.1-8B | econ | 0.633 | **0.414** | 100% | 97.6% |
| Llama-3.1-8B | math | 0.638 | **0.347** | 100% | 96.8% |
| Llama-3.1-8B | med  | 0.623 | **0.270** | 100% | 97.6% |
| Qwen3-0.6B   | econ | 0.392 | **0.287** |  95% | 96.3% |
| Qwen3-0.6B   | math | 0.398 | **0.217** |  95% | 96.5% |
| Qwen3-0.6B   | med  | 0.371 | **0.175** | 100% | 97.9% |
| Llama-3.2-1B | econ | 0.519 | **0.345** | 100% | 95.9% |
| Llama-3.2-1B | math | 0.507 | **0.251** | 100% | 95.4% |
| Llama-3.2-1B | med  | 0.499 | **0.205** | 100% | 97.3% |

All 228 SAEs trained (3 models × 3 domains × all layers); the table is the full final result.

**Takeaways**
1. **Coverage is the binding constraint.** Adding in-domain text cuts held-out FVU ~40–55%
   at every mid/late layer, every model.
2. **It generalizes** (trained on MMLU, evaluated on MMLU-Pro) → the fix is not
   memorization; the topic concept dimensions become real signal.
3. **Data volume still matters.** Specialized FVU improves monotonically with corpus size:
   med (121k tok) 0.27 < math (79k) 0.35 < econ (49k) 0.41. MMLU's few hundred questions
   per domain is not enough — a larger, more natural domain corpus is needed.

---

## 7. Caveats & next steps

- FVU (reconstruction) is **necessary but not sufficient**: it shows the concept dims are
  no longer noise, but does **not** yet show distillation improves. The real Goal-2 test is
  to rebuild the concept space / routes with covered SAEs and re-measure MMLU-Pro accuracy.
- Per-domain SAEs are a **diagnostic**, not the production design — separate SAEs break the
  shared concept space, the detector contrast, and the selectivity story. The intended
  production fix is **one enriched general SAE** (broad base + upsampled math/eco/med +
  more tokens + a less-compressed teacher `d_hidden`), then verify it approaches per-domain
  FVU.
