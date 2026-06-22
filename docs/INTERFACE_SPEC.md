# know_trans — Interface Contract (authoritative)

SAE-based cross-family knowledge distillation. Every module must match the names/signatures
below exactly so imports line up. Python >= 3.10, type hints + docstrings. Implement only —
never run training or downloads from import; guard heavy work behind functions/CLIs.

## Goal (the science)
**Reading A**: the SAE is the *mechanism* of distillation. Transfer knowledge from a **stronger
teacher** to a **weaker student of a different family** by aligning them in interpretable
**concept space** (SAE features), not raw-activation space. The cross-family concept-correspondence
premise is *ours to prove*. Primary baseline to beat: **Transport-and-Merge** (OT on raw
activations + weight merge, arXiv 2602.05495). Secondary: hidden-state distillation (learned
projector) and logit distillation.

## Concrete pair (defaults; never hardcode dims — infer from data / config.json)
- teacher: `Llama-3.1-8B`, path `/share1/zhlu6105/models/Llama-3.1-8B`, family=llama (d_model 4096, 32 layers)
- student: `Qwen3-0.6B`, path `/share1/zhlu6105/models/Qwen3-0.6B`, family=qwen (d_model 1024, 28 layers)
- **Tokenizers differ across families → never align by token index. Pool to example/span level.**

## Repo layout
```
src/know_trans/{__init__,config,utils,capture,sae,concepts,score,match,distill,cli}.py
configs/pair_llama8b_qwen0p6b.yaml ; scripts/*.py ; report/design_report.html
data/{activations,saes,concepts,feature_scores,matches}/   (outputs; gitignored)
```

## Public API (implement EXACTLY)

### config.py
- `@dataclass ModelCfg(name, path, family, layers: list[int] | str = "all", dtype="bfloat16")`
- `@dataclass SAECfg(expansion=16, k=32, lr=4e-4, batch_size=4096, steps=50000, aux_k=256, aux_coef=1/32, activation="topk")`
- `@dataclass CaptureCfg(corpus="benchmarks", n_texts=20000, max_len=512, batch_size=16, hook_point="mlp")`
- `@dataclass DistillCfg(lambda_concept=1.0, freeze_saes=True, lr=1e-5, steps=2000, pool="mean")`
- `@dataclass Paths(models, benchmarks, data)`
- `@dataclass Config(teacher: ModelCfg, student: ModelCfg, sae: SAECfg, capture: CaptureCfg, distill: DistillCfg, paths: Paths, seed=0)`
- `def load_config(path: str) -> Config`  # parse YAML into the dataclasses

### utils.py
- `set_seed(seed: int) -> None`, `get_device() -> str`, `get_dtype(name: str) -> torch.dtype`
- `load_model_and_tokenizer(path, dtype="bfloat16", device=None) -> tuple[model, tokenizer]`
- `get_logger(name) -> logging.Logger`, `ensure_dir(p) -> str`
- `save_json(obj, path)`, `load_json(path)`, `batched(seq, n) -> Iterator`
- `pool_examples(token_acts: Tensor[T,d], example_ids: LongTensor[T], mode="mean") -> Tensor[E,d]`

### capture.py
- `class MLPHook` — forward hook on `model.model.layers[i].mlp` output.
- `def capture_activations(model, tokenizer, texts, layers, out_dir, batch_size=16, max_len=512, hook_point="mlp", dtype="float16") -> None`
  - writes `out_dir/layer{L}.safetensors` (key `acts`: float16 [T, d_model])
  - writes `out_dir/index.parquet` cols: `row, example_id, tok_idx, char_start, char_end`
  - writes `out_dir/meta.json` `{model, d_model, layers, n_examples, n_tokens, hook_point}`
- `class ActivationReader(model_dir)`: `.layers`, `.meta`, `.read(layer) -> (Tensor[T,d], DataFrame index)`
- Use `return_offsets_mapping=True` to fill char spans → enables span pooling. Stream shards; don't hold all in RAM.

### sae.py (OpenAI-style TopK SAE)
- `class TopKSAE(nn.Module)`: `__init__(d_in, d_hidden, k)`; `encode(x)->(values[B,k], indices[B,k])`;
  `encode_dense(x)->Tensor[B,d_hidden]`; `decode(values, indices)->Tensor[B,d_in]`;
  `forward(x)->dict(recon, codes_dense, values, indices)`; `save(path)`; `@classmethod load(path)->TopKSAE`.
- `def train_sae(reader: ActivationReader, layer, cfg: SAECfg, seed, out_path) -> TopKSAE`
  - recon MSE + AuxK dead-feature revival; unit-norm decoder columns; track dead features, L0.
- `class SAEBundle`: `@classmethod load(dir) -> SAEBundle`; `__getitem__(layer)->TopKSAE`; `.layers`.

### concepts.py
- `@dataclass Concept(name, positives: list[str], hard_negatives: list[str], source, group)`
- `def build_battery_from_benchmarks(benchmarks_dir, out_dir, n_per_concept=200) -> list[Concept]`
  - **First inspect the real files** under `/share1/zhlu6105/benchmarks`:
    - `AdvBench/advbench_behaviors.jsonl` → fields `Behavior`, `target`, `BehaviorID`
    - `HarmBench/metadata.csv` → `Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID`; parquets in `DirectRequest/`, `HumanJailbreaks/`
    - `Global-MMLU-Lite/<lang>/test-*.parquet` → confirm real columns (question/subject/answer/...) before using
  - Concept groups: `safety` (AdvBench+HarmBench behaviors vs benign MMLU questions), `safety_semantic` (one per HarmBench SemanticCategory), `topic_<subject>` (Global-MMLU-Lite subject), `language_<xx>` (per-language dirs). Write one jsonl per concept.
- `def load_battery(dir) -> list[Concept]`, `def save_battery(b, dir) -> None`

### score.py
- `def score_features(model, tokenizer, sae_bundle, battery, layers, out_path, max_len=512, batch_size=16) -> DataFrame`
  - run pos+neg, `encode_dense` per example (pool over spans → [E, d_hidden]), score EACH feature as a
    detector via `sklearn.metrics.roc_auc_score`. cols `[layer, feature, concept, auc, n_pos, n_neg]`; parquet.
- `def concept_feature_sets(df, auc_threshold=0.8, top_k=10) -> dict[str, list[dict]]`
  - concept → `[{layer, feature, auc}]` above threshold (feature splitting ⇒ a SET; discovers best layer).

### match.py
- `def run_matching(teacher_sets, student_sets) -> dict`
  - label-anchored; concept in both → `{concept:{teacher:{layer,features}, student:{layer,features}, shared:true}}`;
    teacher-only → `shared:false` (**headline finding** — log separately).
- `def stability_score(setsA, setsB) -> dict[str,float]`  # Jaccard across SAE seeds
- `def null_control(teacher_sets, student_sets, n_shuffle=100) -> dict`  # shuffled-pair gap
- `def causal_steer_test(teacher, student, t_sae, s_sae, match, prompts) -> dict`  # ablate/steer matched feature via decoder direction; measure behavioral delta. Must actually run.
- `def write_matches(matches, path) -> None`

### distill.py (loss design — read carefully)
Map each model into a shared **concept-activation space** (dim = #shared concepts), sidestepping the
d_hidden mismatch: `concept_act_c(example) = mean over that model's concept-c feature set of encode_dense`.
- `def concept_activations(codes_dense: Tensor[E,d_hidden], concept_sets: dict) -> Tensor[E,C]`
- `def concept_distill_loss(student_concept_acts: Tensor[E,C], teacher_concept_acts: Tensor[E,C]) -> Tensor`  # MSE/smooth-L1, optional per-concept norm
- `class MatchedFeatureDistiller`: `__init__(teacher, student, t_sae, s_sae, matches, cfg)`;
  `compute_loss(batch) -> dict(loss, lm_loss, concept_loss)`  # teacher frozen; both SAEs frozen by cfg; grad flows student only.
- `def train_distill(cfg: Config, matches_path: str) -> None`

### cli.py
argparse subcommands `{capture, train-sae, build-concepts, score, match, distill}`, each loads a `--config` YAML and calls the module functions with correct args.

## Cross-module rules
- Import shared helpers from `know_trans.utils` and `know_trans.config` with the exact names above.
- Tokenizer mismatch ⇒ pooling is always example/span level (use `index.parquet` spans / `example_id`).
- Layer correspondence is **discovered** in score/match (best `(t_layer, s_layer)` per concept), never hardcoded.
- Infer `d_in` from activation shards; `d_hidden = expansion * d_in`.
- safetensors for tensors, parquet for tables, json for small structured outputs.
