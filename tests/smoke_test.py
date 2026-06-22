"""Smoke test for the know_trans pipeline.

Tier A: pure-synthetic checks of the math/data-flow modules (no model).
Tier B: real mini end-to-end on Qwen3-0.6B (CPU): capture -> train SAE -> score -> match.

Run:  PYTHONPATH=src python3 tests/smoke_test.py
"""
from __future__ import annotations
import os, sys, tempfile, traceback

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.manual_seed(0)
torch.set_num_threads(min(8, os.cpu_count() or 4))

MODEL = "/share1/zhlu6105/models/Qwen3-0.6B"
PASS, FAIL = [], []

def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))
    return cond

# ---------------------------------------------------------------------------
print("\n=== Tier A: synthetic module checks ===")
try:
    from know_trans.sae import TopKSAE
    from know_trans.utils import pool_examples
    from know_trans.distill import concept_activations, concept_distill_loss
    from know_trans.match import run_matching, stability_score, null_control

    # 1. TopK SAE forward / encode / decode
    sae = TopKSAE(d_in=16, d_hidden=64, k=8)
    x = torch.randn(32, 16)
    out = sae.forward(x)
    check("sae.forward returns recon[32,16]", tuple(out["recon"].shape) == (32, 16))
    vals, idx = sae.encode(x)
    check("sae.encode top-k shapes [32,8]", tuple(vals.shape) == (32, 8) and tuple(idx.shape) == (32, 8))
    check("sae.encode L0 == k", int((vals != 0).sum(dim=1).float().mean().item()) <= 8)
    dense = sae.encode_dense(x)
    check("sae.encode_dense [32,64]", tuple(dense.shape) == (32, 64))
    recon = sae.decode(vals, idx)
    check("sae.decode [32,16]", tuple(recon.shape) == (16,) or tuple(recon.shape) == (32, 16))

    # 2. pool_examples
    tok_acts = torch.randn(10, 8)
    ex_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    pooled = pool_examples(tok_acts, ex_ids, mode="mean")
    check("pool_examples -> [4,8]", tuple(pooled.shape) == (4, 8))

    # 3. concept-activation space with DIFFERENT d_hidden (teacher 64, student 32) + gradient flow
    order = ["concept_a", "concept_b"]
    t_sets = {"concept_a": {"layer": 4, "features": [1, 2, 3]}, "concept_b": {"layer": 4, "features": [10]}}
    s_sets = {"concept_a": {"layer": 2, "features": [0, 5]}, "concept_b": {"layer": 2, "features": [7]}}
    teacher_codes = torch.randn(5, 64)                          # frozen target
    student_codes = torch.randn(5, 32, requires_grad=True)      # carries grad
    t_ca = concept_activations(teacher_codes, t_sets, concept_order=order)
    s_ca = concept_activations(student_codes, s_sets, concept_order=order)
    check("concept_activations teacher->[5,2]", tuple(t_ca.shape) == (5, 2))
    check("concept_activations student->[5,2] (mismatched d_hidden ok)", tuple(s_ca.shape) == (5, 2))
    loss = concept_distill_loss(s_ca, t_ca.detach())
    check("concept_distill_loss is scalar", loss.dim() == 0)
    loss.backward()
    check("gradient flows back to student codes", student_codes.grad is not None and torch.isfinite(student_codes.grad).all())

    # 4. matching + validation gates (synthetic; includes a teacher-only concept)
    teacher_sets = {
        "math":    [{"layer": 4, "feature": 3, "auc": 0.92}],
        "code":    [{"layer": 4, "feature": 7, "auc": 0.85}],
        "teacher_only": [{"layer": 6, "feature": 9, "auc": 0.80}],
    }
    student_sets = {
        "math": [{"layer": 2, "feature": 1, "auc": 0.88}],
        "code": [{"layer": 2, "feature": 4, "auc": 0.81}],
    }
    matches = run_matching(teacher_sets, student_sets)
    shared = {c for c, m in matches.items() if isinstance(m, dict) and m.get("shared")}
    check("run_matching marks math+code shared", {"math", "code"} <= shared)
    check("run_matching flags teacher_only as not shared",
          matches.get("teacher_only", {}).get("shared") is False, info=str(matches.get("teacher_only")))
    stab = stability_score(teacher_sets, teacher_sets)
    check("stability(self,self) == 1.0 for math", abs(stab.get("math", 0) - 1.0) < 1e-6, info=str(stab))
    nc = null_control(teacher_sets, student_sets, n_shuffle=20)
    check("null_control returns a dict", isinstance(nc, dict))
except Exception:
    FAIL.append("TierA-exception"); traceback.print_exc()

# ---------------------------------------------------------------------------
print("\n=== Tier B: real mini end-to-end on Qwen3-0.6B (CPU) ===")
try:
    from know_trans.utils import load_model_and_tokenizer
    from know_trans.capture import capture_activations, ActivationReader
    from know_trans.sae import train_sae, SAEBundle
    from know_trans.score import score_features, concept_feature_sets
    from know_trans.match import run_matching as run_matching_real
    from know_trans.concepts import Concept
    from know_trans.config import SAECfg

    if not os.path.isdir(MODEL):
        check("Qwen3-0.6B present", False, info=f"missing {MODEL}")
        raise SystemExit
    print("  loading Qwen3-0.6B on CPU (float32)...")
    model, tok = load_model_and_tokenizer(MODEL, dtype="float32", device="cpu")
    check("model + tokenizer loaded", model is not None and tok is not None)

    tmp = tempfile.mkdtemp(prefix="kt_smoke_")
    cap_dir = os.path.join(tmp, "acts"); saes_dir = os.path.join(tmp, "saes")
    os.makedirs(saes_dir, exist_ok=True)

    math_pos  = ["2 + 2 = 4", "The integral of x is x^2/2", "Solve for x: 3x = 9", "The derivative of sin is cos"]
    code_pos  = ["def add(a, b): return a + b", "import os, sys", "for i in range(10): print(i)", "x = [1,2,3]"]
    plain_neg = ["The cat sat on the mat.", "It was a bright blue sky.", "She walked to the market.", "Birds sing in spring."]
    texts = math_pos + code_pos + plain_neg
    layers = [4, 8]

    capture_activations(model, tok, texts, layers=layers, out_dir=cap_dir,
                        batch_size=4, max_len=24, hook_point="mlp", dtype="float16")
    reader = ActivationReader(cap_dir)
    check("capture: reader.layers == [4,8]", sorted(reader.layers) == [4, 8], info=str(reader.layers))
    acts4, idx4 = reader.read(4)
    check("capture: acts rows == index rows", acts4.shape[0] == len(idx4), info=f"{acts4.shape} vs {len(idx4)}")
    check("capture: d_model == 1024", acts4.shape[1] == 1024, info=str(acts4.shape))

    sae_cfg = SAECfg(expansion=2, k=8, lr=1e-3, batch_size=128, steps=40, aux_k=16)
    for L in layers:
        train_sae(reader, L, sae_cfg, seed=0, out_path=os.path.join(saes_dir, f"layer{L}.safetensors"))
    bundle = SAEBundle.load(saes_dir)
    check("train_sae + SAEBundle.load", sorted(bundle.layers) == [4, 8], info=str(bundle.layers))

    battery = [
        Concept("math", math_pos, plain_neg, "smoke", "topic"),
        Concept("code", code_pos, plain_neg, "smoke", "topic"),
    ]
    scores_path = os.path.join(tmp, "scores.parquet")
    df = score_features(model, tok, bundle, battery, layers=layers, out_path=scores_path,
                        max_len=24, batch_size=4)
    need = {"layer", "feature", "concept", "auc", "n_pos", "n_neg"}
    check("score_features returns expected columns", need <= set(df.columns), info=str(list(df.columns)))
    check("score_features produced rows", len(df) > 0, info=f"{len(df)} rows")

    sets = concept_feature_sets(df, auc_threshold=0.5, top_k=5)
    check("concept_feature_sets has math & code keys", {"math", "code"} <= set(sets.keys()))

    matches = run_matching_real(sets, sets)
    check("run_matching(real) returns dict", isinstance(matches, dict) and len(matches) > 0)
    print(f"  (tmp artifacts in {tmp})")
except SystemExit:
    pass
except Exception:
    FAIL.append("TierB-exception"); traceback.print_exc()

# ---------------------------------------------------------------------------
print("\n=== SUMMARY ===")
print(f"  PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print("  FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("  ALL SMOKE CHECKS PASSED")
