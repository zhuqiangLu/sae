"""All-layer pilot: capture EVERY layer for one model, then train an SAE per
layer (seed0). For the per-knowledge routing curves at full depth resolution.

Run: PYTHONPATH=src python3 scripts/run_all_layer.py --config <cfg> --role teacher --steps 4000
"""
from __future__ import annotations
import argparse, os
from dataclasses import replace
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from know_trans.config import load_config
from know_trans.utils import load_model_and_tokenizer, ensure_dir, get_logger
from know_trans.capture import capture_activations, ActivationReader, _resolve_layers
from know_trans.sae import train_sae
from know_trans.cli import _load_corpus_texts

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--role", required=True, choices=["teacher", "student"])
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-capture", action="store_true")
a = ap.parse_args()

log = get_logger("all_layer")
cfg = load_config(a.config)
mcfg = cfg.teacher if a.role == "teacher" else cfg.student
data = cfg.paths.data
act_dir = os.path.join(data, "activations", mcfg.name)

# ---- 1. capture ALL layers -------------------------------------------------
if not a.skip_capture:
    log.info("[%s] capture ALL layers -> %s", mcfg.name, act_dir)
    texts = _load_corpus_texts(cfg, None, log)
    model, tok = load_model_and_tokenizer(mcfg.path, dtype=mcfg.dtype, device="cuda")
    layers = _resolve_layers(model, "all")
    capture_activations(model, tok, texts, layers, act_dir,
                        batch_size=cfg.capture.batch_size, max_len=cfg.capture.max_len,
                        hook_point=cfg.capture.hook_point)
    del model; torch.cuda.empty_cache()

# ---- 2. train an SAE per captured layer ------------------------------------
reader = ActivationReader(act_dir)
sae_cfg = replace(cfg.sae, steps=a.steps)
out_root = ensure_dir(os.path.join(data, "saes", mcfg.name, f"seed{a.seed}"))
layers = sorted(int(l) for l in reader.layers)
log.info("[%s] training %d layer-SAEs (steps=%d) -> %s", mcfg.name, len(layers), a.steps, out_root)
for L in layers:
    out = os.path.join(out_root, f"layer{L}.safetensors")
    log.info("[%s] SAE layer %d -> %s", mcfg.name, L, out)
    train_sae(reader, L, sae_cfg, a.seed, out)
log.info("[%s] ALL-LAYER DONE (%d SAEs)", mcfg.name, len(layers))
