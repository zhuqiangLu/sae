"""Train pilot SAEs for one (role, seed) into <data>/saes/<name>/seed{N}/.

Driven directly (not via cli train-sae) so we can train MULTIPLE TEACHER SEEDS
into separate dirs for the cross-seed stability gate. Matched dictionary size
(sae.d_hidden) is set in the config and applies to both models.

Run:  PYTHONPATH=src python3 scripts/train_pilot_saes.py --role teacher --seed 0 [--steps N]
"""
from __future__ import annotations
import argparse, os
from dataclasses import replace

import torch
torch.backends.cuda.matmul.allow_tf32 = True   # ~2x faster matmuls, negligible SAE impact
torch.backends.cudnn.allow_tf32 = True

from know_trans.config import load_config
from know_trans.capture import ActivationReader
from know_trans.sae import train_sae
from know_trans.utils import ensure_dir, get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pair_llama8b_qwen0p6b_pilot.yaml")
    ap.add_argument("--role", required=True, choices=["teacher", "student"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--layers", type=str, default=None, help="comma list; default all captured")
    a = ap.parse_args()

    cfg = load_config(a.config)
    mcfg = cfg.teacher if a.role == "teacher" else cfg.student
    sae_cfg = cfg.sae if a.steps is None else replace(cfg.sae, steps=a.steps)

    act_dir = os.path.join(cfg.paths.data, "activations", mcfg.name)
    reader = ActivationReader(act_dir)
    layers = ([int(x) for x in a.layers.split(",")] if a.layers
              else sorted(int(l) for l in reader.layers))
    out_root = ensure_dir(os.path.join(cfg.paths.data, "saes", mcfg.name, f"seed{a.seed}"))

    log = get_logger("train_pilot")
    log.info("[%s seed%d] model=%s layers=%s d_hidden=%s steps=%d -> %s",
             a.role, a.seed, mcfg.name, layers, sae_cfg.d_hidden, sae_cfg.steps, out_root)
    for L in layers:
        out = os.path.join(out_root, f"layer{L}.safetensors")
        log.info("[%s seed%d] training layer %d -> %s", a.role, a.seed, L, out)
        train_sae(reader, L, sae_cfg, a.seed, out)
    log.info("[%s seed%d] DONE -> %s", a.role, a.seed, out_root)


if __name__ == "__main__":
    main()
