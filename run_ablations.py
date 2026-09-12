"""
CHUNKING ABLATION STUDY.

Reviewers asked us to justify the "chunked tokenisation" design instead of just
asserting it. This script trains the SAME classifier on the SAME fixed split,
changing exactly ONE design choice at a time, and reports the honest metrics.

We test:

  1. CHUNK SIZE  -> number of chunk tokens: 8, 16, 32, 64(baseline), 128, 256,
                    and 1-feature-per-token (num_chunks = n_features, i.e. NO
                    chunking -- the design the reviewers doubted).
  2. POSITIONAL  -> positional encoding ON (baseline) vs OFF.
  3. POOLING     -> attention pooling (baseline) vs mean pooling.

Every run is trained FROM SCRATCH (no pretraining): the pretrained encoder is
tied to 64 chunks and cannot be reused at other chunk counts, and we already
showed pretraining does not help. So this keeps every variant on equal footing.

The architecture files are NOT modified. We swap pieces at runtime through the
optional `model_robust.POST_BUILD_HOOK`, so the real model stays byte-identical.

Run (on the GPU server):
    python run_ablations.py
"""
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src import model_robust
from Complete_Architecture.model_pretrain import PositionalEncoding

SEED = 42
# a checkpoint path that will NEVER exist -> train.py inits from scratch
NO_PRETRAIN = C.CKPT_DIR / "__no_pretrain_sentinel__.pt"

KEY_INTERNAL = ["CRC", "IBD", "T2D"]
KEY_EXTERNAL = ["YangJ_2020", "HeQ_2017", "XuQ_2021"]


# ------------------------------------------------------------------ runtime patches
class MeanPool(nn.Module):
    """Drop-in replacement for AttentionPooling: plain average over tokens."""

    def forward(self, encoder_output):
        pooled = encoder_output.mean(dim=1)
        b, t, _ = encoder_output.shape
        attn = torch.full((b, t), 1.0 / t,
                          device=encoder_output.device, dtype=pooled.dtype)
        return pooled, attn


def hook_chunks(num_chunks):
    """Baseline design at a given chunk count. Give positional encoding enough
    length to cover large chunk counts (sinusoidal values for the first N
    positions are identical to the original, so 64 stays comparable)."""
    def _h(model):
        model.pos_encoding = PositionalEncoding(model.hidden_dim,
                                                max_len=max(256, num_chunks + 8))
        return model
    return _h


def hook_no_posenc(model):
    model.pos_encoding = nn.Identity()
    return model


def hook_mean_pool(model):
    model.attention_pooling = MeanPool()
    return model


# ------------------------------------------------------------------ run one variant
def collect(ev):
    o = ev["internal_overall"]
    row = {
        "acc": o["acc"],
        "bal_acc": o["bal_acc"],
        "macro_f1": o["macro_f1"],
        "top3_acc": o["top3_acc"],
    }
    pd_ = ev.get("internal_per_disease", {})
    for d in KEY_INTERNAL:
        row[f"int_{d}_top1"] = pd_.get(d, {}).get("top1", float("nan"))
    ext = ev.get("external", {})
    for stem in KEY_EXTERNAL:
        row[f"ext_{stem}_top1"] = ext.get(stem, {}).get("top1", float("nan"))
        row[f"ext_{stem}_top3"] = ext.get(stem, {}).get("top3", float("nan"))
    return row


def run_variant(name, num_chunks, hook, train_mod, eval_mod):
    C.NUM_CHUNKS = num_chunks
    C.PRETRAIN_CKPT = NO_PRETRAIN            # always from scratch
    model_robust.POST_BUILD_HOOK = hook
    C.SEED = SEED
    C.set_seed(SEED)

    print("\n" + "=" * 74)
    print(f"[ablation] {name}   num_chunks={num_chunks}")
    print("=" * 74)

    train_mod.main()
    eval_mod.main()

    src = C.RESULTS_DIR / "evaluation.json"
    dst = C.RESULTS_DIR / f"ablation_{name}.json"
    shutil.copy(src, dst)
    with open(dst) as f:
        ev = json.load(f)
    model_robust.POST_BUILD_HOOK = None      # reset
    return collect(ev)


def main():
    # per-feature control = one token per feature (NO chunking)
    cols = pd.read_csv(C.DATA_CSV, nrows=0).columns
    n_features = len(cols) - 2               # minus study_name, disease
    print(f"[ablation] dataset has {n_features} microbiome features")

    import train as train_mod
    import evaluate as eval_mod

    # (name, num_chunks, hook)
    variants = [
        # ---- chunk-size sweep (attention pooling + positional encoding) ----
        ("chunks_8",   8,   hook_chunks(8)),
        ("chunks_16",  16,  hook_chunks(16)),
        ("chunks_32",  32,  hook_chunks(32)),
        ("chunks_64",  64,  hook_chunks(64)),      # <-- baseline design
        ("chunks_128", 128, hook_chunks(128)),
        ("chunks_256", 256, hook_chunks(256)),
        ("per_feature", n_features, hook_chunks(n_features)),  # no chunking
        # ---- design ablations (all at 64 chunks) ----
        ("posenc_off", 64, hook_no_posenc),
        ("mean_pool",  64, hook_mean_pool),
    ]

    rows = {}
    for name, nc, hook in variants:
        try:
            rows[name] = run_variant(name, nc, hook, train_mod, eval_mod)
        except Exception as e:  # keep going even if one variant fails
            print(f"[ablation] !! variant {name} FAILED: {e}")
            rows[name] = {"error": str(e)}

    out = C.RESULTS_DIR / "ablations.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    # ---- pretty table ----
    print("\n" + "=" * 74)
    print("CHUNKING ABLATION SUMMARY (from scratch, same split, seed=42)")
    print("=" * 74)
    hdr = f"{'variant':14s} {'acc':>6s} {'bal':>6s} {'mF1':>6s} {'top3':>6s} " \
          f"{'CRC':>6s} {'IBD':>6s} {'T2D':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for name, _, _ in variants:
        r = rows.get(name, {})
        if "error" in r:
            print(f"{name:14s}  FAILED: {r['error'][:40]}")
            continue
        print(f"{name:14s} {r['acc']:6.3f} {r['bal_acc']:6.3f} {r['macro_f1']:6.3f} "
              f"{r['top3_acc']:6.3f} {r['int_CRC_top1']:6.3f} "
              f"{r['int_IBD_top1']:6.3f} {r['int_T2D_top1']:6.3f}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
