"""
Multi-seed experiment: run the fine-tune + evaluate pipeline for several seeds,
BOTH with self-supervised pretraining and from scratch, on the SAME fixed split.

For each seed we save:
    results/seed{S}_pretrain.json   (encoder warm-started from pretrain ckpt)
    results/seed{S}_scratch.json    (encoder trained from random init)

At the end we aggregate mean +/- SD across seeds for the headline metrics and
save results/seeds_summary.json + print a table.

The data split is built ONCE (make_splits) and kept fixed, so the variance we
report is training/initialisation variance (the standard "5 seeds" the reviewers
asked for). Pretraining is done ONCE and reused for every "with-pretraining" seed.

Run (on the GPU server):
    python run_seeds.py
"""
import sys
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C

SEEDS = [42, 1, 2, 3, 4]

# a checkpoint path that will NEVER exist -> forces train.py to init from scratch
NO_PRETRAIN = C.CKPT_DIR / "__no_pretrain_sentinel__.pt"
# the real pretrained encoder produced by pretrain.py / run_all.sh
PRETRAIN = C.CKPT_DIR / "pretrain_microbiome.pt"

# diseases we care about most for the paper
KEY_INTERNAL = ["CRC", "IBD", "T2D"]
KEY_EXTERNAL = ["YangJ_2020", "HeQ_2017", "XuQ_2021"]


def ensure_pretrained():
    """Make sure a pretrained encoder exists (run pretrain once if missing)."""
    if PRETRAIN.exists():
        print(f"[seeds] using existing pretrained encoder: {PRETRAIN}")
        return
    print("[seeds] no pretrained encoder found -> running pretrain.py once ...")
    C.PRETRAIN_CKPT = PRETRAIN
    C.set_seed(SEEDS[0])
    import pretrain
    pretrain.main()


def run_one(seed, use_pretrain, train_mod, eval_mod):
    """One finetune + evaluate run; returns the parsed evaluation dict."""
    C.SEED = seed
    C.PRETRAIN_CKPT = PRETRAIN if use_pretrain else NO_PRETRAIN
    C.set_seed(seed)

    tag = "pretrain" if use_pretrain else "scratch"
    print("\n" + "=" * 70)
    print(f"[seeds] seed={seed}  mode={tag}  pretrain_ckpt_exists={C.PRETRAIN_CKPT.exists()}")
    print("=" * 70)

    train_mod.main()
    eval_mod.main()

    src = C.RESULTS_DIR / "evaluation.json"
    dst = C.RESULTS_DIR / f"seed{seed}_{tag}.json"
    shutil.copy(src, dst)
    with open(dst) as f:
        return json.load(f)


def collect(ev):
    """Pull the headline numbers out of one evaluation.json dict."""
    o = ev["internal_overall"]
    row = {
        "acc": o["acc"],
        "bal_acc": o["bal_acc"],
        "macro_f1": o["macro_f1"],
        "top3_acc": o["top3_acc"],
    }
    pd = ev.get("internal_per_disease", {})
    for d in KEY_INTERNAL:
        row[f"int_{d}_top1"] = pd.get(d, {}).get("top1", float("nan"))
    ext = ev.get("external", {})
    for stem in KEY_EXTERNAL:
        row[f"ext_{stem}_top1"] = ext.get(stem, {}).get("top1", float("nan"))
    return row


def summarise(rows):
    """mean/std across seeds for each metric."""
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out


def main():
    ensure_pretrained()

    import train as train_mod
    import evaluate as eval_mod

    pretrain_rows, scratch_rows = [], []
    for s in SEEDS:
        ev_p = run_one(s, True, train_mod, eval_mod)
        pretrain_rows.append(collect(ev_p))
        ev_s = run_one(s, False, train_mod, eval_mod)
        scratch_rows.append(collect(ev_s))

    summary = {
        "seeds": SEEDS,
        "with_pretraining": summarise(pretrain_rows),
        "no_pretraining": summarise(scratch_rows),
        "per_seed_with_pretraining": pretrain_rows,
        "per_seed_no_pretraining": scratch_rows,
    }
    out_path = C.RESULTS_DIR / "seeds_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- printable table ----------
    metrics = ["acc", "bal_acc", "macro_f1", "top3_acc",
               "int_CRC_top1", "ext_YangJ_2020_top1",
               "ext_HeQ_2017_top1", "ext_XuQ_2021_top1"]
    label = {
        "acc": "Accuracy", "bal_acc": "Balanced acc", "macro_f1": "Macro-F1",
        "top3_acc": "Top-3 acc", "int_CRC_top1": "CRC internal top1",
        "ext_YangJ_2020_top1": "CRC external top1",
        "ext_HeQ_2017_top1": "IBD external top1",
        "ext_XuQ_2021_top1": "T2D external top1",
    }
    print("\n" + "=" * 78)
    print(f"SEED SUMMARY  (mean +/- SD over {len(SEEDS)} seeds)")
    print("=" * 78)
    print(f"{'metric':22s} {'with pretraining':>22s} {'no pretraining':>22s}")
    print("-" * 78)
    for m in metrics:
        p = summary["with_pretraining"][m]
        n = summary["no_pretraining"][m]
        print(f"{label[m]:22s} "
              f"{p['mean']:.3f} +/- {p['std']:.3f}      "
              f"{n['mean']:.3f} +/- {n['std']:.3f}")
    print("-" * 78)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
