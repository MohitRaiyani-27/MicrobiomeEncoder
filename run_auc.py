"""
COMORBIDITY / PER-CLASS AUC ANALYSIS.

Reviewers asked whether each disease label is a clean single condition or a
composite ("umbrella") label that mixes several conditions, and asked for a
per-class discrimination metric rather than only overall accuracy.

For every disease we compute the ONE-vs-REST ROC-AUC on the held-out internal
test set, using the trained classifier's softmax probability for that disease.
AUC is threshold-free and handles class imbalance, so it is a fair per-class
readout of how well the model separates each disease from all others.

We also flag a small set of composite / multi-condition labels so the paper can
compare single-condition vs composite-label AUC directly.

Run:  python run_auc.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src.prepare import load_split_arrays
from src.model_robust import forward_features
from src.data import collate, MicrobiomeDataset, to_device
from torch.utils.data import DataLoader

# labels that are umbrella / composite terms (mix of conditions), editable
COMPOSITE = {"metabolic_syndrome", "ME/CFS", "ACVD", "IGT"}
# minimum positive test samples for an AUC we call "reliable"
MIN_POS = 5


def load_model(meta, dev):
    from src.model_robust import build_classifier
    ck = torch.load(C.FINETUNE_CKPT, map_location=dev)
    model = build_classifier(meta["feature_names"], meta["n_classes"], C).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck["classes"]


def predict_proba(model, X, y, dev):
    ds = MicrobiomeDataset(X, y, np.full_like(y, -1))
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    logits_all = []
    with torch.no_grad():
        for cat, num, yy, sid in dl:
            cat, num = to_device(cat, num, dev)
            logits, _, _ = forward_features(model, cat, num)
            logits_all.append(logits.cpu().numpy())
    logits = np.concatenate(logits_all)
    # softmax
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p


def main():
    dev = torch.device(C.DEVICE)
    data = load_split_arrays(fit_transform=False)
    meta = data["meta"]
    model, classes = load_model(meta, dev)
    cls2id = {c: i for i, c in enumerate(classes)}

    yte = data["yte"]
    proba = predict_proba(model, data["Xte"], yte, dev)

    per_class = {}
    for disease, di in cls2id.items():
        y_true = (yte == di).astype(int)
        n_pos = int(y_true.sum())
        if n_pos == 0:
            continue                       # class absent from test (LOSO undefined)
        # need both classes present for AUC
        if y_true.min() == y_true.max():
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, proba[:, di]))
        per_class[disease] = {
            "n_pos": n_pos,
            "auc": auc,
            "reliable": bool(n_pos >= MIN_POS and not np.isnan(auc)),
            "composite": disease in COMPOSITE,
        }

    # summaries over the "reliable" classes
    def _mean(items):
        vals = [v["auc"] for v in items if v["reliable"]]
        return float(np.mean(vals)) if vals else float("nan")

    reliable = [v for v in per_class.values() if v["reliable"]]
    single = {k: v for k, v in per_class.items() if not v["composite"]}
    comp = {k: v for k, v in per_class.items() if v["composite"]}

    summary = {
        "macro_auc_reliable": _mean(per_class.values()),
        "n_reliable_classes": len(reliable),
        "macro_auc_single_condition": _mean(single.values()),
        "macro_auc_composite": _mean(comp.values()),
        "min_pos_for_reliable": MIN_POS,
    }

    out = {"summary": summary, "per_class": per_class}
    dst = C.RESULTS_DIR / "per_class_auc.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    # ---- pretty table ----
    print("=" * 62)
    print("PER-CLASS ONE-vs-REST ROC-AUC (internal held-out test)")
    print("=" * 62)
    print(f"{'disease':26s} {'n+':>4s} {'AUC':>6s}  flag")
    print("-" * 62)
    for d in sorted(per_class, key=lambda x: -(per_class[x]['auc']
                                               if not np.isnan(per_class[x]['auc']) else -1)):
        r = per_class[d]
        flags = []
        if r["composite"]:
            flags.append("composite")
        if not r["reliable"]:
            flags.append("low-n")
        auc_s = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "  nan"
        print(f"{d:26s} {r['n_pos']:4d} {auc_s:>6s}  {' '.join(flags)}")

    print("-" * 62)
    print(f"macro-AUC (reliable, n+>={MIN_POS}): {summary['macro_auc_reliable']:.3f}  "
          f"over {summary['n_reliable_classes']} classes")
    print(f"  single-condition macro-AUC: {summary['macro_auc_single_condition']:.3f}")
    print(f"  composite-label  macro-AUC: {summary['macro_auc_composite']:.3f}")
    print(f"\nSaved -> {dst}")


if __name__ == "__main__":
    main()
