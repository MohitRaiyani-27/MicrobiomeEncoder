"""
STEP 4 — Evaluation.

(1) INTERNAL TEST: the held-out test set (leave-study-out for multi-study
    diseases, 15% for single-study). Reports per-disease top-1/top-3 and overall
    macro-F1 / balanced accuracy.

(2) EXTERNAL VALIDATION: the reprocessed-FASTQ MetaPhlAn3 cohorts
    (CRC=YangJ_2020, IBD=HeQ_2017, T2D=XuQ_2021). Species names are normalised to
    align MetaPhlAn3 taxonomy with the training feature space, transformed with
    the SAME fitted CLR, and scored. Reports matched-feature coverage + accuracy.

Run:  python evaluate.py
"""
import sys
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src.prepare import load_split_arrays
from src.data import collate, MicrobiomeDataset, to_device
from src.model_robust import build_classifier, forward_features
from src.transforms import CLRTransform
from torch.utils.data import DataLoader


def norm_species(name: str) -> str:
    """Normalise a taxon name for cross-taxonomy matching."""
    s = str(name)
    s = re.sub(r"^(species|genus):", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)   # drop punctuation
    return re.sub(r"\s+", " ", s).strip()


def load_model(meta, dev):
    ck = torch.load(C.FINETUNE_CKPT, map_location=dev)
    model = build_classifier(meta["feature_names"], meta["n_classes"], C).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck["classes"]


def topk_report(model, X, y, classes, dev, k=3):
    ds = MicrobiomeDataset(X, y, np.full_like(y, -1))
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    logits_all = []
    with torch.no_grad():
        for cat, num, yy, sid in dl:
            cat, num = to_device(cat, num, dev)
            logits, _, _ = forward_features(model, cat, num)
            logits_all.append(logits.cpu().numpy())
    logits = np.concatenate(logits_all)
    top1 = logits.argmax(1)
    topk = np.argsort(-logits, 1)[:, :k]
    return top1, topk, logits


def main():
    dev = torch.device(C.DEVICE)
    data = load_split_arrays(fit_transform=False)     # reuse fitted transform
    meta = data["meta"]
    model, classes = load_model(meta, dev)
    cls2id = {c: i for i, c in enumerate(classes)}

    # ---------------- (1) internal test ----------------
    top1, top3, _ = topk_report(model, data["Xte"], data["yte"], classes, dev)
    yte = data["yte"]
    overall = {
        "acc": float(accuracy_score(yte, top1)),
        "bal_acc": float(balanced_accuracy_score(yte, top1)),
        "macro_f1": float(f1_score(yte, top1, average="macro")),
        "top3_acc": float(np.mean([yte[i] in top3[i] for i in range(len(yte))])),
    }
    per_disease = {}
    for d, di in cls2id.items():
        m = yte == di
        if m.sum() == 0:
            continue
        per_disease[d] = {
            "n": int(m.sum()),
            "top1": float((top1[m] == di).mean()),
            "top3": float(np.mean([di in top3[i] for i in np.where(m)[0]])),
        }

    print("=" * 70)
    print("INTERNAL TEST (held-out studies / 15% single-study)")
    print("=" * 70)
    print(f"overall  acc={overall['acc']:.3f}  bal_acc={overall['bal_acc']:.3f}  "
          f"macroF1={overall['macro_f1']:.3f}  top3={overall['top3_acc']:.3f}\n")
    print(f"{'disease':24s} {'n':>4s} {'top1':>6s} {'top3':>6s}")
    for d in sorted(per_disease, key=lambda x: -per_disease[x]['top1']):
        r = per_disease[d]
        print(f"{d:24s} {r['n']:4d} {r['top1']:6.2f} {r['top3']:6.2f}")

    # ---------------- (2) external validation ----------------
    tf = CLRTransform.load(C.ARTIFACTS / "clr_transform")
    feat = meta["feature_names"]
    norm2idx = {}
    for i, f in enumerate(feat):
        norm2idx.setdefault(norm_species(f), i)   # first wins

    ext_results = {}
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION (reprocessed FASTQ, MetaPhlAn3)")
    print("=" * 70)
    for stem, disease in C.EXTERNAL_STUDIES.items():
        path = C.EXTERNAL_DIR / f"{stem}_abundance.csv"
        if not path.exists():
            print(f"  [skip] {path} missing"); continue
        if disease not in cls2id:
            print(f"  [skip] {disease} not a model class"); continue
        ab = pd.read_csv(path)
        id_col = ab.columns[0]
        spec_cols = [c for c in ab.columns if c != id_col]
        X = np.zeros((len(ab), len(feat)), dtype=np.float64)
        matched = 0
        for c in spec_cols:
            key = norm_species(c)
            j = norm2idx.get(key)
            if j is not None:
                X[:, j] = pd.to_numeric(ab[c], errors="coerce").fillna(0.0).to_numpy()
                matched += 1
        Xt = tf.transform(X)
        di = cls2id[disease]
        yv = np.full(len(ab), di)
        top1, top3, _ = topk_report(model, Xt, yv, classes, dev)
        r = {
            "n": int(len(ab)),
            "matched_features": matched,
            "external_species": len(spec_cols),
            "top1": float((top1 == di).mean()),
            "top3": float(np.mean([di in top3[i] for i in range(len(yv))])),
        }
        ext_results[stem] = {"disease": disease, **r}
        print(f"  {stem:16s} [{disease:4s}] n={r['n']:3d} matched={matched:4d}/"
              f"{len(spec_cols)}  top1={r['top1']:.2f} top3={r['top3']:.2f}")

    with open(C.RESULTS_DIR / "evaluation.json", "w") as f:
        json.dump({"internal_overall": overall, "internal_per_disease": per_disease,
                   "external": ext_results}, f, indent=2)
    print(f"\nSaved -> {C.RESULTS_DIR/'evaluation.json'}")


if __name__ == "__main__":
    main()
