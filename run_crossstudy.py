"""
Clean experiment: does the MICROBIOME ALONE predict disease,
and does it GENERALISE ACROSS STUDIES?

Two evaluation protocols, identical models & features:

  1. RANDOM     - StratifiedKFold. Train/test samples can come from the SAME
                  study. This is the "in-distribution" setting the old model
                  effectively used. Study/batch signatures leak in -> optimistic.

  2. CROSS-STUDY- StratifiedGroupKFold grouped by `study_name`. Test studies are
                  DISJOINT from training studies. This is the honest test of
                  whether microbiome signal generalises to a new cohort.

Features: microbiome relative abundances ONLY (no metadata).
Models   : Logistic Regression (log1p + standardise) and Random Forest.

The headline result is the GAP between protocol 1 and protocol 2.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, top_k_accuracy_score)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "microbiome_only.csv"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

N_SPLITS = 5
SEED = 42


def make_models():
    logreg = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=1.0, n_jobs=-1)),
    ])
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)
    return {"logreg": logreg, "random_forest": rf}


def oof_predictions(X, y, groups, protocol, model):
    """Return out-of-fold predicted labels and top-3 predicted label indices."""
    n = len(y)
    classes = np.unique(y)
    pred = np.empty(n, dtype=object)
    top3 = [None] * n

    if protocol == "random":
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X, y)
    else:  # cross-study
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X, y, groups)

    for tr, te in split_iter:
        m = make_models()[model] if isinstance(model, str) else model
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        pred[te] = p
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X[te])
            cls = m.classes_
            order = np.argsort(-proba, axis=1)[:, :3]
            for i, row in zip(te, order):
                top3[i] = [cls[j] for j in row]
    return pred, top3


def metrics(y_true, y_pred, top3, multi_mask):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    top3_acc = np.mean([yt in (t3 or []) for yt, t3 in zip(y_true, top3)])
    # restricted to multi-study diseases (cross-study is only meaningful there)
    if multi_mask.any():
        acc_m = accuracy_score(y_true[multi_mask], y_pred[multi_mask])
        bal_m = balanced_accuracy_score(y_true[multi_mask], y_pred[multi_mask])
        mf1_m = f1_score(y_true[multi_mask], y_pred[multi_mask], average="macro")
    else:
        acc_m = bal_m = mf1_m = float("nan")
    return dict(accuracy=acc, balanced_accuracy=bal, macro_f1=mf1, top3_accuracy=top3_acc,
                accuracy_multistudy=acc_m, balanced_accuracy_multistudy=bal_m,
                macro_f1_multistudy=mf1_m)


def per_disease_recall(y_true, y_pred):
    out = {}
    for d in np.unique(y_true):
        m = y_true == d
        out[d] = float((y_pred[m] == d).mean())
    return out


def main():
    df = pd.read_csv(DATA)
    feat_cols = [c for c in df.columns if c not in ("study_name", "disease")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["disease"].to_numpy()
    groups = df["study_name"].to_numpy()

    n_studies = df.groupby("disease")["study_name"].nunique()
    multi_diseases = set(n_studies[n_studies >= 2].index)
    multi_mask = np.array([d in multi_diseases for d in y])

    print(f"samples={len(y)}  features={len(feat_cols)}  "
          f"diseases={len(np.unique(y))}  studies={df['study_name'].nunique()}")
    print(f"multi-study diseases (cross-study testable): {len(multi_diseases)}\n")

    results = {}
    for model in ["logreg", "random_forest"]:
        results[model] = {}
        for protocol in ["random", "cross_study"]:
            proto_key = "random" if protocol == "random" else "cross-study"
            print(f"[{model} | {protocol}] running {N_SPLITS}-fold ...")
            pred, top3 = oof_predictions(X, y, groups, proto_key, model)
            m = metrics(y, pred, top3, multi_mask)
            m["per_disease_recall"] = per_disease_recall(y, pred)
            results[model][protocol] = m
            print(f"    acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
                  f"macroF1={m['macro_f1']:.3f}  top3={m['top3_accuracy']:.3f}  "
                  f"| multi-study bal_acc={m['balanced_accuracy_multistudy']:.3f}")

    with open(RES / "crossstudy_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- readable summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY  (microbiome-only, 30 diseases)")
    print("=" * 78)
    hdr = f"{'model':14s} {'protocol':12s} {'acc':>6s} {'bal_acc':>8s} {'macroF1':>8s} {'top3':>6s}"
    print(hdr)
    print("-" * 78)
    for model in results:
        for protocol in results[model]:
            r = results[model][protocol]
            print(f"{model:14s} {protocol:12s} {r['accuracy']:6.3f} {r['balanced_accuracy']:8.3f} "
                  f"{r['macro_f1']:8.3f} {r['top3_accuracy']:6.3f}")
    print("-" * 78)
    print("Interpretation: a large drop from 'random' to 'cross_study' means the model")
    print("was relying on study/batch signatures, not generalisable microbiome signal.")

    # per-disease cross-study recall for the best-represented diseases
    print("\nPer-disease balanced recall (random_forest): random -> cross_study")
    rnd = results["random_forest"]["random"]["per_disease_recall"]
    crs = results["random_forest"]["cross_study"]["per_disease_recall"]
    order = sorted(rnd, key=lambda d: -(d in multi_diseases) * 100 - rnd[d])
    for d in order:
        tag = "multi " if d in multi_diseases else "single"
        print(f"  {d:24s} [{tag}] {rnd[d]:.2f} -> {crs[d]:.2f}")

    print(f"\nSaved -> {RES/'crossstudy_results.json'}")


if __name__ == "__main__":
    main()
