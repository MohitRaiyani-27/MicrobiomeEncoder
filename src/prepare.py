"""
Shared data preparation: load microbiome-only data + splits, apply the CLR
transform (fit on TRAIN only), and return ready-to-use arrays.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src.transforms import CLRTransform


def load_split_arrays(fit_transform=True, transform_path=None):
    df = pd.read_csv(C.DATA_CSV)
    sp = pd.read_csv(C.SPLIT_CSV)
    assert len(df) == len(sp), "splits and data length mismatch"

    feat_cols = [c for c in df.columns if c not in (C.TARGET, C.GROUP)]
    X_all = df[feat_cols].to_numpy(dtype=np.float64)

    # disease label encoding (stable, sorted) over ALL classes
    classes = sorted(df[C.TARGET].unique())
    cls2id = {c: i for i, c in enumerate(classes)}
    y_all = df[C.TARGET].map(cls2id).to_numpy()

    split = sp["split"].to_numpy()
    sid_all = sp["study_id"].to_numpy()

    tr = split == "train"
    va = split == "val"
    te = split == "test"

    # transform
    if fit_transform:
        tf = CLRTransform(use_clr=C.USE_CLR).fit(X_all[tr], feat_cols)
        tf.save(C.ARTIFACTS / "clr_transform")
    else:
        tf = CLRTransform.load(transform_path or (C.ARTIFACTS / "clr_transform"))

    Xtr = tf.transform(X_all[tr])
    Xva = tf.transform(X_all[va])
    Xte = tf.transform(X_all[te])

    # remap train study ids to a dense 0..K-1 range; single-study-disease
    # samples carry -1 and stay -1 (ignored by the adversary).
    train_sids = sid_all[tr]
    uniq = sorted(set(int(s) for s in train_sids if s >= 0))
    remap = {s: i for i, s in enumerate(uniq)}
    sid_tr = np.array([remap[int(s)] if int(s) >= 0 else -1 for s in train_sids],
                      dtype=np.int64)
    n_studies = len(uniq)

    meta = {
        "classes": classes,
        "n_classes": len(classes),
        "feature_names": feat_cols,
        "n_features": len(feat_cols),
        "n_train_studies": n_studies,
    }
    with open(C.ARTIFACTS / "prepare_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "Xtr": Xtr, "ytr": y_all[tr], "sid_tr": sid_tr,
        "Xva": Xva, "yva": y_all[va],
        "Xte": Xte, "yte": y_all[te],
        "meta": meta, "transform": tf,
        "df": df, "split": split,
    }


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return w.astype(np.float32)


def sample_weights(y, n_classes, temperature=1.0):
    """Per-sample weights for the WeightedRandomSampler.

    temperature controls how aggressively rare classes are up-weighted:
      0.0 -> uniform (natural class frequencies preserved)
      1.0 -> full inverse frequency (every class equally likely)
      0.5 -> square-root balancing (rare classes boosted, common ones survive)
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = (1.0 / counts) ** float(temperature)
    return inv[y].astype(np.float64)
