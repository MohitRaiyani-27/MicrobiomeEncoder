"""
STEP 1 — Build the train / val / test split.

Rules (agreed):
  * 13 multi-study diseases  -> LEAVE-ONE-STUDY-OUT for TEST.
                                From the remaining studies, take a random 15%
                                of samples as VAL, the rest as TRAIN.
  * 17 single-study diseases -> random 70 / 15 / 15  (train / val / test).

Each sample belongs to exactly one disease, so it is assigned exactly once
under that disease's rule. `study_name` is used ONLY for grouping, never as a
model feature.

Output: data/splits.csv  with columns [row, disease, study_name, split, study_id]
        where study_id is an integer id for TRAIN studies (for the adversarial
        head); val/test/holdout get study_id = -1.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

import config_exp as C

RNG = np.random.default_rng(C.SEED)


def choose_test_study(counts: pd.Series) -> str:
    """Pick the held-out test study: the one whose sample count is closest to
    20% of this disease's total (keeps a sensible test size, leaves >=1 study
    for training)."""
    total = counts.sum()
    target = 0.20 * total
    # never pick a study that is the ONLY study (guaranteed >=2 here)
    diffs = (counts - target).abs()
    return diffs.sort_values(kind="stable").index[0]


def split_single_study(idx: np.ndarray):
    """Random 70/15/15 for a single-study disease's sample indices."""
    idx = idx.copy()
    RNG.shuffle(idx)
    n = len(idx)
    n_test = max(1, int(round(C.TEST_FRACTION_SINGLE * n))) if n >= 3 else 0
    n_val = max(1, int(round(C.VAL_FRACTION_SINGLE * n))) if n >= 3 else 0
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


def main():
    df = pd.read_csv(C.DATA_CSV)
    df = df.reset_index().rename(columns={"index": "row"})
    n_studies = df.groupby(C.TARGET)[C.GROUP].nunique()

    split = pd.Series("train", index=df.index)  # default
    holdout_study = {}

    for disease, sub in df.groupby(C.TARGET):
        rows = sub["row"].to_numpy()
        if n_studies[disease] >= 2:
            # leave one study out for test
            counts = sub[C.GROUP].value_counts()
            test_study = choose_test_study(counts)
            holdout_study[disease] = test_study
            is_test = sub[C.GROUP] == test_study
            test_rows = sub.loc[is_test, "row"].to_numpy()
            remain = sub.loc[~is_test, "row"].to_numpy()
            remain = remain.copy(); RNG.shuffle(remain)
            n_val = max(1, int(round(C.VAL_FRACTION * len(remain))))
            val_rows = remain[:n_val]
            train_rows = remain[n_val:]
        else:
            train_rows, val_rows, test_rows = split_single_study(rows)

        split.loc[df["row"].isin(train_rows)] = "train"
        split.loc[df["row"].isin(val_rows)] = "val"
        split.loc[df["row"].isin(test_rows)] = "test"

    out = df[["row", C.TARGET, C.GROUP]].copy()
    out["split"] = split.values

    # integer study id for the adversarial head — ONLY for TRAIN samples that
    # belong to MULTI-study diseases. For single-study diseases disease==study,
    # so forcing study-invariance there would erase the disease signal; those
    # samples get study_id = -1 (ignored by the adversary).
    multi = set(n_studies[n_studies >= 2].index)
    train_multi = (out["split"] == "train") & (out[C.TARGET].isin(multi))
    train_studies = sorted(out.loc[train_multi, C.GROUP].unique())
    study2id = {s: i for i, s in enumerate(train_studies)}

    def _sid(r):
        if r["split"] == "train" and r[C.TARGET] in multi:
            return study2id.get(r[C.GROUP], -1)
        return -1
    out["study_id"] = out.apply(_sid, axis=1)

    out.to_csv(C.SPLIT_CSV, index=False)
    with open(C.ARTIFACTS / "split_meta.json", "w") as f:
        json.dump({
            "n_train_studies": len(train_studies),
            "study2id": study2id,
            "holdout_test_study": holdout_study,
            "counts": out["split"].value_counts().to_dict(),
        }, f, indent=2)

    # report
    print("Split sizes:")
    print(out["split"].value_counts().to_string())
    print(f"\nTrain studies (for adversarial head): {len(train_studies)}")
    print("\nPer-disease split counts:")
    tab = out.groupby([C.TARGET, "split"]).size().unstack(fill_value=0)
    for c in ["train", "val", "test"]:
        if c not in tab: tab[c] = 0
    tab = tab[["train", "val", "test"]]
    tab["n_studies"] = n_studies
    tab["holdout"] = tab.index.map(lambda d: holdout_study.get(d, "-"))
    print(tab.sort_values("n_studies", ascending=False).to_string())
    # sanity: every disease present in train
    missing = tab.index[tab["train"] == 0].tolist()
    print(f"\nDiseases with 0 train samples (should be none): {missing}")
    print(f"\nSaved -> {C.SPLIT_CSV}")


if __name__ == "__main__":
    C.set_seed()
    main()
