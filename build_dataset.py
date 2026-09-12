"""
Build a CLEAN microbiome-only dataset.

Goal
----
The original 30-class model achieved high accuracy by exploiting metadata leakage
and study/batch confounding (see diagnosis). This script strips ALL metadata and
keeps ONLY the microbiome taxonomic abundances plus:
  - `disease`     : the prediction target
  - `study_name`  : kept ONLY for group-aware (leave-study-out) splitting,
                    never used as a model feature.

Everything else (age, country, sequencing platform, read counts, clinical labs,
study_condition, subject_id, ...) is dropped, because those are the columns the
old model cheated on.

Output
------
data/microbiome_only.csv   -> study_name, disease, <microbiome features...>
data/feature_list.json     -> list of microbiome feature column names
data/study_disease.csv     -> #samples per (study, disease) for reference
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "fall_data" / "dataset_30class.csv"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

# The first taxonomic column in the raw file. Everything from here on is
# microbiome (species:/genus:/bare taxa names); everything before is metadata.
FIRST_TAXON_COL = "species:Escherichia coli"

META_KEEP = ["study_name", "disease"]  # study_name kept only for splitting


def main():
    print(f"Loading {SRC} ...")
    df = pd.read_csv(SRC, low_memory=False)
    print(f"  {len(df)} samples x {df.shape[1]} columns")

    cols = list(df.columns)
    if FIRST_TAXON_COL not in cols:
        raise SystemExit(f"Could not find '{FIRST_TAXON_COL}' in columns.")
    start = cols.index(FIRST_TAXON_COL)
    micro_cols = cols[start:]
    print(f"  microbiome feature columns: {len(micro_cols)} (from '{FIRST_TAXON_COL}')")

    # Sanity: none of the kept metadata should be inside the microbiome block
    assert "disease" in cols[:start], "disease unexpectedly in microbiome block"
    assert "study_name" in cols[:start], "study_name unexpectedly in microbiome block"

    keep = META_KEEP + micro_cols
    out = df[keep].copy()

    # Coerce every microbiome column to numeric relative abundance; missing -> 0
    for c in micro_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[micro_cols] = out[micro_cols].fillna(0.0)

    # Drop microbiome features that are entirely zero (carry no information)
    nonzero = (out[micro_cols] != 0).any(axis=0)
    dropped = [c for c in micro_cols if not nonzero[c]]
    micro_cols = [c for c in micro_cols if nonzero[c]]
    if dropped:
        out = out.drop(columns=dropped)
    print(f"  dropped {len(dropped)} all-zero features; kept {len(micro_cols)}")

    # Basic target hygiene
    out["disease"] = out["disease"].astype(str)
    out["study_name"] = out["study_name"].astype(str)

    out.to_csv(OUT / "microbiome_only.csv", index=False)
    with open(OUT / "feature_list.json", "w") as f:
        json.dump({"n_features": len(micro_cols), "features": micro_cols}, f, indent=2)

    sd = (out.groupby(["disease", "study_name"]).size()
          .reset_index(name="n").sort_values(["disease", "n"], ascending=[True, False]))
    sd.to_csv(OUT / "study_disease.csv", index=False)

    n_studies_per_disease = out.groupby("disease")["study_name"].nunique()
    multi = n_studies_per_disease[n_studies_per_disease >= 2].index.tolist()
    print("\nSummary")
    print(f"  samples           : {len(out)}")
    print(f"  microbiome features: {len(micro_cols)}")
    print(f"  diseases           : {out['disease'].nunique()}")
    print(f"  studies            : {out['study_name'].nunique()}")
    print(f"  multi-study diseases (>=2 studies, testable cross-study): {len(multi)}")
    print(f"    {sorted(multi)}")
    print(f"\nSaved -> {OUT/'microbiome_only.csv'}")


if __name__ == "__main__":
    main()
