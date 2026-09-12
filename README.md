# MicrobiomeEncoder

A leakage-free deep-learning model that classifies human disease from gut-microbiome
species profiles. It has two stages: a self-supervised **pre-training** stage that learns
species patterns by predicting masked chunks, and a supervised **fine-tuning** stage that
predicts one of 30 diseases. Everything here is microbiome-only — no metadata columns are
used, so there is no label leakage.

## What the model does

- Input: relative abundance of 1,207 bacterial species per stool sample (CLR-transformed).
- The 1,207 species are split into 64 chunks, embedded as tokens, and passed through a
  4-layer Transformer encoder.
- Output: probability over 30 diseases.
- Dataset: 4,046 samples, 47 studies, 30 diseases (majority class T2D = 22.0%).

## Headline results (5 seeds, mean +/- std)

| Metric | Score |
|---|---|
| Cross-study accuracy | 23.3% +/- 1.0 |
| Balanced accuracy | 29.5% +/- 2.3 |
| Macro-F1 | 0.220 +/- 0.020 |
| Top-3 accuracy | 54.8% +/- 3.0 |
| Macro-AUC (20 reliable classes, n_pos >= 5) | 0.70 |
| External CRC cohort (YangJ_2020) top-1 | 68.7% +/- 5.4 |

Note: accuracy is close to the 22.0% majority-class chance level, so the AUC (0.70) and the
external CRC transfer (68.7%) are the more meaningful signals. Self-supervised pre-training
did **not** improve results over training from scratch — we report this honestly.

## Repository layout

```
MicrobiomeEncoder/
├── README.md
├── requirements.txt
├── .gitignore
├── config_exp.py              # all paths + hyper-parameters (single source of truth)
│
├── build_dataset.py           # build the microbiome-only CSV
├── make_splits.py             # cross-study train/val/test split
├── pretrain.py                # stage 1: self-supervised masked-chunk pre-training
├── train.py                   # stage 2: fine-tune the 30-class classifier
├── evaluate.py                # internal test + external validation
│
├── run_all.sh                 # runs the whole pipeline end-to-end
├── run_seeds.py               # repeat over 5 seeds (scratch vs pre-trained)
├── run_auc.py                 # per-class one-vs-rest AUC
├── run_crossstudy.py          # leave-one-study-out cross-study evaluation
├── run_interpret.py           # Integrated-Gradients feature attributions
├── run_ablations.py           # chunk-count / pooling / pos-enc ablations
│
├── src/                       # model + data code
│   ├── model_robust.py        # the encoder + classifier head
│   ├── prepare.py             # load splits into arrays
│   ├── data.py                # Dataset / collate
│   ├── transforms.py          # CLR transform (fit on train only)
│   ├── augment.py             # feature-dropout / noise / mixup
│   ├── losses.py              # focal loss
│   └── grl.py                 # gradient-reversal (adversarial study-invariance)
│
├── data/                      # input data (see below)
│   ├── microbiome_only.csv    # 4,046 x 1,207 species (20 MB)
│   ├── splits.csv             # sample -> train/val/test + study id
│   ├── feature_list.json      # ordered species names
│   └── study_disease.csv      # study -> disease map
│
├── external_data/             # external validation cohorts (reprocessed FASTQ, MetaPhlAn3)
│   ├── YangJ_2020_abundance.csv   + _metadata.csv   (CRC)
│   ├── HeQ_2017_abundance.csv     + _metadata.csv   (IBD)
│   └── XuQ_2021_abundance.csv     + _metadata.csv   (T2D)
│
├── artifacts/                 # saved CLR transform + meta (needed to reload the model)
│   ├── clr_transform.npz
│   ├── clr_transform.features.json
│   ├── prepare_meta.json
│   └── split_meta.json
│
├── checkpoints/               # trained weights (~45 MB total)
│   ├── pretrain_microbiome.pt
│   └── robust_classifier.pt
│
├── results/                   # metrics as JSON (per_class_auc.json, seeds_summary.json, ...)
│
└── paper_figures/             # figure code + PNGs used in the paper
    ├── make_figures.py
    └── fig1..fig9 *.png
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
cd MicrobiomeEncoder
bash run_all.sh                    # build data -> split -> pretrain -> fine-tune -> evaluate
```

Or run one stage at a time:

```bash
python make_splits.py
python pretrain.py
python train.py
python evaluate.py
```

Extra analyses:

```bash
python run_seeds.py                # 5-seed comparison (scratch vs pre-trained)
python run_auc.py                  # per-class AUC  -> results/per_class_auc.json
python run_crossstudy.py           # cross-study evaluation
python run_interpret.py            # Integrated-Gradients attributions
python paper_figures/make_figures.py   # regenerate all paper figures
```

## Notes on the data files

- `data/microbiome_only.csv` is the only large data file (20 MB) and is well under the
  GitHub 100 MB per-file limit, so it can be committed directly.
- `checkpoints/*.pt` are ~22–23 MB each. They are also fine for GitHub. If you prefer a
  lighter repo, remove them (uncomment the last line in `.gitignore`) and attach them to a
  GitHub Release instead.
- `external_data/` only needs the three `_abundance.csv` files to reproduce external
  validation; the `_metadata.csv` files are included for reference.
