#!/usr/bin/env bash
# Robust Microbiome Disease Classifier — full pipeline (run on the GPU server).
#
# Usage:
#   cd Clean_Experiments
#   bash run_all.sh
#
# Requirements: the SAME python env that has torch + sklearn + pandas.
# On the server this is your training env; on the Mac use the repo .venv.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"     # override with:  PYTHON=/path/to/python bash run_all.sh

echo "==> [0/4] Build microbiome-only dataset (if missing)"
[ -f data/microbiome_only.csv ] || "$PY" build_dataset.py

echo "==> [1/4] Make train/val/test split"
"$PY" make_splits.py

echo "==> [2/4] Self-supervised pretraining (masked chunks)"
"$PY" pretrain.py

echo "==> [3/4] Fine-tune classifier (focal + adversarial + augment + mixup)"
"$PY" train.py

echo "==> [4/4] Evaluate: internal test + external validation"
"$PY" evaluate.py

echo "==> DONE. Results in results/  (evaluation.json, finetune_history.json)"
