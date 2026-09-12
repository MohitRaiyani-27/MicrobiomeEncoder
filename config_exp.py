"""
Central configuration for the Robust Microbiome Disease Classifier experiment.

Everything here is microbiome-only (NO leakage/metadata columns).
Paths are relative to this folder so the whole `Clean_Experiments/` directory
can be uploaded to the server and run as-is.
"""
from pathlib import Path
import torch

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_CSV = DATA_DIR / "microbiome_only.csv"          # study_name, disease, <features>
SPLIT_CSV = DATA_DIR / "splits.csv"                  # sample_idx, split, study_id
ARTIFACTS = HERE / "artifacts"                       # scaler/meta saved here
CKPT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"
for d in (ARTIFACTS, CKPT_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True, parents=True)

PRETRAIN_CKPT = CKPT_DIR / "pretrain_microbiome.pt"
FINETUNE_CKPT = CKPT_DIR / "robust_classifier.pt"

# external validation (MetaPhlAn3 reprocessed FASTQ) — 3 diseases
EXTERNAL_DIR = HERE / "external_data"
EXTERNAL_STUDIES = {                # study file stem -> disease label
    "YangJ_2020": "CRC",
    "HeQ_2017": "IBD",
    "XuQ_2021": "T2D",
}

# ---------------------------------------------------------------- split
TARGET = "disease"
GROUP = "study_name"
VAL_FRACTION = 0.15                 # random 15% from remaining training studies
TEST_FRACTION_SINGLE = 0.15        # single-study diseases: 70/15/15
VAL_FRACTION_SINGLE = 0.15
SEED = 42

# ---------------------------------------------------------------- model (UNCHANGED architecture)
HIDDEN_DIM = 256
NUM_CHUNKS = 64
NUM_ENCODER_LAYERS = 4
NUM_ATTENTION_HEADS = 8
FEEDFORWARD_DIM = 1024
DROPOUT = 0.2                       # a bit higher for regularisation
EMBED_DIM = 8

# ---------------------------------------------------------------- techniques
# input transform
USE_CLR = True                     # centered-log-ratio (compositional) else log1p+standardize
# augmentation (train only)
FEATURE_DROPOUT_P = 0.1            # randomly zero this fraction of species per sample
GAUSSIAN_NOISE_STD = 0.05
MIXUP_ALPHA = 0.1                  # 0 disables mixup
# loss
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
# NOTE: we correct class imbalance with the sampler ONLY (below). Turning on
# class weights as well double-corrects and pushes all predictions onto the
# ultra-rare classes (balanced-acc up, overall-acc ~0). Keep this False.
USE_CLASS_WEIGHTS = False
# study-adversarial (gradient reversal) — study-invariance
USE_ADVERSARIAL = True
ADV_LAMBDA = 0.3                   # strength of gradient reversal
ADV_WARMUP_EPOCHS = 5             # ramp lambda 0 -> ADV_LAMBDA over these epochs
ADV_HIDDEN = 128

# ---------------------------------------------------------------- pretraining
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 1e-4
PRETRAIN_WEIGHT_DECAY = 0.01
PRETRAIN_BATCH = 64
MASK_RATIO = 0.3

# ---------------------------------------------------------------- finetuning
FINETUNE_EPOCHS = 30
FINETUNE_LR = 3e-4
FINETUNE_WEIGHT_DECAY = 0.05
FINETUNE_BATCH = 64
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 30          # on val macro-F1 (>= FINETUNE_EPOCHS so all 30 epochs print)
CLASS_BALANCED_SAMPLER = True
# 0.0 = no rebalancing (natural freq), 1.0 = full inverse-freq (equal per class).
# 0.5 = square-root sampling: boosts rare classes but still lets the common,
# well-sampled diseases (CRC/T2D/IBD/adenoma) be learned. This is the sweet spot.
SAMPLER_TEMPERATURE = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed=None):
    import random, numpy as np
    if seed is None:
        seed = SEED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
