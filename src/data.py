"""
Dataset + collate for the microbiome classifier.

Features are microbiome-only and NUMERICAL. The original encoder expects at least
one categorical feature (it reads the batch size from it), so we feed a single
constant dummy categorical `_bias` (always index 0). This is a no-op learned bias
token and keeps the ORIGINAL model architecture untouched.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

DUMMY_CAT = "_bias"


class MicrobiomeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, study_id: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.int64))
        self.sid = torch.as_tensor(np.asarray(study_id, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.sid[i]


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    sids = torch.stack([b[2] for b in batch])
    cat = {DUMMY_CAT: torch.zeros(xs.size(0), dtype=torch.long)}
    return cat, xs, ys, sids


def to_device(cat, num, device):
    cat = {k: v.to(device) for k, v in cat.items()}
    return cat, num.to(device)
