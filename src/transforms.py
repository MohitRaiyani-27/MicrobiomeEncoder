"""
Compositional input transform for microbiome relative abundances.

Microbiome abundances are COMPOSITIONAL (each sample sums to a constant).
Feeding raw percentages to a model is statistically wrong and lets scale/library
effects leak in. We use the Centered Log-Ratio (CLR) transform, the standard
choice for compositional data:

    clr(x)_i = log(x_i + eps) - mean_j log(x_j + eps)

CLR is invariant to per-sample scaling (percent vs fraction vs read depth), which
already removes a big chunk of study/batch effects. We then standardise each
feature using TRAIN statistics only.

The transform is FIT on training data and reused unchanged for val/test/external.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class CLRTransform:
    def __init__(self, use_clr: bool = True, eps: float | None = None):
        self.use_clr = use_clr
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self.features_ = None

    def fit(self, X: np.ndarray, feature_names: list[str]):
        X = np.asarray(X, dtype=np.float64)
        self.features_ = list(feature_names)
        if self.eps is None:
            nz = X[X > 0]
            self.eps = float(nz.min() / 2.0) if nz.size else 1e-6
        Z = self._core(X)
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-8
        return self

    def _core(self, X: np.ndarray) -> np.ndarray:
        if self.use_clr:
            L = np.log(X + self.eps)
            return L - L.mean(axis=1, keepdims=True)
        return np.log1p(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Z = self._core(X)
        Z = (Z - self.mean_) / self.std_
        return Z.astype(np.float32)

    def fit_transform(self, X, feature_names):
        return self.fit(X, feature_names).transform(X)

    # ---- persistence ----
    def save(self, path: str | Path):
        path = Path(path)
        np.savez(path, mean=self.mean_, std=self.std_,
                 eps=np.array([self.eps]), use_clr=np.array([int(self.use_clr)]))
        with open(path.with_suffix(".features.json"), "w") as f:
            json.dump(self.features_, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        d = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
        obj = cls(use_clr=bool(d["use_clr"][0]), eps=float(d["eps"][0]))
        obj.mean_ = d["mean"]; obj.std_ = d["std"]
        fpath = Path(str(path).replace(".npz", "")).with_suffix(".features.json")
        if fpath.exists():
            obj.features_ = json.load(open(fpath))
        return obj
