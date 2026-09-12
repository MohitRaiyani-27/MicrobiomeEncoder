"""
INTEGRATED-GRADIENTS INTERPRETATION.

Replaces the "attention-only" explanation (which the reviewers, and our own
pooling ablation, showed is not a reliable importance signal) with Integrated
Gradients (IG) -- a standard, axiomatic attribution method for deep models.

For each disease we take the correctly-predicted test patients and attribute the
model's logit for that disease back onto the input microbiome features. IG
integrates the gradient along a straight path from a neutral baseline (all-zero
in CLR space) to the real sample:

    attribution_i = (x_i - baseline_i) * mean_alpha[ d logit / d x_i  at
                                                     baseline + alpha*(x-baseline) ]

Averaging attributions across a disease's patients gives a ranked list of the
species that most drive that disease's prediction. Positive = pushes toward the
disease. These can be checked against the microbiome literature.

Run (CPU is fine):  python run_interpret.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src.prepare import load_split_arrays
from src.model_robust import build_classifier, forward_features
from src.data import DUMMY_CAT

IG_STEPS = 32           # integration steps (Riemann sum)
MIN_CORRECT = 5         # only interpret diseases with >= this many correct patients
TOP_K = 15              # top features to keep per disease
PLOT_DISEASES = ["CRC", "IBD", "T2D"]


def load_model(meta, dev):
    ck = torch.load(C.FINETUNE_CKPT, map_location=dev)
    model = build_classifier(meta["feature_names"], meta["n_classes"], C).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck["classes"]


def predict(model, X, dev):
    """Return predicted class ids for every row of X."""
    with torch.no_grad():
        num = torch.tensor(X, dtype=torch.float32, device=dev)
        cat = {DUMMY_CAT: torch.zeros(len(X), dtype=torch.long, device=dev)}
        logits, _, _ = forward_features(model, cat, num)
    return logits.argmax(1).cpu().numpy()


def integrated_gradients(model, X, di, dev, steps=IG_STEPS):
    """IG attributions for target class `di`, averaged over the rows of X.
    Returns a [n_features] numpy array."""
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    baseline = torch.zeros_like(x)
    cat = {DUMMY_CAT: torch.zeros(len(X), dtype=torch.long, device=dev)}

    total_grad = torch.zeros_like(x)
    for s in range(steps):
        alpha = (s + 0.5) / steps                      # midpoint rule
        scaled = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits, _, _ = forward_features(model, cat, scaled)
        target = logits[:, di].sum()
        grad = torch.autograd.grad(target, scaled)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / steps
    attributions = (x - baseline) * avg_grad           # [n, F]
    return attributions.mean(0).cpu().numpy()          # [F]


def main():
    dev = torch.device(C.DEVICE)
    data = load_split_arrays(fit_transform=False)
    meta = data["meta"]
    feat = np.array(meta["feature_names"])
    model, classes = load_model(meta, dev)
    cls2id = {c: i for i, c in enumerate(classes)}

    Xte, yte = data["Xte"], data["yte"]
    preds = predict(model, Xte, dev)

    results = {}
    print("=" * 66)
    print("INTEGRATED-GRADIENTS TOP SPECIES PER DISEASE (correct test patients)")
    print("=" * 66)
    for disease, di in cls2id.items():
        mask = (yte == di) & (preds == di)             # correctly predicted only
        n = int(mask.sum())
        if n < MIN_CORRECT:
            continue
        attr = integrated_gradients(model, Xte[mask], di, dev)
        order = np.argsort(-attr)                       # most positive first
        top = [{"feature": str(feat[j]), "attribution": float(attr[j])}
               for j in order[:TOP_K]]
        results[disease] = {"n_correct": n, "top_features": top}

        print(f"\n{disease}  (n_correct={n})")
        for t in top[:8]:
            print(f"    {t['attribution']:+.4f}  {t['feature']}")

    dst = C.RESULTS_DIR / "interpretation.json"
    with open(dst, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {dst}")

    # ---- optional bar charts for the headline diseases ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = C.RESULTS_DIR / "figures"
        fig_dir.mkdir(exist_ok=True)
        for d in PLOT_DISEASES:
            if d not in results:
                continue
            top = results[d]["top_features"][:12][::-1]
            names = [t["feature"].split("|")[-1][:40] for t in top]
            vals = [t["attribution"] for t in top]
            plt.figure(figsize=(8, 5))
            plt.barh(names, vals, color="#3b7dd8")
            plt.xlabel("Integrated-gradients attribution")
            plt.title(f"Top species driving {d} prediction")
            plt.tight_layout()
            out = fig_dir / f"ig_{d}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"  plot -> {out}")
    except Exception as e:
        print(f"  [plots skipped: {e}]")


if __name__ == "__main__":
    main()
