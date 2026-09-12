"""
STEP 2 — Self-supervised PRETRAINING (masked chunk autoencoding) on the
microbiome training data only. No disease labels used. This warms up the
chunk-embedding + encoder so fine-tuning starts from a good representation.

Uses the ORIGINAL MaskedChunkPretraining model unchanged (dummy categorical).
Run:  python pretrain.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src.prepare import load_split_arrays
from src.model_robust import build_pretrainer, DUMMY_CAT


def main():
    C.set_seed()
    dev = torch.device(C.DEVICE)
    print(f"Device: {dev}")

    data = load_split_arrays(fit_transform=True)
    # pretrain on TRAIN + VAL microbiome vectors (unsupervised, no labels)
    X = np.concatenate([data["Xtr"], data["Xva"]], axis=0).astype(np.float32)
    print(f"Pretrain samples: {len(X)}  features: {X.shape[1]}")

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=C.PRETRAIN_BATCH, shuffle=True, drop_last=True)

    model = build_pretrainer(data["meta"]["feature_names"], C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.PRETRAIN_LR,
                            weight_decay=C.PRETRAIN_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.PRETRAIN_EPOCHS)

    best = float("inf")
    for ep in range(1, C.PRETRAIN_EPOCHS + 1):
        model.train()
        tot, nb = 0.0, 0
        for (xb,) in dl:
            xb = xb.to(dev)
            cat = {DUMMY_CAT: torch.zeros(xb.size(0), dtype=torch.long, device=dev)}
            pred, tgt, mask = model(cat, xb)
            loss = model.compute_loss(pred, tgt, mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        avg = tot / max(nb, 1)
        print(f"[pretrain] epoch {ep:3d}/{C.PRETRAIN_EPOCHS}  loss={avg:.5f}")
        if avg < best:
            best = avg
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": ep, "loss": avg}, C.PRETRAIN_CKPT)
    print(f"\nSaved pretrained encoder -> {C.PRETRAIN_CKPT}  (best loss {best:.5f})")


if __name__ == "__main__":
    main()
