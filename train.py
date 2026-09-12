"""
STEP 3 — Fine-tune the encoder classifier on all 30 diseases (microbiome only),
with the techniques we agreed on:

  * loads self-supervised pretrained encoder (if available)
  * CLR-transformed input
  * class-balanced sampling + focal loss + class weights + label smoothing
  * augmentation: feature dropout + gaussian noise + mixup
  * study-adversarial head (gradient reversal) for cross-study invariance
  * cosine LR with warmup, early stopping on VAL macro-F1

Run:  python train.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_exp as C
from src.prepare import load_split_arrays, class_weights, sample_weights
from src.data import MicrobiomeDataset, collate, to_device, DUMMY_CAT
from src.model_robust import build_classifier, StudyAdversary, forward_features
from src.losses import FocalLoss
from src.augment import feature_dropout, gaussian_noise


def lr_factor(epoch):
    if epoch < C.WARMUP_EPOCHS:
        return (epoch + 1) / C.WARMUP_EPOCHS
    prog = (epoch - C.WARMUP_EPOCHS) / max(1, C.FINETUNE_EPOCHS - C.WARMUP_EPOCHS)
    return 0.5 * (1 + np.cos(np.pi * prog))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    preds, ys = [], []
    for cat, num, y, sid in loader:
        cat, num = to_device(cat, num, dev)
        logits, _, _ = forward_features(model, cat, num)
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    return {
        "acc": accuracy_score(ys, preds),
        "bal_acc": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro"),
    }


def main():
    C.set_seed()
    dev = torch.device(C.DEVICE)
    print(f"Device: {dev}")

    data = load_split_arrays(fit_transform=True)
    meta = data["meta"]
    nC = meta["n_classes"]; nS = meta["n_train_studies"]
    print(f"classes={nC}  train_studies={nS}  features={meta['n_features']}")

    tr = MicrobiomeDataset(data["Xtr"], data["ytr"], data["sid_tr"])
    va = MicrobiomeDataset(data["Xva"], data["yva"], np.full_like(data["yva"], -1))

    if C.CLASS_BALANCED_SAMPLER:
        sw = sample_weights(data["ytr"], nC, temperature=C.SAMPLER_TEMPERATURE)
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        tr_loader = DataLoader(tr, batch_size=C.FINETUNE_BATCH, sampler=sampler,
                               collate_fn=collate, drop_last=True)
    else:
        tr_loader = DataLoader(tr, batch_size=C.FINETUNE_BATCH, shuffle=True,
                               collate_fn=collate, drop_last=True)
    va_loader = DataLoader(va, batch_size=256, shuffle=False, collate_fn=collate)

    pretrain = C.PRETRAIN_CKPT if Path(C.PRETRAIN_CKPT).exists() else None
    if pretrain:
        print(f"Loading pretrained encoder: {pretrain}")
    model = build_classifier(meta["feature_names"], nC, C, pretrain_ckpt=pretrain).to(dev)
    adversary = StudyAdversary(C.HIDDEN_DIM, nS, C.ADV_HIDDEN).to(dev) if C.USE_ADVERSARIAL else None

    cw = torch.tensor(class_weights(data["ytr"], nC)) if C.USE_CLASS_WEIGHTS else None
    if cw is not None: cw = cw.to(dev)
    focal = FocalLoss(gamma=C.FOCAL_GAMMA, weight=cw, label_smoothing=C.LABEL_SMOOTHING).to(dev)

    params = list(model.parameters()) + (list(adversary.parameters()) if adversary else [])
    opt = torch.optim.AdamW(params, lr=C.FINETUNE_LR, weight_decay=C.FINETUNE_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_factor)

    best_f1, best_state, patience, history = -1.0, None, 0, []
    for ep in range(C.FINETUNE_EPOCHS):
        model.train()
        if adversary: adversary.train()
        lambd = C.ADV_LAMBDA * min(1.0, (ep + 1) / max(1, C.ADV_WARMUP_EPOCHS)) if C.USE_ADVERSARIAL else 0.0
        run_cls, run_adv, nb = 0.0, 0.0, 0

        for cat, num, y, sid in tr_loader:
            cat, num = to_device(cat, num, dev)
            y = y.to(dev); sid = sid.to(dev)

            # augmentation (train only)
            num = feature_dropout(num, C.FEATURE_DROPOUT_P)
            num = gaussian_noise(num, C.GAUSSIAN_NOISE_STD)

            # mixup
            if C.MIXUP_ALPHA > 0:
                lam = float(np.random.beta(C.MIXUP_ALPHA, C.MIXUP_ALPHA))
                perm = torch.randperm(num.size(0), device=dev)
                num = lam * num + (1 - lam) * num[perm]
                y_a, y_b = y, y[perm]
                sid_a, sid_b = sid, sid[perm]
            else:
                lam, y_a, y_b, sid_a, sid_b = 1.0, y, y, sid, sid

            logits, pooled, _ = forward_features(model, cat, num)
            cls_loss = lam * focal(logits, y_a) + (1 - lam) * focal(logits, y_b)

            if adversary is not None:
                adv_logits = adversary(pooled, lambd)

                def _safe_ce(target):
                    # only samples from multi-study diseases have sid>=0
                    if (target >= 0).any():
                        return F.cross_entropy(adv_logits, target, ignore_index=-1)
                    return torch.zeros((), device=dev)
                adv_loss = lam * _safe_ce(sid_a) + (1 - lam) * _safe_ce(sid_b)
            else:
                adv_loss = torch.zeros((), device=dev)

            loss = cls_loss + adv_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run_cls += cls_loss.item(); run_adv += float(adv_loss.detach()); nb += 1

        sched.step()
        val = evaluate(model, va_loader, dev)
        history.append({"epoch": ep + 1, "cls_loss": run_cls / nb,
                        "adv_loss": run_adv / nb, "lambda": lambd, **val})
        print(f"[ft] ep {ep+1:3d}/{C.FINETUNE_EPOCHS}  cls={run_cls/nb:.3f} adv={run_adv/nb:.3f} "
              f"lam={lambd:.2f} | val acc={val['acc']:.3f} bal={val['bal_acc']:.3f} "
              f"macroF1={val['macro_f1']:.3f}")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {ep+1} (best val macroF1={best_f1:.3f})")
                break

    torch.save({"model_state_dict": best_state, "classes": meta["classes"],
                "feature_names": meta["feature_names"], "best_val_macro_f1": best_f1},
               C.FINETUNE_CKPT)
    with open(C.RESULTS_DIR / "finetune_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved best model -> {C.FINETUNE_CKPT}  (val macroF1={best_f1:.3f})")


if __name__ == "__main__":
    main()
