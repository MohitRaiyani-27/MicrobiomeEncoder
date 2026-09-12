"""
Regenerate the paper figures on the CLEAN (leakage-free) 30-disease pipeline.

Produces (into this folder):
  fig1_chunked_embedding.png   schematic: 1,207 species -> 64 chunks -> tokens
  fig4_pretrain_loss.png       masked-chunk pre-training MSE loss (40 epochs)
  fig5_finetune_head.png       schematic: fine-tuning head (30 classes)
  fig6_integrated_gradients.png top species by Integrated Gradients (CRC/RA/IBD)
  fig7_training_curves.png     fine-tuning train loss + validation acc/macro-F1
  fig8_confusion_matrix.png    row-normalised confusion matrix (internal test)

Run (Mac -> force CPU, MPS crashes this model):
  ../venv/bin/python make_figures.py         # server
  <repo>/.venv/bin/python make_figures.py    # mac (CPU forced automatically)
"""
import re
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # Clean_Experiments
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BLUE = "#2b6cb0"
GREEN = "#2f855a"
ORANGE = "#c05621"
GREY = "#4a5568"
LIGHT = "#ebf4ff"


# ----------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color="#1a202c")


def _arrow(ax, x1, y1, x2, y2, text=None, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, text, ha="center",
                va="bottom", fontsize=8.5, color=color, style="italic")


# ----------------------------------------------------------------------------
def fig1_chunked_embedding():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    # input feature vector
    _box(ax, 0.2, 1.4, 1.7, 2.2,
         "1,207\nmicrobiome\nspecies\n(CLR)", fc="#f0fff4", ec=GREEN,
         fs=10, weight="bold")

    _arrow(ax, 1.95, 2.5, 3.1, 2.5, "partition into\n64 chunks")

    # chunk stack
    cx = 3.2
    ys = [3.5, 2.9, 2.3, 1.1]
    labels = ["chunk 1\n(~19 sp.)", "chunk 2", "chunk 3", "chunk 64"]
    for y, lab in zip(ys, labels):
        _box(ax, cx, y, 1.5, 0.5, lab, fc=LIGHT, ec=BLUE, fs=8)
    ax.text(cx + 0.75, 1.75, "$\\vdots$", ha="center", va="center", fontsize=16)

    _arrow(ax, 4.8, 2.5, 6.0, 2.5, "Linear -> GELU\n-> LayerNorm")

    # token stack
    tx = 6.1
    for y in ys:
        _box(ax, tx, y, 1.4, 0.5, "token\n256-dim", fc="#fffaf0", ec=ORANGE, fs=8)
    ax.text(tx + 0.7, 1.75, "$\\vdots$", ha="center", va="center", fontsize=16)

    _arrow(ax, 7.55, 2.5, 8.7, 2.5, "64 tokens\nx 256")

    _box(ax, 8.8, 1.4, 2.9, 2.2,
         "Transformer\nEncoder\n(4 layers, 8 heads)", fc="#ebf8ff", ec=BLUE,
         fs=11, weight="bold")

    ax.set_title("Figure 1. Chunked feature embedding: 1,207 species are split into "
                 "64 chunks (~19 features each),\neach projected to a 256-dim token.",
                 fontsize=11, loc="center")
    fig.savefig(HERE / "fig1_chunked_embedding.png")
    plt.close(fig)
    print("saved fig1_chunked_embedding.png")


# ----------------------------------------------------------------------------
def fig4_pretrain_loss():
    log = RESULTS / "run_main.log"
    ep, loss = [], []
    pat = re.compile(r"\[pretrain\]\s+epoch\s+(\d+)/\d+\s+loss=([0-9.]+)")
    for line in log.read_text().splitlines():
        m = pat.search(line)
        if m:
            ep.append(int(m.group(1)))
            loss.append(float(m.group(2)))
    ep, loss = np.array(ep), np.array(loss)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ep, loss, "-o", color=BLUE, ms=4, lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Pre-training epoch")
    ax.set_ylabel("Masked-chunk MSE loss (log scale)")
    ax.set_title("Figure 4. Self-supervised pre-training loss")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.annotate(f"epoch 1: {loss[0]:.1f}", (ep[0], loss[0]),
                textcoords="offset points", xytext=(20, -4), fontsize=9)
    ax.annotate(f"epoch {ep[-1]}: {loss[-1]:.2f}", (ep[-1], loss[-1]),
                textcoords="offset points", xytext=(-70, 12), fontsize=9)
    fig.savefig(HERE / "fig4_pretrain_loss.png")
    plt.close(fig)
    print(f"saved fig4_pretrain_loss.png ({len(ep)} epochs)")


# ----------------------------------------------------------------------------
def fig5_finetune_head():
    from matplotlib.patches import Rectangle

    ENC = "#6b7280"      # encoder (grey)
    IO = "#1f2937"       # input/output (dark)
    RM = "#d1d5db"       # removed (light grey)
    NEW = "#2b6cb0"      # new trainable (blue)

    fig, ax = plt.subplots(figsize=(14, 5.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

    w, h, y = 1.5, 1.5, 1.9
    sp = 1.74
    x0 = 0.15
    xs = [x0 + i * sp for i in range(8)]

    boxes = [
        ("Input\n1,207\nfeatures", IO, "white"),
        ("Linear\nProjection\n-> 256-dim", ENC, "white"),
        ("Positional\nEncoding", ENC, "white"),
        ("Transformer\nEncoder\n4L x 8H", ENC, "white"),
        ("Prediction\nHead\n(MSE)", RM, "#4a5568"),
        ("Attention\nPooling\n-> 256-dim", NEW, "white"),
        ("MLP\nClassifier\n256 -> 30", NEW, "white"),
        ("30 Disease\nClasses", IO, "white"),
    ]

    for i, (txt, fc, tc) in enumerate(boxes):
        ax.add_patch(Rectangle((xs[i], y), w, h, fc=fc, ec="#2d3748", lw=1.3))
        ax.text(xs[i] + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9.5, color=tc, weight="bold")
        if i < 7:
            ax.add_patch(FancyArrowPatch((xs[i] + w, y + h / 2),
                                         (xs[i + 1], y + h / 2),
                                         arrowstyle="-|>", mutation_scale=12,
                                         lw=1.3, color="#4a5568"))

    # cross out the removed prediction head
    bx = xs[4]
    ax.plot([bx, bx + w], [y, y + h], color="#c53030", lw=2.2)
    ax.plot([bx, bx + w], [y + h, y], color="#c53030", lw=2.2)
    ax.text(bx + w / 2, y - 0.28, "REMOVED", ha="center", va="top",
            fontsize=9, weight="bold", color="#c53030")

    # top labels
    ax.text((xs[2] + w / 2 + xs[3] + w / 2) / 2, y + h + 0.55, "ENCODER",
            ha="center", fontsize=9, weight="bold", color=ENC)
    for i in (5, 6):
        ax.text(xs[i] + w / 2, y + h + 0.18, "NEW", ha="center",
                fontsize=8.5, weight="bold", color=NEW)

    # bracket over new head
    bxl, bxr = xs[5], xs[6] + w
    ax.plot([bxl, bxr], [y + h + 0.75, y + h + 0.75], color=NEW, lw=1.4)
    ax.plot([bxl, bxl], [y + h + 0.62, y + h + 0.75], color=NEW, lw=1.4)
    ax.plot([bxr, bxr], [y + h + 0.62, y + h + 0.75], color=NEW, lw=1.4)
    ax.text((bxl + bxr) / 2, y + h + 0.85, "Fine-tuning head (trainable)",
            ha="center", va="bottom", fontsize=9.5, weight="bold", color=NEW)

    # bracket under encoder
    exl, exr = xs[1], xs[3] + w
    ax.plot([exl, exr], [y - 0.55, y - 0.55], color=ENC, lw=1.4)
    ax.plot([exl, exl], [y - 0.42, y - 0.55], color=ENC, lw=1.4)
    ax.plot([exr, exr], [y - 0.42, y - 0.55], color=ENC, lw=1.4)
    ax.text((exl + exr) / 2, y - 0.68, "Encoder trained during fine-tuning "
            "(pre-training optional \u2013 see ablation)",
            ha="center", va="top", fontsize=9, color=ENC)

    # legend
    leg_y = 0.35
    items = [("Encoder", ENC), ("Removed (prediction head)", RM),
             ("New components (trainable)", NEW)]
    lx = 2.6
    for lab, c in items:
        ax.add_patch(Rectangle((lx, leg_y), 0.35, 0.3, fc=c, ec="#2d3748", lw=1))
        ax.text(lx + 0.5, leg_y + 0.15, lab, ha="left", va="center", fontsize=9)
        lx += 0.7 + len(lab) * 0.11

    ax.set_title("Figure 5. Stage 2 fine-tuning: the pre-training prediction head "
                 "is removed and replaced\nby attention pooling and a 2-layer MLP "
                 "classifier over 30 diseases.", fontsize=11, pad=16)
    fig.savefig(HERE / "fig5_finetune_head.png")
    plt.close(fig)
    print("saved fig5_finetune_head.png")


# ----------------------------------------------------------------------------
def _clean_sp(name):
    return re.sub(r"^(species|genus):", "", str(name)).replace("[", "").replace("]", "")


def fig6_integrated_gradients():
    data = json.load(open(RESULTS / "interpretation.json"))
    diseases = [d for d in ["CRC", "RA", "IBD"] if d in data]
    colors = {"CRC": BLUE, "RA": GREEN, "IBD": ORANGE}

    # 2 rows x 2 cols: CRC + RA on top row, IBD bottom-left, note bottom-right
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    slots = {"CRC": axes[0, 0], "RA": axes[0, 1], "IBD": axes[1, 0]}
    note_ax = axes[1, 1]

    for dis in diseases:
        ax = slots[dis]
        feats = data[dis]["top_features"][:10]
        names = [_clean_sp(f["feature"]) for f in feats][::-1]
        vals = [f["attribution"] for f in feats][::-1]
        ax.barh(range(len(vals)), vals, color=colors.get(dis, BLUE), alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=9, style="italic")
        ax.set_xlabel("mean attribution")
        ax.set_title(f"{dis}   (n = {data[dis]['n_correct']})")
        ax.grid(True, axis="x", ls=":", alpha=0.4)

    # bottom-right corner left blank (the "how to read" text now lives in the caption)
    note_ax.axis("off")

    fig.suptitle("Figure 6. Top species by Integrated Gradients "
                 "(higher = pushes prediction toward the disease)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "fig6_integrated_gradients.png")
    plt.close(fig)
    print(f"saved fig6_integrated_gradients.png ({diseases})")


# ----------------------------------------------------------------------------
def fig7_training_curves():
    hist = json.load(open(RESULTS / "finetune_history.json"))
    ep = [h["epoch"] for h in hist]
    loss = [h["cls_loss"] for h in hist]
    acc = [h["acc"] for h in hist]
    mf1 = [h["macro_f1"] for h in hist]
    best = int(np.argmax(mf1))

    # stacked panels avoid twin-axis text overlap
    fig, (axL, axV) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)

    axL.plot(ep, loss, "-o", color=GREY, ms=3, lw=1.8, label="training loss")
    axL.set_ylabel("Training loss")
    axL.axvline(ep[best], color="#a0aec0", ls="--", lw=1)
    axL.grid(True, ls=":", alpha=0.4)
    axL.legend(loc="upper right", fontsize=9)
    axL.set_title("Figure 7. Fine-tuning: training loss (top) and "
                  "validation accuracy / macro-F1 (bottom)")

    axV.plot(ep, acc, "-s", color=BLUE, ms=3, lw=1.8, label="validation accuracy")
    axV.plot(ep, mf1, "-^", color=GREEN, ms=3, lw=1.8, label="validation macro-F1")
    axV.set_xlabel("Fine-tuning epoch")
    axV.set_ylabel("Validation metric")
    axV.set_ylim(0, max(max(acc), max(mf1)) * 1.32)
    axV.axvline(ep[best], color="#a0aec0", ls="--", lw=1)
    axV.grid(True, ls=":", alpha=0.4)
    axV.legend(loc="lower right", fontsize=9)
    ytop = max(max(acc), max(mf1)) * 1.24
    axV.annotate(f"best epoch {ep[best]} (val macro-F1 = {mf1[best]:.2f})",
                 (ep[best], mf1[best]), xytext=(ep[best] - 1.0, ytop),
                 ha="right", fontsize=9, color=GREEN,
                 arrowprops=dict(arrowstyle="->", color="#a0aec0"))

    fig.tight_layout()
    fig.savefig(HERE / "fig7_training_curves.png")
    plt.close(fig)
    print("saved fig7_training_curves.png")


# ----------------------------------------------------------------------------
def fig8_confusion_matrix():
    # force CPU on mac (MPS crashes this model)
    import torch
    import config_exp as C
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() \
            and not torch.cuda.is_available():
        C.DEVICE = "cpu"
    from src.prepare import load_split_arrays
    from src.data import MicrobiomeDataset, collate, to_device
    from src.model_robust import build_classifier, forward_features
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix

    dev = torch.device(C.DEVICE)
    data = load_split_arrays(fit_transform=False)
    meta = data["meta"]
    ck = torch.load(C.FINETUNE_CKPT, map_location=dev)
    model = build_classifier(meta["feature_names"], meta["n_classes"], C).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    classes = ck["classes"]

    Xte, yte = data["Xte"], data["yte"]
    ds = MicrobiomeDataset(Xte, yte, np.full_like(yte, -1))
    dl = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for cat, num, yy, sid in dl:
            cat, num = to_device(cat, num, dev)
            logits, _, _ = forward_features(model, cat, num)
            preds.append(logits.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

    # keep classes present in test, ordered by support (desc)
    present, counts = np.unique(yte, return_counts=True)
    order = present[np.argsort(-counts)]
    labels = [classes[i] for i in order]

    cm = confusion_matrix(yte, preds, labels=order)
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    n = len(order)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.42), max(7, n * 0.42)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Figure 8. Row-normalised confusion matrix, internal test "
                 f"({n} classes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall")
    fig.savefig(HERE / "fig8_confusion_matrix.png")
    plt.close(fig)
    print(f"saved fig8_confusion_matrix.png ({n} classes)")


# ----------------------------------------------------------------------------
def fig9_per_class_auc():
    data = json.load(open(RESULTS / "per_class_auc.json"))
    macro = data["summary"]["macro_auc_reliable"]

    # keep only the reliable classes (n_pos >= 5), sort by AUC ascending
    # (ascending so the highest bar ends up at the top of a horizontal chart)
    items = [(name, d["auc"], d["n_pos"])
             for name, d in data["per_class"].items() if d.get("reliable")]
    items.sort(key=lambda t: t[1])
    names = [f"{n}  (n={p})" for n, _, p in items]
    vals = [a for _, a, _ in items]

    fig, ax = plt.subplots(figsize=(9, max(6, len(items) * 0.42)))
    ypos = range(len(vals))
    ax.barh(ypos, vals, color=BLUE, alpha=0.85)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(names, fontsize=9)
    lo = min(0.45, min(vals) - 0.03)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("one-vs-rest AUC")
    ax.axvline(0.5, color="#a0aec0", ls="--", lw=1)          # chance
    ax.axvline(macro, color=ORANGE, ls="--", lw=1.4,
               label=f"macro-AUC = {macro:.2f}")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    # value labels at the end of each bar
    for y, v in zip(ypos, vals):
        ax.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=8, color="#1a202c")

    ax.set_title(f"Per-class one-vs-rest AUC "
                 f"({len(items)} reliable classes, n_pos >= 5)")
    fig.tight_layout()
    fig.savefig(HERE / "fig9_per_class_auc.png")
    plt.close(fig)
    print(f"saved fig9_per_class_auc.png ({len(items)} reliable classes, "
          f"macro-AUC = {macro:.3f})")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fig1_chunked_embedding()
    fig4_pretrain_loss()
    fig5_finetune_head()
    fig6_integrated_gradients()
    fig7_training_curves()
    fig8_confusion_matrix()
    fig9_per_class_auc()
    print("\nAll figures written to:", HERE)
