"""
Robust model = ORIGINAL encoder-only classifier (unchanged) + external study
adversarial head.

We do NOT modify the architecture. We reuse the exact building blocks
(`ChunkedFeatureEmbedding`, transformer encoder, `AttentionPooling`, classifier)
via the existing `get_finetune_model` / `get_pretrain_model`, feeding a single
dummy categorical. The adversarial head lives OUTSIDE the model and consumes the
pooled representation through a gradient-reversal layer.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# make new_project importable (config + Complete_Architecture)
NEW_PROJECT = Path(__file__).resolve().parents[2]
if str(NEW_PROJECT) not in sys.path:
    sys.path.insert(0, str(NEW_PROJECT))

from Complete_Architecture.model_finetune import get_finetune_model  # noqa: E402
from Complete_Architecture.model_pretrain import get_pretrain_model  # noqa: E402

from src.data import DUMMY_CAT
from src.grl import grad_reverse

DUMMY_VOCAB = {DUMMY_CAT: 1}

# Optional runtime hook used ONLY by ablation experiments (run_ablations.py).
# It is called on the freshly built model and may return a modified model.
# Default None => the model is returned exactly as built (architecture unchanged).
POST_BUILD_HOOK = None


def build_classifier(numerical_features, num_classes, cfg, pretrain_ckpt=None):
    """Original EncoderClassifier, unchanged, with one dummy categorical."""
    model = get_finetune_model(
        categorical_vocab_sizes=DUMMY_VOCAB,
        categorical_features=[DUMMY_CAT],
        numerical_features=list(numerical_features),
        num_classes=num_classes,
        pretrain_checkpoint=str(pretrain_ckpt) if pretrain_ckpt else None,
        hidden_dim=cfg.HIDDEN_DIM,
        num_chunks=cfg.NUM_CHUNKS,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_attention_heads=cfg.NUM_ATTENTION_HEADS,
        feedforward_dim=cfg.FEEDFORWARD_DIM,
        dropout=cfg.DROPOUT,
    )
    if POST_BUILD_HOOK is not None:
        model = POST_BUILD_HOOK(model) or model
    return model


def build_pretrainer(numerical_features, cfg):
    """Original masked-chunk pretraining model, unchanged, with dummy categorical."""
    return get_pretrain_model(
        categorical_vocab_sizes=DUMMY_VOCAB,
        categorical_features=[DUMMY_CAT],
        numerical_features=list(numerical_features),
        hidden_dim=cfg.HIDDEN_DIM,
        num_chunks=cfg.NUM_CHUNKS,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_attention_heads=cfg.NUM_ATTENTION_HEADS,
        feedforward_dim=cfg.FEEDFORWARD_DIM,
        dropout=cfg.DROPOUT,
        mask_ratio=cfg.MASK_RATIO,
    )


def forward_features(model, cat, num):
    """Replicate EncoderClassifier.forward but also return the pooled vector,
    using the model's OWN (unchanged) submodules."""
    chunk = model.chunk_embedding(cat, num)
    x = model.pos_encoding(chunk)
    enc = model.encoder(x)
    enc = model.encoder_norm(enc)
    pooled, attn = model.attention_pooling(enc)
    logits = model.classifier(pooled)
    return logits, pooled, attn


class StudyAdversary(nn.Module):
    """Predicts the training study from the pooled representation, through a
    gradient-reversal layer -> pushes the encoder to be study-invariant."""

    def __init__(self, hidden_dim, n_studies, adv_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, adv_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(adv_hidden, n_studies),
        )

    def forward(self, pooled, lambd):
        return self.net(grad_reverse(pooled, lambd))
