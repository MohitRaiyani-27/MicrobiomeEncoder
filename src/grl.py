"""
Gradient Reversal Layer (GRL) for study-adversarial training.

During the forward pass the input is passed through unchanged. During the
backward pass the gradient is multiplied by -lambda. A study classifier is
trained on top of the pooled representation to predict which STUDY a sample came
from; because of the reversal, the encoder is pushed to make its representation
study-INVARIANT — removing batch/cohort fingerprints and keeping only signal that
generalises to new studies.
"""
import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)
