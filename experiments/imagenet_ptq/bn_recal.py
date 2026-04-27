"""BatchNorm re-calibration after quantization.

Standard PTQ hygiene: after quantizing weights, the pre-computed BN running
stats (mean, var) are slightly stale because the weight-reconstruction error
shifts the pre-activation distribution. Running a few hundred calibration
images through the quantized model in train() mode re-accumulates the stats
and typically recovers 1-3 pp of accuracy on CNNs with BN (negligible on
ViT which uses LayerNorm, not BN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _has_batchnorm(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            return True
    return False


def _reset_bn_stats(model: nn.Module):
    """Reset running_mean/var to initial state so calibration re-accumulates."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.reset_running_stats()
            # momentum=None switches BN to cumulative averaging (better for
            # calibration from a small number of batches)
            m.momentum = None


@torch.no_grad()
def bn_recalibrate(model: nn.Module, calibration_loader: DataLoader,
                   device: str, n_batches: int = 16) -> nn.Module:
    """Re-accumulate BN stats by running calibration images through in train mode.

    Returns the same model (modified in place). Safe no-op if no BN layers.
    """
    if not _has_batchnorm(model):
        return model

    model.train()  # so BN updates running stats
    _reset_bn_stats(model)

    seen = 0
    for i, batch in enumerate(calibration_loader):
        if i >= n_batches:
            break
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device, non_blocking=True)
        model(x)
        seen += x.size(0)

    model.eval()
    return model
