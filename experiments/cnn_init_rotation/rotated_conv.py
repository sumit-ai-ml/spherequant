"""Rotated-basis conv and linear layers for H3.

The trainable parameter is U, shape (C_out, fan_in). The effective weight used
in the actual forward pass is W_flat = U @ M.T, where M is a fixed (d, d)
orthogonal matrix set at init. In the conv case W_flat is reshaped to
(C_out, C_in, kH, kW).

At quantization time we quantize U (the "rotated coordinates"), not W. The
effective weight is then reconstructed as W_q = U_q @ M.T.

Sanity: when M is the identity, these modules are numerically equivalent to
nn.Conv2d / nn.Linear with the same initialization. test_sanity.py verifies.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotation_utils import build_torch_rotation


class RotatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 seed: int, stride: int = 1, padding: int = 0, bias: bool = True,
                 rotation_type: str = "srht"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size

        # Kaiming init for U. Since M is orthogonal, U @ M.T has the same
        # per-element distribution, so this matches the baseline init.
        U = torch.empty(out_channels, fan_in)
        nn.init.kaiming_normal_(U, mode="fan_in", nonlinearity="relu")
        self.U = nn.Parameter(U)

        # Fixed orthogonal M. Seed drives which rotation.
        M = build_torch_rotation(fan_in, seed, rotation_type)
        self.register_buffer("M", M)

        if bias:
            b = torch.zeros(out_channels)
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        W_flat = self.U @ self.M.T  # (C_out, fan_in)
        return W_flat.view(self.out_channels, self.in_channels, *self.kernel_size)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias,
                        stride=self.stride, padding=self.padding)


class RotatedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, seed: int,
                 bias: bool = True, rotation_type: str = "srht"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        U = torch.empty(out_features, in_features)
        nn.init.kaiming_normal_(U, mode="fan_in", nonlinearity="relu")
        self.U = nn.Parameter(U)

        M = build_torch_rotation(in_features, seed, rotation_type)
        self.register_buffer("M", M)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        return self.U @ self.M.T  # (out_features, in_features)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class RotatedCNN3(nn.Module):
    """Same architecture as model.CNN3 but with rotated-basis layers.

    Each layer gets a distinct per-layer rotation seed derived from (seed, idx)
    so different layers don't share the same orthogonal matrix.
    """

    def __init__(self, num_classes: int = 10, seed: int = 0,
                 rotation_type: str = "srht"):
        super().__init__()
        self.conv1 = RotatedConv2d(3, 32, 3, seed=seed * 1000 + 1,
                                   padding=1, rotation_type=rotation_type)
        self.conv2 = RotatedConv2d(32, 64, 3, seed=seed * 1000 + 2,
                                   padding=1, rotation_type=rotation_type)
        self.conv3 = RotatedConv2d(64, 128, 3, seed=seed * 1000 + 3,
                                   padding=1, rotation_type=rotation_type)
        self.pool = nn.MaxPool2d(2)
        self.fc = RotatedLinear(128 * 4 * 4, num_classes,
                                seed=seed * 1000 + 4,
                                rotation_type=rotation_type)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        return self.fc(x)

    def rotated_layers(self):
        """Yield (name, module) for layers with U + M. Order matches CNN3."""
        yield "conv1", self.conv1
        yield "conv2", self.conv2
        yield "conv3", self.conv3
        yield "fc", self.fc
