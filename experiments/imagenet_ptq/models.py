"""Wrappers around torchvision pretrained models for the PTQ sweep."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def load_pretrained(name: str) -> nn.Module:
    """Load an ImageNet-pretrained model by short name."""
    name = name.lower()
    if name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if name == "mobilenet_v2":
        return models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if name == "alexnet":
        return models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    if name == "vgg16":
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if name == "vit_b_16":
        return models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    if name == "vit_s_16" or name == "vit_b_32":
        # torchvision has no vit_s_16 but vit_b_32 is the smaller practical option
        return models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
    if name == "efficientnet_b0":
        return models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if name == "convnext_tiny":
        return models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    raise ValueError(f"unknown model: {name}")


def count_weights_bytes(model: nn.Module) -> tuple[int, int, int]:
    """Returns (n_quantizable_weights, n_bias_params, total_rows).

    'Quantizable' means Conv2d and Linear weights. BN, biases, etc. stay FP32.
    'Rows' is the number of output-channel rows summed across quantizable layers
    (we store one FP32 norm per row when per-row L2 normalizing).
    """
    n_w, n_b, n_rows = 0, 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            n_w += m.weight.numel()
            if m.bias is not None:
                n_b += m.bias.numel()
            n_rows += m.weight.shape[0]
    return n_w, n_b, n_rows


def model_size_at_bits(model: nn.Module, bits: int | None) -> dict:
    """Return dict with fp32_mb, quantized_mb, ratio.

    bits=None returns FP32 size only.
    """
    n_w, n_b, n_rows = count_weights_bytes(model)
    # Also count all other parameters (BN weights, biases, etc.) — these stay FP32
    total_params = sum(p.numel() for p in model.parameters())
    other_params = total_params - n_w  # everything except conv/linear weights
    fp32_bytes = total_params * 4
    result = {
        "fp32_mb": fp32_bytes / (1024 ** 2),
        "n_weights_quant": n_w,
        "n_params_total": total_params,
        "n_rows": n_rows,
    }
    if bits is not None:
        code_bytes = n_w * bits / 8
        norm_bytes = n_rows * 4  # per-row FP32 norm storage
        other_bytes = other_params * 4  # BN, biases, etc. FP32
        quant_bytes = code_bytes + norm_bytes + other_bytes
        result["quantized_mb"] = quant_bytes / (1024 ** 2)
        result["ratio"] = fp32_bytes / quant_bytes
    return result
