"""Tests for apexquant.audit.

Real torchvision models are loaded with weights=None (no download, no
training); the audit is structural and does not need pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from apexquant import BAD, EMPTY, GOOD, MARGINAL, audit


# --- Architecture-level verdicts: validated against the paper's results -----

def test_resnet50_verdict_good():
    model = tvm.resnet50(weights=None)
    report = audit(model, verbose=False)
    assert report.overall_verdict == GOOD
    assert report.bad_layer_fraction == 0.0
    assert report.params_good / report.quantizable_params > 0.99


def test_mobilenet_v2_verdict_bad():
    model = tvm.mobilenet_v2(weights=None)
    report = audit(model, verbose=False)
    assert report.overall_verdict == BAD
    # MobileNet-V2 has 17 depthwise 3x3 convolutions.
    assert report.n_layers_depthwise >= 15
    assert report.bad_layer_fraction > 0.30


def test_vit_b_16_verdict_good():
    model = tvm.vit_b_16(weights=None)
    report = audit(model, verbose=False)
    assert report.overall_verdict == GOOD
    # Every layer in ViT is large-d (patch projection, attention, MLP).
    assert report.n_layers_bad == 0
    assert report.params_good == report.quantizable_params


def test_efficientnet_b0_verdict_bad():
    model = tvm.efficientnet_b0(weights=None)
    report = audit(model, verbose=False)
    assert report.overall_verdict == BAD
    # EfficientNet-B0 has 16 depthwise convolutions (some 3x3, some 5x5).
    assert report.n_layers_depthwise >= 14


# --- Structural unit tests: layer-type detection ---------------------------

def test_depthwise_detection():
    model = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
    )
    report = audit(model, verbose=False)
    bad_layers = [l for l in report.layers if l.verdict == BAD]
    assert len(bad_layers) == 1
    assert bad_layers[0].fan_in == 9
    assert bad_layers[0].note == "depthwise"


def test_mha_in_proj_detected():
    model = nn.Sequential(
        nn.MultiheadAttention(embed_dim=512, num_heads=8),
    )
    report = audit(model, verbose=False)
    mha_layers = [l for l in report.layers if l.note == "MHA in_proj"]
    assert len(mha_layers) == 1
    assert mha_layers[0].fan_in == 512


def test_skip_norms_and_embeddings():
    model = nn.Sequential(
        nn.LayerNorm(768),
        nn.BatchNorm2d(64),
        nn.Embedding(1000, 256),
    )
    report = audit(model, verbose=False)
    assert len(report.layers) == 0
    assert report.quantizable_params == 0
    # And the empty-model verdict is EMPTY, not MARGINAL by accident.
    assert report.overall_verdict == EMPTY


def test_grouped_conv_fan_in():
    # 64 in / 4 groups -> 16 channels per group; kernel 3x3 -> 9 spatial.
    # d = 16 * 9 = 144. Should be GOOD (>= LARGE_D = 100), not depthwise.
    model = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4),
    )
    report = audit(model, verbose=False)
    assert len(report.layers) == 1
    layer = report.layers[0]
    assert layer.fan_in == 144  # (64 / 4) * 9
    assert "grouped" in layer.note
    assert layer.verdict == GOOD


# --- Integration: quantize_model preflight behavior -------------------------

def test_quantize_model_raises_on_bad_verdict():
    from apexquant import ApexQuantPreflightWarning, quantize_model
    model = tvm.mobilenet_v2(weights=None)
    try:
        quantize_model(model, bits=4)
    except ApexQuantPreflightWarning as e:
        assert e.report is not None
        assert e.report.overall_verdict == BAD
        # Message must mention the actual depthwise count.
        assert "depthwise" in str(e)
    else:
        raise AssertionError("expected ApexQuantPreflightWarning")


def test_quantize_model_preflight_false_overrides():
    from apexquant import quantize_model
    # MobileNet-V2 audit is BAD, but preflight=False should let it through.
    model = tvm.mobilenet_v2(weights=None)
    quant_model, stats = quantize_model(model, bits=4, preflight=False)
    assert quant_model is not None
    assert len(stats) > 0
