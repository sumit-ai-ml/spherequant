# ImageNet PTQ: does TurboQuant rotation help mainstream models?

Post-training quantization sweep on pretrained ImageNet models.

## Scope

This experiment tests **H2 (post-hoc rotation)** only — take a torchvision
pretrained model, rotate each layer's flattened weights on the fan-in axis,
then quantize. No retraining. Reuses the quantization primitives from
`../cnn_init_rotation/ptq.py`.

H3 (train-in-rotated-basis) is skipped here because it would defeat the
"pretrained" premise.

## Models

All loaded from `torchvision.models` with ImageNet weights:

- **ResNet-18** (11.7M params) — residual, BatchNorm, standard 3x3 convs
- **ResNet-50** (25.6M params) — bottleneck blocks with 1x1 + 3x3 convs
- **MobileNet-V2** (3.5M params) — depthwise separable convs (small fan-in)

Skipped (FC layers too large for naive dense rotation):
- AlexNet (FC1 = 9216 → 4096 would need a 340 MB rotation matrix)
- VGG-16 (FC1 = 25088 → 4096, 2.5 GB rotation matrix)

To add these later, implement padded-SRHT for large fan-in.

## Dataset

HuggingFace `benjamin-paine/imagenet-1k-256x256` (ungated, pre-resized to 256).
Standard ImageNet val split: 50,000 images, 1000 classes.

Preprocessing: center-crop to 224x224, ImageNet normalize
`[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` — matches torchvision's
pretrained expectations.

## Sweep

- Models: resnet18, resnet50, mobilenet_v2
- Quantization: H2 (rotate then quantize) vs baseline (quantize directly)
- Bits: {2, 4, 6, 8}
- Codebooks: {uniform, beta}
- = 3 models x (1 FP32 baseline + 2 variants x 4 bits x 2 codebooks) = 51 evaluations

Metrics per evaluation: top-1 accuracy, top-5 accuracy, model size in MB.

## Runtime

First run: ~15 min to download + cache ~7 GB dataset. Then ~60 min for the
full sweep on an A3000 laptop GPU. Subsequent runs reuse the cache.

## How to run

```bash
python run.py --models resnet18 resnet50 mobilenet_v2 --bits 2 4 6 8
python plot.py
```

Results: `results/results.jsonl`. Figures: `figures/`.
