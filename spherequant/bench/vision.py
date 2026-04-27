"""Vision benchmark path: image classifier + image dataset.

Two entry points:

  - :func:`benchmark_image_classifier` — public Python API. Takes any
    ``nn.Module`` and any ``DataLoader`` yielding ``(x, y)``. Use this when
    your model is a torchvision-style classifier, a state-dict you've
    already loaded, or anything that isn't a HuggingFace
    ``AutoModelForImageClassification``.
  - :func:`run` — CLI helper called by ``python -m spherequant.bench``.
    Loads an HF model + HF dataset, builds the loader, then dispatches to
    ``benchmark_image_classifier``.

Default HF dataset for ``run``: ``benjamin-paine/imagenet-1k-256x256``
validation split, the same one the paper's ImageNet sweep uses.
"""

from __future__ import annotations

import copy
import gc
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spherequant.audit import audit
from spherequant.bench._eval import BenchResult
from spherequant.exceptions import SphereQuantPreflightWarning
from spherequant.ptq import quantize_model

DEFAULT_DATASET = "benjamin-paine/imagenet-1k-256x256"
DEFAULT_SPLIT = "validation"


def _pick_split(hf, requested: Optional[str]) -> str:
    """Honor a requested split if given, else prefer validation > test > train."""
    if requested is not None:
        return requested
    if hasattr(hf, "keys"):
        keys = list(hf.keys())
    else:
        return "validation"
    for cand in ("validation", "val", "test", "train"):
        if cand in keys:
            return cand
    return keys[0]


def _detect_columns(hf_split, image_col: Optional[str], label_col: Optional[str]) -> tuple[str, str]:
    """Find the image and label columns by feature type if not explicitly given."""
    from datasets import ClassLabel, Image as HFImage

    features = hf_split.features
    if image_col is None:
        for name, feat in features.items():
            if isinstance(feat, HFImage):
                image_col = name
                break
    if image_col is None:
        raise SystemExit(
            f"Could not find an image column. Dataset features: "
            f"{list(features.keys())}. Pass --image-col explicitly."
        )

    if label_col is None:
        for name, feat in features.items():
            if isinstance(feat, ClassLabel):
                label_col = name
                break
        if label_col is None and "label" in features:
            label_col = "label"
    if label_col is None:
        raise SystemExit(
            f"Could not find a label column. Dataset features: "
            f"{list(features.keys())}. Pass --label-col explicitly."
        )
    return image_col, label_col


class _HFImageDataset(Dataset):
    """Wrap an HF image dataset with the model's AutoImageProcessor."""

    def __init__(self, hf_split, processor, image_col: str, label_col: str):
        self.hf = hf_split
        self.processor = processor
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        sample = self.hf[idx]
        img = sample[self.image_col]
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        out = self.processor(images=img, return_tensors="pt")
        pixel_values = out["pixel_values"].squeeze(0)
        label = int(sample[self.label_col])
        return pixel_values, label


def _torchvision_transform(image_size: int = 224):
    """Standard ImageNet preprocessing for torchvision-style models."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class _HFImageDatasetTorchvision(Dataset):
    """HF image dataset wrapped with torchvision transforms (no HF processor)."""

    def __init__(self, hf_split, transform, image_col: str, label_col: str):
        self.hf = hf_split
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        sample = self.hf[idx]
        img = sample[self.image_col]
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(sample[self.label_col])


def _logits_from_output(out) -> torch.Tensor:
    """Unwrap HF ImageClassifierOutput.logits or pass-through a raw tensor."""
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def _eval_top1_top5(model: nn.Module, loader, device: str,
                    n_classes: Optional[int] = None
                    ) -> tuple[float, float, int, int]:
    """Returns (top1, top5, n_scored, n_skipped). Samples with labels outside
    ``[0, n_classes)`` are skipped (and counted) — handles HF mirrors whose
    ClassLabel metadata is off-by-one from the actual label range.

    If ``n_classes`` is None, infer it from the first batch's logits.
    """
    model.eval()
    correct1, correct5, total, skipped = 0, 0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if n_classes is None:
            with torch.no_grad():
                probe = _logits_from_output(model(x[:1]))
            n_classes = probe.shape[1]

        in_range = y < n_classes
        skipped += int((~in_range).sum().item())
        if not bool(in_range.any()):
            continue
        x = x[in_range]
        y = y[in_range]
        logits = _logits_from_output(model(x))
        k = min(5, logits.shape[1])
        _, pred_topk = logits.topk(k, dim=1)
        correct1 += (pred_topk[:, 0] == y).sum().item()
        correct5 += (pred_topk == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
    if total == 0:
        raise SystemExit(
            "Every sample was out of the model's label range. "
            "The dataset's labels don't match the model's classifier head."
        )
    return correct1 / total, correct5 / total, total, skipped


def benchmark_image_classifier(
    model: nn.Module,
    dataloader,
    *,
    model_name: str = "<local>",
    dataset_name: str = "<local>",
    n_classes: Optional[int] = None,
    bits_list: list[int] = (4, 8),
    methods: list[str] = ("spherequant", "quarot", "rtn_absmax"),
    codebook: str = "beta",
    rotation_seed: int = 0,
    device: Optional[str] = None,
    preflight: bool = True,
    verbose: bool = True,
) -> list[BenchResult]:
    """Audit, quantize, and benchmark any image classifier.

    The model can be any ``nn.Module`` whose forward call accepts a batch of
    images and returns logits (or an HF ``ImageClassifierOutput``).
    The dataloader yields ``(x, y)`` batches with integer class labels.

    ``n_classes`` is inferred from the first batch's logits if not supplied.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if n_classes is None and hasattr(model, "config") and hasattr(model.config, "num_labels"):
        n_classes = model.config.num_labels

    if verbose:
        print("\nRunning fan-in audit...")
    audit(model, verbose=verbose)

    results: list[BenchResult] = []

    if verbose:
        print("\nEvaluating reference model...")
    t0 = time.time()
    model_dev = model.to(device)
    top1, top5, n, skipped = _eval_top1_top5(model_dev, dataloader, device, n_classes)
    elapsed = time.time() - t0
    if verbose:
        skip_note = f"  (skipped {skipped} OOR)" if skipped else ""
        print(f"  reference: top1={top1*100:.2f}%  top5={top5*100:.2f}%  "
              f"n={n}{skip_note}  [{elapsed:.0f}s]")
    results.append(BenchResult(
        model=model_name, task="image_classification", dataset=dataset_name,
        variant="reference", bits=32, codebook="fp32", n_samples=n,
        top1=top1, top5=top5, elapsed_s=elapsed,
        extra={"n_skipped_oor": skipped} if skipped else {},
    ))
    model = model_dev.cpu()

    for bits in bits_list:
        for method in methods:
            cb = codebook if method in ("spherequant", "baseline") else "symabs_uniform"
            t0 = time.time()
            model_q = copy.deepcopy(model)
            try:
                model_q, _stats = quantize_model(
                    model_q, bits=bits, method=method,
                    codebook=cb, rotation_seed=rotation_seed,
                    preflight=preflight,
                )
            except SphereQuantPreflightWarning as e:
                if verbose:
                    print(f"  {method:11s} bits={bits}  REFUSED by preflight: {e}")
                del model_q
                gc.collect()
                continue
            model_q = model_q.to(device)
            top1, top5, n, skipped = _eval_top1_top5(model_q, dataloader, device, n_classes)
            elapsed = time.time() - t0
            if verbose:
                print(f"  {method:11s} bits={bits}  cb={cb:16s}  "
                      f"top1={top1*100:6.2f}%  top5={top5*100:6.2f}%  "
                      f"[{elapsed:.0f}s]")
            results.append(BenchResult(
                model=model_name, task="image_classification", dataset=dataset_name,
                variant=method, bits=bits, codebook=cb, n_samples=n,
                top1=top1, top5=top5, elapsed_s=elapsed,
                extra={"n_skipped_oor": skipped} if skipped else {},
            ))
            del model_q
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def _load_hf_dataset(dataset_id: str, dataset_config: Optional[str],
                     data_dir: Optional[str]):
    """Call load_dataset with the right kwargs for both Hub and imagefolder."""
    from datasets import load_dataset
    kwargs = {}
    if dataset_config is not None:
        kwargs["name"] = dataset_config
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    return load_dataset(dataset_id, **kwargs)


def run(
    *,
    model_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    image_size: int = 224,
    dataset_id: Optional[str],
    dataset_config: Optional[str],
    dataset_split: Optional[str],
    data_dir: Optional[str] = None,
    image_col: Optional[str],
    label_col: Optional[str],
    bits_list: list[int],
    methods: list[str],
    codebook: str,
    rotation_seed: int,
    subset_size: Optional[int],
    batch_size: int,
    num_workers: int,
    device: str,
    preflight: bool,
) -> list[BenchResult]:
    """CLI helper. Loads model + HF dataset and dispatches to
    :func:`benchmark_image_classifier`."""
    if (model_id is None) == (checkpoint_path is None):
        raise SystemExit("Pass exactly one of model_id or checkpoint_path.")

    dataset_id = dataset_id or DEFAULT_DATASET

    if model_id is not None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print(f"Loading HF model {model_id}...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model_name = model_id
        n_classes = model.config.num_labels
        use_hf_processor = True
    else:
        print(f"Loading checkpoint {checkpoint_path}...")
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise SystemExit(
                "--checkpoint must contain a full nn.Module (torch.save(model, ...)), "
                "not a state_dict. For state-dicts, reconstruct the model in code "
                "and call spherequant.bench.benchmark_image_classifier directly."
            )
        model = obj
        processor = None
        model_name = checkpoint_path
        n_classes = None  # inferred from first batch
        use_hf_processor = False
    model.eval()

    print(f"Loading dataset {dataset_id} (config={dataset_config}, "
          f"data_dir={data_dir})...")
    hf = _load_hf_dataset(dataset_id, dataset_config, data_dir)
    split = _pick_split(hf, dataset_split or DEFAULT_SPLIT)
    hf_split = hf[split] if hasattr(hf, "keys") else hf
    print(f"  using split: {split}  ({len(hf_split)} samples)")

    img_c, lbl_c = _detect_columns(hf_split, image_col, label_col)
    print(f"  columns: image={img_c!r}, label={lbl_c!r}")

    if use_hf_processor:
        from datasets import ClassLabel
        feat = hf_split.features[lbl_c]
        n_data_labels = feat.num_classes if isinstance(feat, ClassLabel) else None
        if n_data_labels is not None and n_data_labels != n_classes:
            print(f"  WARNING: model has {n_classes} classes, dataset's "
                  f"ClassLabel reports {n_data_labels}. Will skip samples "
                  f"whose label is outside [0, {n_classes}) at eval time.")

    if subset_size is not None:
        hf_split = hf_split.select(range(min(subset_size, len(hf_split))))
        print(f"  subset: {len(hf_split)} samples")

    if use_hf_processor:
        torch_ds = _HFImageDataset(hf_split, processor, img_c, lbl_c)
    else:
        torch_ds = _HFImageDatasetTorchvision(
            hf_split, _torchvision_transform(image_size), img_c, lbl_c,
        )
    loader = DataLoader(
        torch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    return benchmark_image_classifier(
        model, loader,
        model_name=model_name, dataset_name=dataset_id, n_classes=n_classes,
        bits_list=bits_list, methods=methods, codebook=codebook,
        rotation_seed=rotation_seed, device=device, preflight=preflight,
    )
