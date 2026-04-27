"""ImageNet-1k val loader via HuggingFace `benjamin-paine/imagenet-1k-256x256`.

Caches the dataset on disk (first run downloads ~7 GB). Applies the standard
torchvision preprocessing for pretrained models: center-crop 224, normalize
with ImageNet mean/std. Returns a PyTorch DataLoader.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import load_dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HF_DATASET = "benjamin-paine/imagenet-1k-256x256"


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HFImageNetVal(Dataset):
    """Thin torch Dataset wrapper over the HF val split.

    Images already 256x256. We center-crop to 224 and normalize.
    """

    def __init__(self, hf_dataset, transform):
        self.hf = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        sample = self.hf[idx]
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.transform(img)
        y = int(sample["label"])
        return x, y


def get_imagenet_val_loader(batch_size: int = 64, num_workers: int = 4,
                            subset_size: Optional[int] = None) -> DataLoader:
    """Download (first time) + return a DataLoader over ImageNet val.

    subset_size: if not None, use only the first N images (for smoke tests).
    """
    print(f"Loading HF dataset {HF_DATASET} (cached in ~/.cache/huggingface)...")
    hf = load_dataset(HF_DATASET, split="validation")
    if subset_size is not None:
        hf = hf.select(range(min(subset_size, len(hf))))
    print(f"  {len(hf)} validation images ready.")

    torch_ds = HFImageNetVal(hf, build_transform())
    return DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
