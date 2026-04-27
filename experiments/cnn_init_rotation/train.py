"""CIFAR-10 training for baseline CNN3 and RotatedCNN3 (H3).

Deterministic given (seed, variant). 10 epochs, SGD+momentum, cosine LR.
Target: ~75% test accuracy on baseline CNN3 (reasonable for a 3-layer CNN
without augmentation beyond the standard normalization).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN3
from rotated_conv import RotatedCNN3


DATA_DIR = Path(__file__).resolve().parent / "data"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    # lr=0.05 was too high for H3 with momentum: H3's effective weight W = U @ M.T
    # reaches a Kaiming-equivalent distribution but the rotation-equivariance of
    # momentum SGD is numerically fragile. lr=0.01 trains both baseline and H3 to
    # similar FP32 accuracy.
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size: int, num_workers: int = 2):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_model(variant: str, seed: int) -> nn.Module:
    """variant in {baseline, h3_rotated_basis}. SphereQuant uses the baseline at train time."""
    if variant in ("baseline", "spherequant"):
        return CNN3()
    if variant == "h3_rotated_basis":
        # H3 derives per-layer rotation seeds from this top-level seed
        return RotatedCNN3(seed=seed)
    raise ValueError(f"unknown variant: {variant}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train(variant: str, seed: int, cfg: TrainConfig | None = None,
          device: str | None = None, verbose: bool = True) -> dict:
    cfg = cfg or TrainConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_loader, test_loader = get_dataloaders(cfg.batch_size, cfg.num_workers)
    model = build_model(variant, seed).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr,
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        scheduler.step()
        train_loss = running_loss / max(1, n_seen)
        test_acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc})
        if verbose:
            print(f"  [{variant} seed={seed}] epoch {epoch+1}/{cfg.epochs}  "
                  f"loss={train_loss:.4f}  test_acc={test_acc*100:.2f}%")

    elapsed = time.time() - t0
    final_acc = evaluate(model, test_loader, device)
    return {
        "variant": variant,
        "seed": seed,
        "final_test_acc": final_acc,
        "history": history,
        "elapsed_s": elapsed,
        "model": model,
    }
