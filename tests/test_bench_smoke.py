"""Smoke tests for spherequant.bench.

Two layers:

1. Unit-level (always runs in CI): result formatting, JSONL writing, and the
   small helpers in vision/llm that don't need transformers or datasets.

2. End-to-end (skipped if transformers/datasets unavailable): runs the full
   bench pipeline on hf-internal-testing tiny random models. These are the
   tiny test fixtures HF maintains for exactly this purpose: ~1MB each,
   load in seconds, no network of substance after caching.

CI installs only the [test] extras (torchvision + pytest), so layer 2 is
gated behind importorskip and won't run in the matrix. Local devs with
[experiments] extras get the full coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spherequant.bench import BenchResult, format_summary, write_jsonl


# ---------- Layer 1: unit tests ---------------------------------------------

def test_format_summary_image_classification_includes_drop_column():
    rows = [
        BenchResult(model="m", task="image_classification", dataset="d",
                    variant="reference", bits=32, codebook="fp32",
                    n_samples=8, top1=0.50, top5=1.00, elapsed_s=1.0),
        BenchResult(model="m", task="image_classification", dataset="d",
                    variant="spherequant", bits=4, codebook="beta",
                    n_samples=8, top1=0.25, top5=0.75, elapsed_s=2.0),
    ]
    out = format_summary(rows)
    assert "spherequant" in out
    assert "Δtop1" in out
    # 0.25 - 0.50 = -25.00pp
    assert "-25.00" in out


def test_format_summary_causal_lm_uses_perplexity_column():
    rows = [
        BenchResult(model="m", task="causal_lm", dataset="d",
                    variant="reference", bits=16, codebook="fp16",
                    n_samples=2, perplexity=10.0, elapsed_s=1.0),
        BenchResult(model="m", task="causal_lm", dataset="d",
                    variant="spherequant", bits=4, codebook="beta",
                    n_samples=2, perplexity=12.5, elapsed_s=2.0),
    ]
    out = format_summary(rows)
    assert "perplexity" in out
    assert "10.000" in out
    assert "12.500" in out


def test_write_jsonl_round_trip(tmp_path: Path):
    rows = [
        BenchResult(model="m", task="causal_lm", dataset="d",
                    variant="reference", bits=16, codebook="fp16",
                    n_samples=1, perplexity=7.5, elapsed_s=0.1),
    ]
    path = tmp_path / "out.jsonl"
    write_jsonl(rows, path)
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded[0]["perplexity"] == 7.5
    assert loaded[0]["variant"] == "reference"


# ---------- Layer 2: end-to-end with HF tiny test fixtures ------------------

@pytest.fixture(scope="module")
def _hf_available():
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")


def test_llm_path_end_to_end(_hf_available, monkeypatch):
    """Quantize + benchmark a tiny random Llama on a tiny synthetic corpus."""
    import datasets as _datasets
    from datasets import Dataset

    from spherequant.bench import llm

    fake_ds = Dataset.from_dict({"text": ["hello world " * 50] * 8})
    monkeypatch.setattr(_datasets, "load_dataset",
                        lambda *a, **kw: {"test": fake_ds})

    # preflight=False because tiny test fixtures have small hidden sizes that
    # may trip the architectural-boundary refusal — we're testing wiring, not
    # the preflight rule itself.
    results = llm.run(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        dataset_id="ignored",
        dataset_config=None,
        dataset_split="test",
        text_col="text",
        bits_list=[4],
        methods=["spherequant"],
        codebook="beta",
        rotation_seed=0,
        seq_len=32,
        max_chunks=2,
        subset_size=None,
        device="cpu",
        preflight=False,
    )

    assert len(results) == 2
    assert [r.variant for r in results] == ["reference", "spherequant"]
    for r in results:
        assert r.perplexity is not None and r.perplexity > 0
        assert r.task == "causal_lm"


def test_vision_path_end_to_end(_hf_available, monkeypatch):
    """Quantize + benchmark a tiny random ViT on a tiny synthetic image set."""
    import numpy as np
    import datasets as _datasets
    from datasets import ClassLabel, Dataset, Features, Image as HFImage
    from PIL import Image
    from transformers import AutoConfig

    from spherequant.bench import vision

    model_id = "hf-internal-testing/tiny-random-ViTForImageClassification"
    n_classes = AutoConfig.from_pretrained(model_id).num_labels

    images = [Image.fromarray(
        (np.random.RandomState(i).rand(32, 32, 3) * 255).astype(np.uint8)
    ) for i in range(4)]
    labels = [i % n_classes for i in range(4)]
    fake_ds = Dataset.from_dict(
        {"image": images, "label": labels},
        features=Features({"image": HFImage(),
                           "label": ClassLabel(num_classes=n_classes)}),
    )
    monkeypatch.setattr(_datasets, "load_dataset",
                        lambda *a, **kw: {"validation": fake_ds})

    results = vision.run(
        model_id=model_id,
        dataset_id="ignored",
        dataset_config=None,
        dataset_split="validation",
        image_col="image",
        label_col="label",
        bits_list=[4],
        methods=["spherequant"],
        codebook="beta",
        rotation_seed=0,
        subset_size=None,
        batch_size=2,
        num_workers=0,
        device="cpu",
        preflight=False,
    )

    assert len(results) == 2
    assert [r.variant for r in results] == ["reference", "spherequant"]
    for r in results:
        assert r.top1 is not None
        assert 0.0 <= r.top1 <= 1.0
        assert r.task == "image_classification"


# ---------- Layer 3: local Python API + checkpoint CLI ----------------------

import torch.nn as _nn  # noqa: E402


class TinyMLPClassifier(_nn.Module):
    """A non-HF nn.Module that returns logits directly (no .logits wrapper).
    Module-level class so torch.save can pickle it for the CLI test."""

    def __init__(self, in_dim=3 * 16 * 16, n_classes=4, hidden=128):
        super().__init__()
        self.flat = _nn.Flatten()
        self.fc1 = _nn.Linear(in_dim, hidden)
        self.fc2 = _nn.Linear(hidden, hidden)
        self.head = _nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.fc2(self.fc1(self.flat(x)).relu()).relu())


def _make_tiny_torchvision_classifier():
    return TinyMLPClassifier()


def test_benchmark_image_classifier_local_torchvision_style():
    """Public Python API on a plain nn.Module + plain DataLoader."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from spherequant.bench import benchmark_image_classifier

    torch.manual_seed(0)
    model = _make_tiny_torchvision_classifier()
    n_classes = 4
    x = torch.randn(16, 3, 16, 16)
    y = torch.randint(0, n_classes, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    results = benchmark_image_classifier(
        model, loader,
        model_name="tiny_mlp", dataset_name="synthetic",
        bits_list=[4],
        methods=["spherequant"],
        device="cpu",
        preflight=False,  # tiny d on the head; we're testing wiring
        verbose=False,
    )
    assert len(results) == 2
    assert [r.variant for r in results] == ["reference", "spherequant"]
    for r in results:
        assert r.top1 is not None
        assert r.model == "tiny_mlp"
        assert r.dataset == "synthetic"


def test_benchmark_image_classifier_infers_n_classes_from_logits():
    """If n_classes isn't passed and the model has no .config, it should be
    inferred from the first batch's logits shape — no crash."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from spherequant.bench import benchmark_image_classifier

    model = _make_tiny_torchvision_classifier()  # 4 classes, no .config
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 16, 16), torch.zeros(8, dtype=torch.long)),
        batch_size=4,
    )
    results = benchmark_image_classifier(
        model, loader, methods=["spherequant"], bits_list=[4],
        device="cpu", preflight=False, verbose=False,
    )
    # All labels are 0 (in-range for 4-class model), nothing skipped.
    assert results[0].extra.get("n_skipped_oor", 0) == 0


def test_cli_checkpoint_round_trip(tmp_path: Path, monkeypatch):
    """Save a tiny module with torch.save, then run the CLI on it via run()."""
    import torch
    from datasets import ClassLabel, Dataset, Features, Image as HFImage
    import datasets as _datasets
    import numpy as np
    from PIL import Image

    pytest.importorskip("transformers")
    from spherequant.bench import vision

    n_classes = 4
    model = _make_tiny_torchvision_classifier()
    ckpt_path = tmp_path / "tiny.pt"
    torch.save(model, ckpt_path)

    images = [Image.fromarray(
        (np.random.RandomState(i).rand(16, 16, 3) * 255).astype(np.uint8)
    ) for i in range(8)]
    labels = [i % n_classes for i in range(8)]
    fake_ds = Dataset.from_dict(
        {"image": images, "label": labels},
        features=Features({"image": HFImage(),
                           "label": ClassLabel(num_classes=n_classes)}),
    )
    monkeypatch.setattr(_datasets, "load_dataset",
                        lambda *a, **kw: {"validation": fake_ds})

    results = vision.run(
        model_id=None,
        checkpoint_path=str(ckpt_path),
        image_size=16,
        dataset_id="ignored",
        dataset_config=None,
        dataset_split="validation",
        data_dir=None,
        image_col="image",
        label_col="label",
        bits_list=[4],
        methods=["spherequant"],
        codebook="beta",
        rotation_seed=0,
        subset_size=None,
        batch_size=4,
        num_workers=0,
        device="cpu",
        preflight=False,
    )
    assert len(results) == 2
    assert [r.variant for r in results] == ["reference", "spherequant"]
    assert results[0].model == str(ckpt_path)


def test_cli_checkpoint_rejects_state_dict(tmp_path: Path):
    """If the user saved a state_dict instead of a full module, refuse with a
    clear pointer to the Python API."""
    import torch
    from spherequant.bench import vision

    sd_path = tmp_path / "sd.pt"
    torch.save({"weight": torch.zeros(2, 2)}, sd_path)

    with pytest.raises(SystemExit, match="full nn.Module"):
        vision.run(
            model_id=None,
            checkpoint_path=str(sd_path),
            image_size=16,
            dataset_id="ignored",
            dataset_config=None,
            dataset_split=None,
            data_dir=None,
            image_col=None, label_col=None,
            bits_list=[4], methods=["spherequant"], codebook="beta",
            rotation_seed=0, subset_size=None, batch_size=4, num_workers=0,
            device="cpu", preflight=False,
        )
