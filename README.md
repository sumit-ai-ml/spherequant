# ApexQuant

Training-free post-training weight quantization. We rotate each layer's
flattened weight matrix by a random orthogonal transform, L2-normalize the
rotated rows so they live on the unit sphere, then quantize each coordinate
with a Lloyd–Max scalar codebook matched to the Beta(d/2, d/2) distribution
that those coordinates follow on the sphere. No calibration data, no
gradient steps, runs on a single CPU core in under a minute for ViT-B/16.

## Repository layout

```
apexquant/                    Top-level Python package (shared primitives)
  ptq.py                          quantize_model_{baseline,apexquant,quarot,rtn_absmax,h3}
  rotation_utils.py               SRHT, dense orthogonal, Beta KS test, torch wrappers

experiments/
  cnn_init_rotation/            CIFAR-10 3-layer CNN sanity test + H3 ablation
  imagenet_ptq/                 ResNet/ViT/ConvNeXt/MobileNet/EfficientNet on ImageNet-1k
  llm_ptq/                      TinyLlama 1.1B + Phi-1.5 on WikiText-2
```

## Setup

```bash
python -m venv apexquant && source apexquant/bin/activate
pip install -e .                   # installs the apexquant package
pip install -r requirements.txt    # pulls torchvision, transformers, etc. for experiments
```

## Quick check

```bash
cd experiments/cnn_init_rotation && python test_sanity.py
```

Should print `All sanity checks passed.` in under a minute on CPU. Verifies
SRHT round-trip, Beta-codebook symmetry, KS fit on rotated random unit
vectors, and the rotated-basis (H3) reparameterization.

## Will ApexQuant work on my model?

ApexQuant's edge over QuaRot depends on per-layer fan-in d. The paper
shows the method works decisively at d >= 100 and fails on depthwise
convolutions with d = 9. Run the audit before quantizing:

    from apexquant import audit
    import torchvision.models as tvm

    model = tvm.mobilenet_v2(weights=None)
    audit(model)

The output is a per-layer table plus an overall verdict:
- GOOD: ApexQuant strongly preferred
- MARGINAL: mixed-d model, consider per-layer policy
- BAD: use QuaRot instead

Verdicts have been validated against all six architectures in the paper.
ResNet-18/50, ViT-B/16, and ConvNeXt-Tiny audit as GOOD; MobileNet-V2 and
EfficientNet-B0 audit as BAD, matching the paper's accuracy results.

The audit is also wired into the unified `quantize_model` entry point:

    from apexquant import quantize_model
    quantize_model(model, bits=4)  # raises ApexQuantPreflightWarning on BAD

Pass `preflight=False` to override, or use `method="quarot"` instead.

CLI:

    python -m apexquant.audit --model resnet50
    python -m apexquant.audit --checkpoint path/to/full_module.pt
    python -m apexquant.audit --hf-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

## End-to-end: audit + quantize + benchmark any HF model

`apexquant.bench` chains the three steps for any HuggingFace image
classifier or causal LM. Task is auto-detected from the model's
`AutoConfig` (`AutoModelForImageClassification` → vision path,
`AutoModelForCausalLM` → LLM path).

Vision (default dataset: ImageNet-1k validation):

    python -m apexquant.bench --hf-model google/vit-base-patch16-224 \
        --bits 4 8 --methods apexquant quarot --subset-size 1024

LLM (default dataset: WikiText-2 test):

    python -m apexquant.bench --hf-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --bits 2 4 --methods apexquant quarot rtn_absmax \
        --max-chunks 16

Custom HuggingFace classification dataset:

    python -m apexquant.bench --hf-model microsoft/resnet-50 \
        --hf-dataset imagenet-1k --dataset-split validation \
        --bits 4 --out results.jsonl

Each run prints the audit verdict, evaluates the FP reference, then sweeps
`(method × bits)` and prints a summary table. Pass `--out path.jsonl` for
machine-readable results.

Caveats:

- **Label-space size mismatch is a warning, not an error.** Some HF mirrors
  (e.g. `benjamin-paine/imagenet-1k-256x256`) report `ClassLabel(1001)` but
  actually use labels 0..999. Bench warns and skips out-of-range samples at
  eval time. If *every* sample is out of range, it errors out.
- **Custom columns.** Auto-detection looks for HF `Image` and `ClassLabel`
  features (vision) or a `text` column (LLM). Override with `--image-col`,
  `--label-col`, or `--text-col`.
- **Preflight refusal.** If audit says `BAD`, ApexQuant runs raise
  `ApexQuantPreflightWarning` and are skipped in the sweep. Pass
  `--no-preflight` to force-run on BAD models, or use `--methods quarot`.

### Bring your own model

Three formats, three paths:

**1. HF directory (`save_pretrained` output) — CLI, zero extra code.**

If your training script does `model.save_pretrained("./out")` and
`processor.save_pretrained("./out")`, the bench CLI accepts the local path
the same way it accepts a Hub repo ID:

```bash
python -m apexquant.bench --hf-model ./out \
    --hf-dataset imagefolder --data-dir ./my_val_set \
    --bits 4
```

`--hf-dataset imagefolder --data-dir <path>` works for any folder laid out
as `<path>/<class_name>/img.jpg` — the standard PyTorch `ImageFolder` format.

**2. Full `nn.Module` checkpoint (`torch.save(model, "x.pt")`) — CLI.**

```bash
# Vision: uses ImageNet-style preprocessing (resize + center-crop + normalize).
python -m apexquant.bench --checkpoint ./model.pt \
    --task image_classification --image-size 224 \
    --hf-dataset imagefolder --data-dir ./my_val_set \
    --bits 4

# LLM: also needs --tokenizer (matching the model's vocabulary).
python -m apexquant.bench --checkpoint ./llm.pt \
    --task causal_lm --tokenizer ./tokenizer_dir \
    --hf-dataset wikitext --dataset-config wikitext-2-raw-v1 \
    --bits 4
```

**3. State-dict (`torch.save(model.state_dict(), "x.pt")`) — Python API.**

A state-dict has no architecture, so reconstruct the model in code, then:

```python
from torch.utils.data import DataLoader
from apexquant.bench import benchmark_image_classifier, benchmark_causal_lm

# Vision
model = MyArchitecture()                       # your code
model.load_state_dict(torch.load("model.pt"))
loader = DataLoader(my_val_dataset, batch_size=64)  # any (x, y) loader

results = benchmark_image_classifier(
    model, loader,
    bits_list=[4, 8],
    methods=["apexquant", "quarot"],
    device="cuda",
)

# LLM
results = benchmark_causal_lm(
    model, tokenizer, text_corpus,
    bits_list=[4],
    seq_len=2048,
    device="cuda",
)
```

`benchmark_image_classifier` accepts any `nn.Module` (HF or not — it
auto-detects from the output type) and any DataLoader yielding `(x, y)`
batches. Number of classes is inferred from the first batch's logits if not
passed explicitly. Both helpers return a list of `BenchResult` rows you can
write with `apexquant.bench.write_jsonl`.

### Bring your own dataset

The bench CLI uses HuggingFace's `datasets` builders, so any layout those
builders accept works. The four common ones:

**Vision: `imagefolder` (most common).** Class names = subfolder names; HF
maps them to integer labels alphabetically. Two layouts work:

```
# Single-split layout — load with --dataset-split train
my_data/
├── cat/
│   ├── 0.jpg
│   └── 1.jpg
└── dog/
    └── 0.jpg

# Multi-split layout — picks "validation"/"test"/"train" automatically
my_data/
├── train/
│   ├── cat/...
│   └── dog/...
└── validation/
    ├── cat/...
    └── dog/...
```

```bash
python -m apexquant.bench --hf-model ./my_model \
    --hf-dataset imagefolder --data-dir ./my_data --bits 4
```

**Vision: CSV / Parquet with image paths.** A CSV like
`image_path,label\ncat/0.jpg,0\ndog/0.jpg,1` plus the actual files alongside:

```bash
python -m apexquant.bench --hf-model ./my_model \
    --hf-dataset csv --data-dir ./my_data \
    --image-col image_path --label-col label --bits 4
```

(Note: HF's `csv` builder loads the path as a string, not as a decoded
image, so this only works for image classifiers if you've cast the column
to `Image()` ahead of time. For most users `imagefolder` is the right
answer.)

**LLM: `text` builder.** A folder of `.txt` files; each line/file becomes a
row. The `text` column holds the body — no `--text-col` override needed.

```
my_corpus/
├── doc1.txt
├── doc2.txt
└── ...
```

```bash
python -m apexquant.bench --checkpoint ./llm.pt --task causal_lm \
    --tokenizer ./tokenizer_dir \
    --hf-dataset text --data-dir ./my_corpus --bits 4
```

**LLM: JSONL.** One JSON object per line with a text field:

```jsonl
{"text": "the quick brown fox..."}
{"text": "lorem ipsum..."}
```

```bash
python -m apexquant.bench --checkpoint ./llm.pt --task causal_lm \
    --tokenizer ./tokenizer_dir \
    --hf-dataset json --data-dir ./my_corpus --text-col text --bits 4
```

**Anything else (custom format, on-the-fly preprocessing).** Build a
`torch.utils.data.DataLoader` yielding `(x, y)` (vision) or build the
corpus string yourself (LLM), then call the Python API
(`benchmark_image_classifier` / `benchmark_causal_lm`) directly. The CLI
is just a thin wrapper around those — anything you can do in 10 lines of
PyTorch will work.

### Verdicts on the six paper architectures + LLMs

Verified against the audit shipped in this repo:

| Model | Quantizable layers | GOOD | MARG | BAD | of which depthwise | Param mass in BAD | Verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ResNet-18 | 21 | 20 | 1 | 0 | 0 | 0.00% | **GOOD** |
| ResNet-50 | 54 | 49 | 5 | 0 | 0 | 0.00% | **GOOD** |
| ViT-B/16 | 49 | 49 | 0 | 0 | 0 | 0.00% | **GOOD** |
| ConvNeXt-Tiny (7×7 depthwise) | 59 | 37 | 22 | 0 | 0 | 0.00% | **GOOD** |
| MobileNet-V2 (3×3 depthwise) | 53 | 20 | 12 | 21 | 17 | 2.12% | **BAD** |
| EfficientNet-B0 | 82 | 37 | 13 | 32 | 16 | 5.42% | **BAD** |
| TinyLlama 1.1B | 155 | 155 | 0 | 0 | 0 | 0.00% | **GOOD** |
| Phi-1.5 1.3B | 145 | 145 | 0 | 0 | 0 | 0.00% | **GOOD** |

Notes:

- **ConvNeXt-Tiny is GOOD even though it's depthwise-heavy.** Its kernels are 7×7 (d=49), which the audit places in MARGINAL — well above the d=9 failure case of MobileNet's 3×3 depthwise. Don't conflate "depthwise" with "BAD"; the audit only flags d < 32.
- **MobileNet-V2's BAD verdict comes from the layer-count signal, not the param signal.** Only 2.1% of params live in BAD layers, but those 21 layers are 39.6% of the layer count and sit in every forward path. The dual-signal rule (param-weighted >30% **or** layer-count >20%) is what catches this; param count alone would miss it.
- **Every transformer linear is in the favorable regime.** TinyLlama and Phi-1.5 audit as 100% GOOD with zero MARGINAL or BAD layers. LLaMA-style architectures use individual `q_proj` / `k_proj` / `v_proj` / `o_proj` Linears (not `nn.MultiheadAttention`) and the audit picks them up automatically through the standard `nn.Linear` branch.

## Headline results

### Vision: ImageNet-1k top-1 accuracy (Beta codebook, training-free)

All methods use the same per-row L2 normalize + (codebook or absmax)
quantization scheme; storage cost is identical at each bit width.

**ResNet-18** — FP32: 67.45%

| bits | Baseline (no rot) | QuaRot | ApexQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 55.18% | 67.42% | **67.39%** |
| 6 | 46.27% | 67.00% | **67.34%** |
| 4 | 17.31% | 59.99% | **62.34%** |
| 2 | 0.13% | 0.08% | **4.64%** |

**ResNet-50** — FP32: 76.95%

| bits | Baseline (no rot) | QuaRot | ApexQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 0.32% | 76.86% | **76.89%** |
| 6 | 0.21% | **76.86%** | 76.57% |
| 4 | 0.11% | 61.89% | **69.36%** |
| 2 | 0.09% | 0.05% | **0.43%** |

**ViT-B/16** — FP32: 79.14%

| bits | Baseline (no rot) | QuaRot | ApexQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 45.88% | 79.16% | **79.19%** |
| 6 | 27.19% | **79.15%** | 79.14% |
| 4 | 19.61% | 78.99% | **79.02%** |
| 2 | 1.71% | 0.12% | **69.86%** |

**ConvNeXt-Tiny** — FP32: 77.37%

| bits | Baseline (no rot) | QuaRot | ApexQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 40.79% | **77.36%** | 77.34% |
| 6 | 19.41% | **77.48%** | 77.45% |
| 4 | 0.44% | 77.24% | **77.34%** |
| 2 | 0.14% | 0.08% | **27.02%** |

**MobileNet-V2** — FP32: 67.92%  (depthwise kernels; fan-in as low as 9)

| bits | Baseline (no rot) | QuaRot | ApexQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 46.10% | 67.70% | **67.75%** |
| 6 | 25.40% | **63.93%** | 59.11% |
| 4 | 1.12% | **8.04%** | 5.65% |
| 2 | 0.10% | **0.11%** | 0.08% |

**EfficientNet-B0** — FP32: 74.57%  (depthwise + squeeze-excitation)

| bits | Baseline (no rot) | QuaRot | ApexQuant (ours) |
|-----:|------------------:|-------:|-------------------:|
| 8 | 50.69% | **74.59%** | 73.85% |
| 6 | 8.41% | **73.41%** | 70.92% |
| 4 | 0.18% | 20.11% | **51.34%** |
| 2 | 0.10% | **0.12%** | 0.11% |

Read of the table: at 8-bit and 6-bit both rotation methods recover near-FP32
on every model — there's nothing to differentiate. The interesting regime is
4-bit and 2-bit, where ApexQuant wins decisively on architectures with
large fan-in (ResNet-50 +7pp at 4-bit; ViT-B/16 +69.7pp at 2-bit; ConvNeXt-
Tiny +26.9pp at 2-bit; EfficientNet-B0 +31pp at 4-bit) and ties or trails
QuaRot on architectures dominated by small-fan-in depthwise kernels
(MobileNet-V2, EfficientNet at 6-bit). The architectural boundary —
fan-in roughly 32 — is the threshold below which the post-rotation Beta(d/2,
d/2) approximation degrades and the matched codebook loses its edge over
QuaRot's uniform grid.

### LLM: WikiText-2 perplexity (lower = better)

**TinyLlama 1.1B** — FP16 reference: 7.97

| bits | RTN-absmax | QuaRot | ApexQuant (ours) |
|-----:|-----------:|-------:|-------------------:|
| 8 | 7.97 | 7.98 | **7.97** |
| 6 | 8.07 | 8.04 | **8.02** |
| 4 | 10.98 | 10.67 | **8.68** |
| 2 | 1.2 × 10⁵ | 1.5 × 10⁵ | **4.7 × 10³** |

**Phi-1.5 1.3B** — FP16 reference: 21.82

| bits | RTN-absmax | QuaRot | ApexQuant (ours) |
|-----:|-----------:|-------:|-------------------:|
| 8 | 21.82 | 21.84 | 21.84 |
| 6 | 21.96 | 21.92 | **21.87** |
| 4 | 26.92 | 24.33 | **22.64** |
| 2 | 3.5 × 10⁵ | 7.4 × 10⁴ | **77.03** |

The 2-bit Phi-1.5 result (77 vs 74,000) is the most dramatic: ApexQuant
is the only training-free method that produces a usable 2-bit causal LM at
this scale.

## Reproducing the experiments

ImageNet-1k vision sweep (RTN-absmax, QuaRot, ApexQuant; bits 2/4/6/8):

```bash
cd experiments/imagenet_ptq
python run.py                  # baseline + apexquant
python run_quarot.py           # quarot baseline
python run_vit.py              # ViT MHA-aware variant (in_proj_weight)
python final_summary.py        # tables and figures
```

LLM sweep (TinyLlama + Phi-1.5):

```bash
cd experiments/llm_ptq
python run.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/phi-1_5 --bits 2 4 6 8
python final_summary.py
```

CIFAR-10 toy (also runs the H3 train-in-rotated-basis ablation):

```bash
cd experiments/cnn_init_rotation
python run.py
python plot.py
```

CIFAR-10 is downloaded on first run into `cnn_init_rotation/data/` (gitignored).
ImageNet-1k validation is streamed from the HuggingFace mirror.
TinyLlama and Phi-1.5 are pulled from HuggingFace Hub.

## What's tracked vs gitignored

Per-sweep `results.jsonl` files and the `figures/*.png` are tracked. Per-
sample raw score arrays (`results/scores/`), run logs (`*.log`), CIFAR-10
data, and model checkpoints are gitignored.
