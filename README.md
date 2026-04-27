# SphereQuant

Training-free post-training weight quantization that combines random
orthogonal rotation with a scalar codebook *matched* to the
post-rotation coordinate distribution. The rotation places each weight
row on the unit sphere, where its coordinates follow Beta(d/2, d/2);
the codebook is the Lloyd–Max minimum-MSE quantizer for that
distribution. No calibration data, no optimization, no gradient steps.

The method (internal codename: H2) is described in `paper/methodology.tex`.
Experiments and tables are in `paper/results.tex`.

## Repository layout

```
paper/                          LaTeX (NeurIPS submission)
  methodology.tex
  results.tex
  related_works.tex
  references.bib

experiments/
  cnn_init_rotation/            Vision toy + shared ptq.py
  imagenet_ptq/                 ResNet/ViT/ConvNeXt/MobileNet/EfficientNet on ImageNet-1k
  llm_ptq/                      TinyLlama 1.1B + Phi-1.5 on WikiText-2
```

`imagenet_ptq/run.py` and `llm_ptq/run.py` import quantization
primitives from `cnn_init_rotation/ptq.py`; the sibling layout is load-bearing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments

ImageNet-1k vision sweep (RTN-absmax, QuaRot-RTN, SphereQuant; bits 2/4/6/8):

```bash
cd experiments/imagenet_ptq
python run.py
```

LLM sweep (TinyLlama 1.1B + Phi-1.5):

```bash
cd experiments/llm_ptq
python run.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bits 2 4 6 8
python final_summary.py
```

CIFAR-10 toy / coordinate-distribution validation:

```bash
cd experiments/cnn_init_rotation
python run.py
```

CIFAR-10 is downloaded on first run into `cnn_init_rotation/data/`
(gitignored).

## Results

Per-sweep `results.jsonl` files are tracked. Per-sample raw score
arrays (`results/scores/`) and run logs (`*.log`) are gitignored as
regenerable.
