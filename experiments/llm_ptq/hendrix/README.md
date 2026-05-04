# Llama 3 8B and 70B PTQ on Hendrix

Full package for running the ApexQuant post-training quantization sweep on Llama 3
8B and Llama 3 70B with WikiText-2 perplexity plus lm-eval-harness zero-shot
tasks. Designed for the Hendrix HPC cluster at DIKU.

## Prerequisites

### 1. HuggingFace access to Llama 3 (gated models)

Llama 3 is gated. You need to:

1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B and accept the
   license. Same for https://huggingface.co/meta-llama/Meta-Llama-3-70B.
2. Get an access token from https://huggingface.co/settings/tokens with
   "Read" scope. Copy the token.
3. On Hendrix: `huggingface-cli login` and paste the token.

Verify access:
```bash
python -c "from transformers import AutoTokenizer; \
  AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')"
```

If this prints no error, you have access. If it says "You have to agree to
the terms and conditions", revisit step 1.

### 2. Conda environment on Hendrix

```bash
# On a login node
module load miniconda/latest  # or equivalent on Hendrix
conda create -n h2quant python=3.11 -y
conda activate h2quant

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 accelerate==0.34.0
pip install datasets==3.0.0 scipy numpy matplotlib
pip install lm-eval==0.4.4   # for zero-shot benchmarks
```

Adjust `cu121` to whatever CUDA version Hendrix uses. Check with `nvidia-smi`
on a GPU node.

### 3. Copy the repository

```bash
cd ~/scratch   # or wherever you keep code on Hendrix
git clone <your turboquant repo url> turboquant-rs
cd turboquant-rs/experiments/llm_ptq/hendrix
```

If the repo is already on your laptop, `rsync -avz` it up to Hendrix.

## What gets run

Three training-free methods at four bit widths, on two Llama 3 sizes, with
two evaluation protocols.

| Method | Description |
|---|---|
| RTN-absmax | Per-channel symmetric uniform round-to-nearest, no rotation |
| QuaRot-RTN | Hadamard rotation + RTN-absmax |
| ApexQuant + Beta (ours) | Rotation + per-row L2 normalize + Beta Lloyd-Max codebook |

| Bit widths | 2, 4, 6, 8 |
| --- | --- |
| Evaluation 1 | WikiText-2 raw test perplexity at seq_len=2048 |
| Evaluation 2 | lm-eval-harness zero-shot: LAMBADA, HellaSwag, PIQA, WinoGrande, ARC-easy, ARC-challenge |

## Resource requirements

### Llama 3 8B

- **Single A100 80GB node** is sufficient
- Weights: 16 GB in fp16
- Working memory for quantization: ~2 GB extra
- KV cache and activations at seq_len=2048: ~4 GB
- **Estimated wall-clock: 3-5 hours for full sweep** (12 quantized evals + 1 FP16 ref + lm-eval)

### Llama 3 70B

- **Multi-GPU required**, aim for 2x A100 80GB or 4x A100 40GB
- Weights: 140 GB in fp16
- Model must be loaded with `device_map="auto"` to shard across GPUs
- Our pipeline keeps a CPU-side backup of original weights for revert between
  methods, which needs a node with >200 GB CPU RAM
- **Estimated wall-clock: 12-20 hours for full sweep**

## How to submit

```bash
cd ~/scratch/turboquant-rs/experiments/llm_ptq/hendrix

# First, edit SLURM scripts to match your Hendrix account / partition / time
# Then submit:

sbatch slurm_8b.sh           # Llama 3 8B on single-GPU
sbatch slurm_70b.sh          # Llama 3 70B on multi-GPU

# Check status:
squeue -u $USER

# Logs go to logs/<job>.out and logs/<job>.err
```

## Output

Results append to `results/llama3_results.jsonl`. Each row is one evaluation:

```json
{
  "model": "meta-llama/Meta-Llama-3-8B",
  "variant": "apexquant",
  "bits": 4,
  "codebook": "beta",
  "perplexity": 5.72,
  "lm_eval": {
    "lambada_openai": 0.7123,
    "hellaswag": 0.7812,
    "piqa": 0.8011,
    "winogrande": 0.7230,
    "arc_easy": 0.7950,
    "arc_challenge": 0.5420
  },
  "size_mb": 4203.2,
  "compression_ratio_vs_fp16": 3.81,
  "rotation_seed": 0
}
```

## After the runs complete

Copy results back to your laptop:

```bash
# from your laptop:
rsync -avz hendrix:~/scratch/turboquant-rs/experiments/llm_ptq/hendrix/results/ \
       ./experiments/llm_ptq/results/
```

Regenerate the paper's Table 5 with the new numbers:

```bash
cd experiments/llm_ptq
python final_summary.py
```

## Optional: calibration-based baselines (GPTQ, AWQ)

For a NeurIPS-strength comparison, also run GPTQ and AWQ on the same models.
See `slurm_gptq.sh` and `slurm_awq.sh`, which wrap the reference
implementations at https://github.com/IST-DASLab/gptq and
https://github.com/mit-han-lab/llm-awq. These take 1-2 additional hours per
model on A100.

## Troubleshooting

**HuggingFace rate limit during download.** Set `export HF_HUB_ETAG_TIMEOUT=60`
and retry.

**OOM on 70B during quantization.** The CPU-side backup of original weights
can exceed 200 GB on some nodes. Either use a bigger memory node, or run the
sweep one method at a time by passing `--methods apexquant` and repeating for each.

**lm-eval-harness version conflict.** If `lm-eval` throws import errors, pin
to version 0.4.4 exactly. Newer versions have breaking API changes.
