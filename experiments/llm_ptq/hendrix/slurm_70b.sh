#!/bin/bash
#SBATCH --job-name=h2_llama70b
#SBATCH --output=logs/h2_llama70b_%j.out
#SBATCH --error=logs/h2_llama70b_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu            # adjust to your Hendrix partition
#SBATCH --gres=gpu:a100:4          # 4 A100s for Llama 3 70B (140 GB in fp16)
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G                 # CPU snapshot of 70B weights is ~140 GB
#SBATCH --nodes=1

# ---------------------------------------------------------------------------
# H2 sweep on Llama 3 70B. Requires multi-GPU sharding via device_map="auto".
# 4x A100 80GB recommended. 8x A100 40GB also works.
# Wall-clock target: 12-20 hours for WikiText-2 perplexity + lm-eval.
# ---------------------------------------------------------------------------

mkdir -p logs

module load miniconda/latest     # or equivalent on Hendrix
source activate h2quant

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SLURM env ==="
echo "job_id=$SLURM_JOB_ID  host=$(hostname)  date=$(date)"
nvidia-smi
echo "=== Python env ==="
which python
python --version
free -h
echo "================="

cd ~/scratch/turboquant-rs/experiments/llm_ptq/hendrix

# Start with the 4-bit setting as a canary. If this completes and the
# perplexity is sensible (< 10 on WikiText-2), the full sweep should work.
# You can comment this out and run the full sweep directly if you are
# confident in the setup.
echo ">>> Canary: Llama 3 70B at 4-bit only, all three methods, no lm-eval"
srun python -u run_llama.py \
    --model meta-llama/Meta-Llama-3-70B \
    --bits 4 \
    --methods rtn_absmax quarot h2 \
    --rotation-seed 0

echo ">>> Full sweep: all bits, all methods, with lm-eval tasks"
srun python -u run_llama.py \
    --model meta-llama/Meta-Llama-3-70B \
    --bits 2 4 6 8 \
    --methods rtn_absmax quarot h2 \
    --lm-eval \
    --rotation-seed 0

echo "=== Done $(date) ==="
