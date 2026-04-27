#!/bin/bash
#SBATCH --job-name=h2_llama8b
#SBATCH --output=logs/h2_llama8b_%j.out
#SBATCH --error=logs/h2_llama8b_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu           # adjust to your Hendrix partition
#SBATCH --gres=gpu:a100:1         # single A100 is enough for 8B
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1

# ---------------------------------------------------------------------------
# H2 sweep on Llama 3 8B. Single A100 80GB is sufficient.
# Wall-clock target: 3-5 hours for WikiText-2 perplexity + lm-eval.
# Edit the top of this script for your Hendrix account and partition.
# ---------------------------------------------------------------------------

mkdir -p logs

# Activate the conda env you created per the README.
module load miniconda/latest    # or equivalent on Hendrix
source activate h2quant

# Make sure HF cache lives on scratch, not home (HF models can be huge).
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

# Route PyTorch temp allocations to scratch to avoid /tmp fills.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diagnostic info in the log.
echo "=== SLURM env ==="
echo "job_id=$SLURM_JOB_ID  host=$(hostname)  date=$(date)"
nvidia-smi
echo "=== Python env ==="
which python
python --version
echo "================="

cd ~/scratch/turboquant-rs/experiments/llm_ptq/hendrix

# Full sweep with lm-eval tasks.
srun python -u run_llama.py \
    --model meta-llama/Meta-Llama-3-8B \
    --bits 2 4 6 8 \
    --methods rtn_absmax quarot h2 \
    --lm-eval \
    --rotation-seed 0

echo "=== Done $(date) ==="
