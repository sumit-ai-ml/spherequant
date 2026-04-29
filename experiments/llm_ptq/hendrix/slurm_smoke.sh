#!/bin/bash
#SBATCH --job-name=sq_smoke
#SBATCH --chdir=/home/pds981/spherequant
#SBATCH --output=/home/pds981/spherequant/logs/sq_smoke_%j.out
#SBATCH --error=/home/pds981/spherequant/logs/sq_smoke_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=ml4good
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --exclude=hendrixgpu26fl

# ---------------------------------------------------------------------------
# 30-45 minute smoke test of the SphereQuant LLM PTQ pipeline.
# Runs TinyLlama 1.1B at 4-bit with SphereQuant only — verifies the full
# load -> quantize -> evaluate -> JSONL-write path before you submit a real
# 7-8B sweep. No HF auth needed (TinyLlama is open-weight).
#
# Resource ask is intentionally tiny so SLURM schedules it fast on a busy
# partition.
# ---------------------------------------------------------------------------

set -euo pipefail

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Activate the venv. Same default as slurm_8b_ml4good.sh.
VENV=${VENV:-$HOME/spherequant/spherequant}
source "$VENV/bin/activate"

# HF cache on $HOME so weights persist across jobs.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SLURM env ==="
echo "job_id=$SLURM_JOB_ID  host=$(hostname)  date=$(date)"
which python
python --version
echo "================="

RUN_LLAMA=/home/pds981/spherequant/experiments/llm_ptq/hendrix/run_llama.py

# Small, open-weight, fast. Single bit, single method. Just confirm the
# pipeline is wired up end-to-end.
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
BITS=${BITS:-"4"}
METHODS=${METHODS:-"spherequant"}

echo "==================================================================="
echo "SMOKE TEST"
echo "Model:    $MODEL"
echo "Bits:     $BITS"
echo "Methods:  $METHODS"
echo "Started:  $(date)"
echo "==================================================================="

srun python -u "$RUN_LLAMA" \
    --model "$MODEL" \
    --bits $BITS \
    --methods $METHODS \
    --rotation-seed 0

echo "=== Smoke test done $(date) ==="
echo "If you see a JSONL row appended in"
echo "  experiments/llm_ptq/hendrix/results/llama3_results.jsonl"
echo "the pipeline is working; proceed to the real 7-8B sweep with"
echo "  sbatch experiments/llm_ptq/hendrix/slurm_8b_ml4good.sh"
