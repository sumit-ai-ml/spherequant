#!/bin/bash
#SBATCH --job-name=sq_llm_8b
#SBATCH --output=logs/sq_llm_8b_%j.out
#SBATCH --error=logs/sq_llm_8b_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=ml4good
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=16
#SBATCH --mem=192GB
#SBATCH --nodes=1
#SBATCH --exclude=hendrixgpu26fl

# ---------------------------------------------------------------------------
# SphereQuant PTQ sweep on 7-8B causal LMs on the ml4good partition (L40s).
# A single L40s 48GB is sufficient for one 7-8B model in FP16 plus the
# transient dequantization buffer SphereQuant uses during quantization.
# Wall-clock target: ~6-10 hours per model for WikiText-2 perplexity, plus
# another 4-8 hours per model if --lm-eval is enabled.
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs results

hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Activate the venv created with: python -m venv spherequant && pip install -e .
# Adjust the path below if your venv lives elsewhere.
VENV=${VENV:-$HOME/spherequant/spherequant}
source "$VENV/bin/activate"

# HF cache on $HOME so weights persist across jobs (use $SCRATCH if your
# cluster gives you scratch with more headroom -- 7-8B weights are ~15GB).
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"

# Avoid /tmp fills with PyTorch's intermediate allocations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SLURM env ==="
echo "job_id=$SLURM_JOB_ID  host=$(hostname)  date=$(date)"
which python
python --version
echo "================="

# Models to sweep. Override at submission with:
#   sbatch --export=ALL,MODELS="meta-llama/Meta-Llama-3-8B" slurm_8b_ml4good.sh
# Mistral-7B-v0.1 and Qwen2.5-7B are open-weight (no HF token needed).
# Llama-2-7b-hf and Meta-Llama-3-8B require `huggingface-cli login` first.
MODELS=${MODELS:-"mistralai/Mistral-7B-v0.1 Qwen/Qwen2.5-7B"}
BITS=${BITS:-"2 4 6 8"}
METHODS=${METHODS:-"rtn_absmax quarot spherequant"}
ROTATION_SEED=${ROTATION_SEED:-0}
LM_EVAL_FLAG=${LM_EVAL_FLAG:-""}    # set to "--lm-eval" to add zero-shot tasks

# run_llama.py lives next to this script.
cd "$(dirname "$0")"

for MODEL in $MODELS; do
    echo "==================================================================="
    echo "Model:          $MODEL"
    echo "Bits:           $BITS"
    echo "Methods:        $METHODS"
    echo "Rotation seed:  $ROTATION_SEED"
    echo "lm-eval:        ${LM_EVAL_FLAG:-disabled}"
    echo "Started:        $(date)"
    echo "==================================================================="

    srun python -u run_llama.py \
        --model "$MODEL" \
        --bits $BITS \
        --methods $METHODS \
        --rotation-seed "$ROTATION_SEED" \
        $LM_EVAL_FLAG
done

echo "=== Done $(date) ==="
