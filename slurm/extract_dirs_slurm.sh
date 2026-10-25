#!/bin/bash
#SBATCH --job-name=extractdirs
#SBATCH --output=logs/extractdirs_%j.out
#SBATCH --error=logs/extractdirs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=GPU_PARTITION_H200

# Re-extract reasoning directions at mid layers (16,18,20,22,24) from stored base TT/TF rollouts.
# NOT a steering job (reforward/analysis). 16/20/24 also extracted to validate vs stored stride-4.
set -euo pipefail
LAYERS=${LAYERS:-"16,18,20,22,24"}
SD_SIF=${SD_SIF:-"/path/to/repo/analysis.sif"}
HF_CACHE_DIR=${HF_CACHE_DIR:-"/path/to/hf_cache"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/path/to/workdir/eval_basemodel"}
WORKHDD_DIR=${WORKHDD_DIR:-"/path/to/workdir"}
_SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"; PROJECT_ROOT="${_SD}"
[[ -f "${SLURM_SUBMIT_DIR:-}/scripts/03b_directions_more_layers.py" ]] && PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p logs

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/app" --bind "${HF_CACHE_DIR}:/hf_cache" \
    --bind "${OUTPUT_ROOT}:/output" --bind "${WORKHDD_DIR}:${WORKHDD_DIR}" \
    --env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SD_SIF}" python /app/scripts/03b_directions_more_layers.py \
    --layers "${LAYERS}" --src /output/Qwen-Qwen3-8B_stride4 \
    --out_dir /output/Qwen-Qwen3-8B_mid/analysis \
    --parquet /app/data/test.parquet \
    --max_roll_tt 6 --max_roll_tf 8
echo "EXTRACT DONE -> /output/Qwen-Qwen3-8B_mid/analysis"
