#!/bin/bash
#SBATCH --job-name=rotation
#SBATCH --output=logs/rotation_%j.out
#SBATCH --error=logs/rotation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=GPU_PARTITION_H200

# Rotation control: thinking-TRUE generation + reforward on a post ckpt, then compare
# re-estimated post directions to the base directions.
#   sbatch --export=ALL,STEP=200 measure_rotation_slurm.sh
#   smoke: sbatch --export=ALL,STEP=200,DATASETS=aime24,MAX_QUESTIONS=2,MAX_TOKENS=16384 ...
set -euo pipefail

STEP=${STEP:-200}
DATASETS=${DATASETS:-""}
MAX_QUESTIONS=${MAX_QUESTIONS:-0}
MAX_TOKENS=${MAX_TOKENS:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-34816}
FORWARD_BATCH=${FORWARD_BATCH:-2}
STAGE=${STAGE:-all}

HF_CACHE_DIR=${HF_CACHE_DIR:-"/path/to/hf_cache"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/path/to/workdir/eval_basemodel"}
WORKHDD_DIR=${WORKHDD_DIR:-"/path/to/workdir"}
CKPT_ROOT=${CKPT_ROOT:-"/path/to/workdir/grpo_ckpts/grpo-Qwen-Qwen3-8B-dapo_math-standard"}
SIF_PATH=${SIF_PATH:-"/path/to/repo/analysis.sif"}

_SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PROJECT_ROOT="${_SD}"; [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/10_rotation.py" ]] && PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_CACHE_DIR}"; export VLLM_USE_V1=1; export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}"; mkdir -p logs

# HF_SUBDIR=huggingface_bf16 halves the bytes read (fp32 8B ckpt is 31G, bf16 15.3G) and both
# vLLM and the reforward cast to bf16 on load anyway.
CKPT="${CKPT_ROOT}/global_step_${STEP}/actor/${HF_SUBDIR:-huggingface}"
OUT="/output/leakage/grpo_step${STEP}/rotation"
TF_H5="/output/leakage/grpo_step${STEP}/post_sol_f.h5"

# --- stage the ckpt into node-local RAM (tmpfs) ONCE -----------------------------------------
# The model is read TWICE per job: vLLM in the gen stage, then HF in the reforward. The
# safetensors loader's access pattern gets ~18 MB/s on this Lustre; one sequential copy runs at
# ~865 MB/s and turns both reads into memory reads (measured ~35 min saved on an 8B ckpt).
# tmpfs pages are charged to the job cgroup and are NOT reclaimable, so require size + 15G under
# --mem. STAGE_MODEL=0 disables.
STAGE_MODEL=${STAGE_MODEL:-1}
STAGE_BIND=""
if [ "${STAGE_MODEL}" != "0" ] && [ -d "${CKPT}" ]; then
    _need=$(du -sb "${CKPT}" | cut -f1)
    _avail=$(df -B1 --output=avail /dev/shm | tail -1)
    if [ "${_avail}" -gt $((_need + 15 * 1024 * 1024 * 1024)) ]; then
        STAGE_HOST="/dev/shm/rot_${SLURM_JOB_ID:-$$}"
        trap 'rm -rf "${STAGE_HOST}"' EXIT INT TERM
        mkdir -p "${STAGE_HOST}"
        echo "=== staging $(numfmt --to=iec "${_need}") -> ${STAGE_HOST} (shm avail $(numfmt --to=iec "${_avail}")) ==="
        time cp -r "${CKPT}/." "${STAGE_HOST}/"
        CKPT="/staged_model"
        STAGE_BIND="--bind ${STAGE_HOST}:/staged_model"
        echo "=== staged; gen and reforward both read weights from RAM ==="
    else
        echo "=== skip staging: /dev/shm avail $(numfmt --to=iec "${_avail}") < $(numfmt --to=iec "${_need}") + 15G ==="
    fi
fi

# INFER_TP>1 shards the weights across GPUs in the vLLM gen stage. KV cache is what limits this
# workload (at max_model_len=34816 a single H200 fits only ~20 concurrent sequences), so tp=2
# buys both KV room and decode bandwidth -- roughly halves the ~6h generation. Remember to ask
# for the GPUs too: sbatch --gres=gpu:h200:2 --export=ALL,INFER_TP=2,...
INFER_TP=${INFER_TP:-1}
INFER_GPU_MEM=${INFER_GPU_MEM:-0.90}

EXEC="apptainer exec --nv \
    --bind ${PROJECT_ROOT}:/app \
    --bind ${HF_CACHE_DIR}:/hf_cache \
    --bind ${OUTPUT_ROOT}:/output \
    --bind ${WORKHDD_DIR}:${WORKHDD_DIR} ${STAGE_BIND} \
    --env HF_HOME=/hf_cache --env VLLM_USE_V1=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app \
    --env INFER_TP=${INFER_TP} --env INFER_GPU_MEM=${INFER_GPU_MEM} \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ${SIF_PATH}"

EXTRA=""
[ -n "${DATASETS}" ] && EXTRA="${EXTRA} --datasets ${DATASETS}"
[ "${MAX_QUESTIONS}" != "0" ] && EXTRA="${EXTRA} --max_questions ${MAX_QUESTIONS}"

echo "rotation: step ${STEP}  ckpt=${CKPT}  out=${OUT}  tp=${INFER_TP}  gpu_mem=${INFER_GPU_MEM}"
# shellcheck disable=SC2086
${EXEC} python /app/scripts/10_rotation.py \
    --stage "${STAGE}" --post_ckpt "${CKPT}" --out_dir "${OUT}" --tf_h5 "${TF_H5}" \
    --max_tokens "${MAX_TOKENS}" --max_model_len "${MAX_MODEL_LEN}" \
    --forward_batch "${FORWARD_BATCH}" ${EXTRA}
