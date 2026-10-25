#!/bin/bash
#SBATCH --job-name=genrm
#SBATCH --output=logs/genrm_%j.out
#SBATCH --error=logs/genrm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=GPU_PARTITION_H200

# One missing-cell ckpt: (1) gen_chunked iterative 8192->32768 plain rollouts, (2) 3-layer remeasure.
#   sbatch --export=ALL,TAG=opsd4b_25,MODEL=<ckpt>,[LORA_PATH=<adapter>,]BASE_STRIDE4=...,BASE_TF=...,\
#          BASE_TT=...,BASE_TOK=...,MEASURE_LAYERS=8_20_32  gen_remeasure_slurm.sh
set -euo pipefail

TAG=${TAG:?}; MODEL=${MODEL:?}; TOKENIZER=${TOKENIZER:-$MODEL}; LORA_PATH=${LORA_PATH:-}
BASE_STRIDE4=${BASE_STRIDE4:?}; BASE_TF=${BASE_TF:?}; BASE_TT=${BASE_TT:?}; BASE_TOK=${BASE_TOK:?}
MEASURE_LAYERS=${MEASURE_LAYERS:?}
MAX_CAP=${MAX_CAP:-32768}; STEP=${STEP:-8192}; N_ROLLOUT=${N_ROLLOUT:-16}
FORWARD_BATCH=${FORWARD_BATCH:-4}; GPU_MEM=${GPU_MEM:-0.92}

HF_CACHE_DIR=${HF_CACHE_DIR:-"/path/to/hf_cache"}
WORKHDD=/path/to/workdir
EB=$WORKHDD/eval_basemodel
SIF=${SIF:-"/path/to/repo/analysis.sif"}
_SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"; PROJECT_ROOT="${_SD}"
[[ -f "${SLURM_SUBMIT_DIR:-}/scripts/01_generate.py" ]] && PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_CACHE_DIR}"; mkdir -p logs

GENJSON=$EB/eval_basemodel/${TAG}/results_thinking_false.json
LEAKOUT=$EB/leakage_3layer/${TAG}
DUMMY_LAST8=/tmp/nope_$$

BIND="--bind ${PROJECT_ROOT}:/app --bind ${HF_CACHE_DIR}:/hf_cache --bind ${WORKHDD}:${WORKHDD}"
ENVS="--env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "=== [1/2] gen_chunked ${TAG} (model=${MODEL} lora=${LORA_PATH:-none}) ==="
# shellcheck disable=SC2086
apptainer exec --nv $BIND $ENVS "${SIF}" python /app/scripts/01_generate.py \
    --model "${MODEL}" --tokenizer "${TOKENIZER}" ${LORA_PATH:+--lora_path ${LORA_PATH}} \
    --parquet /app/data/test.parquet --out "${GENJSON}" \
    --n_rollout "${N_ROLLOUT}" --step "${STEP}" --max_cap "${MAX_CAP}" --gpu_mem "${GPU_MEM}"

echo "=== [2/2] remeasure 3-layer ${TAG} ==="
# shellcheck disable=SC2086
apptainer exec --nv $BIND $ENVS --env MEASURE_LAYERS="${MEASURE_LAYERS}" "${SIF}" \
    python /app/scripts/04_drift.py --stage all \
    --post_ckpt "${MODEL}" --post_val "${GENJSON}" --out_dir "${LEAKOUT}" \
    --forward_batch "${FORWARD_BATCH}" ${LORA_PATH:+--lora_path ${LORA_PATH}} \
    --base_last8 "${DUMMY_LAST8}" --base_stride4 "${BASE_STRIDE4}" \
    --base_tf_rollouts "${BASE_TF}" --base_tt_rollouts "${BASE_TT}" --base_tokenizer "${BASE_TOK}"
echo "DONE genrm ${TAG}"
