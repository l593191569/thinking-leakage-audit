#!/bin/bash
#SBATCH --job-name=eb3l
#SBATCH --output=logs/eb3l_%j.out
#SBATCH --error=logs/eb3l_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=GPU_PARTITION_H200

# Base-model thinking true/false experiment, 3-layer POOLED hidden states + chunked generation.
#   [1/4] gen_chunked thinking=true  -> results_thinking_true.json   (8192->32768, loop-filtered)
#   [2/4] gen_chunked thinking=false -> results_thinking_false.json
#   [3/4] pooled_reforward  -> analysis/selections.h5  (region means + per-region n_tok, 3 layers)
#   [4/4] analyze --stage analyze --layers <L> -> directions_pooled.npy (incl. token-weighted thinkmodetk)
#   sbatch --export=ALL,TAG=Qwen-Qwen3-32B_3layer,MODEL=Qwen/Qwen3-32B,LAYERS=16,36,56 eval_base3l_slurm.sh
set -euo pipefail

TAG=${TAG:?}; MODEL=${MODEL:?}; LAYERS=${LAYERS:?}        # LAYERS: pass underscore-sep (8_16_24)
LAYERS=${LAYERS//_/,}                                     # -> comma form; sbatch --export splits on commas
TOKENIZER=${TOKENIZER:-$MODEL}
N_ROLLOUT=${N_ROLLOUT:-16}; STEP=${STEP:-8192}; MAX_CAP=${MAX_CAP:-32768}
TP=${TP:-1}                                                # gen tensor-parallel GPUs (submit with --gres=gpu:h200:$TP)
FB_TRUE=${FB_TRUE:-1}; FB_FALSE=${FB_FALSE:-1}; GPU_MEM=${GPU_MEM:-0.90}
MAX_QUESTIONS=${MAX_QUESTIONS:-0}
PARQUET=${PARQUET:-/app/data/test.parquet}   # container path; train set for GRPO-trainset vectors

HF_CACHE_DIR=${HF_CACHE_DIR:-"/path/to/hf_cache"}
WORKHDD=/path/to/workdir
EB=$WORKHDD/eval_basemodel
SIF=${SIF:-"/path/to/repo/analysis.sif"}
_SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"; PROJECT_ROOT="${_SD}"
[[ -f "${SLURM_SUBMIT_DIR:-}/scripts/02_pooled_reforward.py" ]] && PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p logs

OUT=$EB/${TAG}
BIND="--bind ${PROJECT_ROOT}:/app --bind ${HF_CACHE_DIR}:/hf_cache --bind ${WORKHDD}:${WORKHDD}"
ENVS="--env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
MQ=""; [ "${MAX_QUESTIONS}" != "0" ] && MQ="--max_questions ${MAX_QUESTIONS}"

run() { echo "+ $*"; apptainer exec --nv $BIND $ENVS "${SIF}" "$@"; }

# gen: gen_chunked self-skips a complete file and resumes an incomplete one (per-round checkpoints).
for mode in true false; do
  GENJSON=$OUT/results_thinking_${mode}.json
  echo "=== [gen ${mode}] ${TAG} (tp=${TP}) ==="
  # shellcheck disable=SC2086
  run python /app/scripts/01_generate.py --model "${MODEL}" --tokenizer "${TOKENIZER}" \
      --enable_thinking ${mode} --parquet "${PARQUET}" --out "${GENJSON}" --tp "${TP}" \
      --n_rollout "${N_ROLLOUT}" --step "${STEP}" --max_cap "${MAX_CAP}" --gpu_mem "${GPU_MEM}" ${MQ}
done

if [ -f "${OUT}/analysis/selections.h5" ]; then
  echo "=== [skip pooled_reforward] selections.h5 exists ==="
else
  echo "=== [pooled_reforward] ${TAG} layers=${LAYERS} ==="
  # shellcheck disable=SC2086
  run python /app/scripts/02_pooled_reforward.py --model "${MODEL}" --tokenizer "${TOKENIZER}" \
      --eval_dir "${OUT}" --parquet "${PARQUET}" --layers "${LAYERS}" \
      --forward_batch "${FB_TRUE}" ${MQ}
fi

if [ -f "${OUT}/analysis/directions_pooled.npy" ]; then
  echo "=== [skip analyze] directions_pooled.npy exists ==="
else
  echo "=== [analyze directions] ${TAG} layers=${LAYERS} ==="
  # shellcheck disable=SC2086
  run python /app/scripts/03_directions.py --stage analyze \
      --eval_dir "${OUT}" --model "${MODEL}" --layers "${LAYERS}" --marker_family qwen3
fi
echo "DONE eb3l ${TAG} -> ${OUT}/analysis/directions_pooled.npy"
