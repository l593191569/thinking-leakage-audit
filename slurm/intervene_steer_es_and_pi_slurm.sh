#!/bin/bash
#SBATCH --job-name=steerpi
#SBATCH --output=logs/steerpi_%j.out
#SBATCH --error=logs/steerpi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=GPU_PARTITION_H200

# EasySteer activation-steering intervention (chunked generation + loop filtering) PLUS a third stage: HF teacher-forced reforward (scripts/08_measure_pi.py,
# analysis.sif) to measure the REAL achieved pi at the SAME layer/frac -- not a separate job, same
# allocation, vLLM engine freed before HF loads. ANALYSIS is now REQUIRED (needed by stage 3 even
# when the steer vector already exists and stage-1's auto-export doesn't run).
#   steer (base model, FRAC = alpha):
#   sbatch --export=ALL,ARM=induce,FRAC=1.0,INTERVENE_SUBDIR=intervene_4b,\
#     INDUCE_MODEL=Qwen/Qwen3-4B,BASE_TOKENIZER=Qwen/Qwen3-4B,ANALYSIS=<base_dir>/analysis \
#     slurm/intervene_steer_es_and_pi_slurm.sh
#   counter-steer (checkpoint, FRAC = -drift/||v_leak||, see configs/cells.yaml):
#   sbatch --export=ALL,ARM=ablate,FRAC=-0.6017,MODEL_OVERRIDE=<local ckpt dir>,... slurm/intervene_steer_es_and_pi_slurm.sh
set -euo pipefail

ARM=${ARM:-induce}
DIR=${DIR:-tmodeTk}             # v_leak; the only exported direction
FRAC=${FRAC:?set FRAC}
LAYERS=${LAYERS:-20}            # OUR-convention hidden_states layer
STEP=${STEP:-200}              # post ckpt for ablate
N_ROLLOUT=${N_ROLLOUT:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
GPU_MEM=${GPU_MEM:-0.90}

SIF=${SIF:-"/path/to/workdir/containers/easysteer.sif"}
SD_SIF=${SD_SIF:-"/path/to/repo/analysis.sif"}   # for scoring (analysis container)
HF_CACHE_DIR=${HF_CACHE_DIR:-"/path/to/hf_cache"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/path/to/workdir/eval_basemodel"}
WORKHDD_DIR=${WORKHDD_DIR:-"/path/to/workdir"}
CKPT_ROOT=${CKPT_ROOT:-"/path/to/workdir/grpo_ckpts/grpo-Qwen-Qwen3-8B-dapo_math-standard"}

_SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"; PROJECT_ROOT="${_SD}"
[[ -f "${SLURM_SUBMIT_DIR:-}/scripts/06_steer.py" ]] && PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p logs

INDUCE_MODEL=${INDUCE_MODEL:-"Qwen/Qwen3-8B"}      # base model for induce arm (override for 4B etc.)
BASE_TOKENIZER=${BASE_TOKENIZER:-"Qwen/Qwen3-8B"}  # prompt tokenizer (Qwen3 family shares one)
INTERVENE_SUBDIR=${INTERVENE_SUBDIR:-intervene}    # output/vector subdir (e.g. intervene_4b)
if [ -n "${MODEL_OVERRIDE:-}" ]; then
    # SFT / OPSD / merged-LoRA checkpoints do not live under the verl global_step_N layout, so
    # let the caller pass the directory straight through.
    MODEL="${MODEL_OVERRIDE}"
elif [ "${ARM}" = "ablate" ]; then
    # HF_SUBDIR=huggingface_bf16 selects the bf16 copy made by convert_bf16.py (half the bytes,
    # half the load time; the fp32 original is what verl exports and is kept untouched).
    MODEL="${CKPT_ROOT}/global_step_${STEP}/actor/${HF_SUBDIR:-huggingface}"
else
    MODEL="${INDUCE_MODEL}"
fi

# vector file: tag = L<layers, '_' -> nothing>
LTAG="L$(echo "${LAYERS}" | tr ',' '_')"
VECNAME="${DIR}"
VEC="/output/${INTERVENE_SUBDIR}/vectors/${VECNAME}_${LTAG}.pt"

# auto-export the steer vector on the compute node if missing (login-node FS may be flaky for
# the stride4 selections.h5). Needs ANALYSIS (the *_stride4/analysis dir holding directions).
VEC_HOST="${OUTPUT_ROOT}/${INTERVENE_SUBDIR}/vectors/${VECNAME}_${LTAG}.pt"
ANALYSIS=${ANALYSIS:?set ANALYSIS (needed by stage 3 / pi measurement even if the vector already exists)}
if [ ! -f "${VEC_HOST}" ]; then
    [ -n "${ANALYSIS}" ] || { echo "VEC missing and no ANALYSIS given to export from: ${VEC_HOST}"; exit 1; }
    echo "=== export steer vectors L=${LAYERS} from ${ANALYSIS} (analysis.sif) ==="
    apptainer exec --nv \
        --bind "${PROJECT_ROOT}:/app" --bind "${HF_CACHE_DIR}:/hf_cache" \
        --bind "${OUTPUT_ROOT}:/output" --bind "${WORKHDD_DIR}:${WORKHDD_DIR}" \
        --env HF_HOME=/hf_cache --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app \
        "${SD_SIF}" python /app/scripts/05_export_steer_vectors.py \
        --analysis "${ANALYSIS}" --layers "${LAYERS}" --out "/output/${INTERVENE_SUBDIR}/vectors"
fi

# out tag: frac sign/decimal safe (+layer, so different layers don't collide)
FT=$(echo "${FRAC}" | sed 's/-/m/; s/\./p/')
TAG="${ARM}_${DIR}_${LTAG}_frac${FT}"
# drift-normalised runs use a per-checkpoint FRAC, so the frac-derived name says nothing about
# which checkpoint it is -- let the caller name the cell instead.
[ -n "${TAG_OVERRIDE:-}" ] && TAG="${TAG_OVERRIDE}"
OUT="/output/${INTERVENE_SUBDIR}/${TAG}"

EXTRA=""
[ -n "${DATASETS:-}" ] && EXTRA="${EXTRA} --datasets ${DATASETS}"
[ "${MAX_QUESTIONS:-0}" != "0" ] && EXTRA="${EXTRA} --max_questions ${MAX_QUESTIONS}"

# --- stage the ckpt into node-local RAM (tmpfs) ONCE ---
# The model is read TWICE per job: vLLM in stage 1, HF in stage 3. For the fp32 8B ckpt (31G on
# the shared filesystem) that measured 1728s + 1322s = 51 min, i.e. 46% of a 1:50 job. One sequential copy to
# /dev/shm turns both reads into memory reads. Bound into the containers at a fixed path so we
# don't depend on how apptainer handles the host /dev/shm.
# Only applies when MODEL is a local dir -- the induce arm's MODEL is an HF hub id (Qwen/Qwen3-8B)
# living in the /projects hf_cache, which is fast and shared. STAGE_MODEL=0 disables.
# NB: tmpfs pages are charged to the job's cgroup and are NOT reclaimable, so the staged bytes eat
# into --mem (200G). We only stage if /dev/shm has the size + 15G headroom.
STAGE_MODEL=${STAGE_MODEL:-1}
STAGE_HOST=""; STAGE_BIND=""
if [ "${STAGE_MODEL}" != "0" ] && [ -d "${MODEL}" ]; then
    _need=$(du -sb "${MODEL}" | cut -f1)
    _avail=$(df -B1 --output=avail /dev/shm | tail -1)
    if [ "${_avail}" -gt $((_need + 15 * 1024 * 1024 * 1024)) ]; then
        STAGE_HOST="/dev/shm/ckpt_${SLURM_JOB_ID:-$$}"
        trap 'rm -rf "${STAGE_HOST}"' EXIT INT TERM
        echo "=== staging ckpt $(numfmt --to=iec "${_need}") -> ${STAGE_HOST} (shm avail $(numfmt --to=iec "${_avail}")) ==="
        mkdir -p "${STAGE_HOST}"
        time cp -r "${MODEL}/." "${STAGE_HOST}/"
        MODEL="/staged_model"
        STAGE_BIND="--bind ${STAGE_HOST}:/staged_model"
        echo "=== staged; stages 1 and 3 now read weights from RAM ==="
    else
        echo "=== skip staging: /dev/shm avail $(numfmt --to=iec "${_avail}") < $(numfmt --to=iec "${_need}") + 15G ==="
    fi
fi

echo "steer-es ${TAG}  model=${MODEL}  vec=${VEC}"
# NB: mount the repo at /repo (NOT /app -- easysteer is an editable install under /app/easysteer)
# shellcheck disable=SC2086
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/repo" --bind "${HF_CACHE_DIR}:/hf_cache" \
    --bind "${OUTPUT_ROOT}:/output" --bind "${WORKHDD_DIR}:${WORKHDD_DIR}" ${STAGE_BIND} \
    --env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH=/repo --env BASE_TOKENIZER="${BASE_TOKENIZER}" \
    "${SIF}" /usr/bin/python3 /repo/scripts/06_steer.py \
    --arm "${ARM}" --model "${MODEL}" --dir "${DIR}" --frac "${FRAC}" \
    --vec_path "${VEC}" --layers "${LAYERS}" \
    --parquet /repo/data/test.parquet \
    --out_dir "${OUT}" --n_rollout "${N_ROLLOUT}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --gpu_mem "${GPU_MEM}" ${EXTRA}

echo "=== score accuracy ${TAG} (analysis.sif, has verl) ==="
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/app" --bind "${HF_CACHE_DIR}:/hf_cache" \
    --bind "${OUTPUT_ROOT}:/output" --bind "${WORKHDD_DIR}:${WORKHDD_DIR}" \
    --env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app \
    "${SD_SIF}" python /app/scripts/07_score.py \
    --int_dir "${OUT}" --parquet /app/data/test.parquet --tokenizer "${BASE_TOKENIZER}"

# --- stage 3: measure the REAL achieved pi (analysis.sif, HF). NOT a generation pass -- a single
# teacher-forced forward over the already-generated (prompt+response) tokens with the SAME
# additive hook re-applied at the SAME layer/frac, so it's cheap (no autoregressive decode, no
# OOM risk) even though it uses HF instead of vLLM. vLLM engine from stage 1 is already freed. ---
echo "=== measure_pi ${TAG} (analysis.sif, HF reforward) ==="
# DUMP_HIDDEN=1 -> also dump per-rollout POOLED hidden states at ALL analysis layers to
# ${OUT}/hidden_pooled.h5 (used for 8B only; other models just need the scalar pi). L20 is
# POST-hook, matching the pi in pi_measured.json.
DUMP_HIDDEN=${DUMP_HIDDEN:-}
DUMP_ARG=""
[ -n "${DUMP_HIDDEN}" ] && DUMP_ARG="--dump_hidden ${OUT}/hidden_pooled.h5"
# shellcheck disable=SC2086
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/app" --bind "${HF_CACHE_DIR}:/hf_cache" \
    --bind "${OUTPUT_ROOT}:/output" --bind "${WORKHDD_DIR}:${WORKHDD_DIR}" ${STAGE_BIND} \
    --env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app \
    "${SD_SIF}" python /app/scripts/08_measure_pi.py \
    --mode additive --rollouts_json "${OUT}/rollouts.json" --model "${MODEL}" \
    --tokenizer "${BASE_TOKENIZER}" --base_dir "${ANALYSIS}" --layer "${LAYERS}" --frac "${FRAC}" \
    --parquet /app/data/test.parquet --n_per_q 4 --forward_batch 2 \
    --out_json "${OUT}/pi_measured.json" ${DUMP_ARG}

echo "DONE ${TAG}"
