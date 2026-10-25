#!/bin/bash
#SBATCH --job-name=depthprof
#SBATCH --output=logs/depthprof_%j.out
#SBATCH --error=logs/depthprof_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=GPU_PARTITION_H200

# Per-layer pi profiles for the bare / behavioural halves of the depth decomposition.
#
#   pi_L(behav) - pi_L(bare)     text effect     (the intervention changed what was written)
#   pi_L(hooked) - pi_L(behav)   injection effect (mechanical propagation of the perturbation)
#
# The hooked half already exists (the nadd runs dumped hidden states with the hook active). This
# job produces the other two, both with --frac 0 so NO hook is registered:
#   BARE=1  convert a bare generation to rollouts.json, then reforward it
#   else    reforward an existing intervened rollouts.json without the hook
#
# Reforward only -- nothing is generated. Model staged to /dev/shm once per checkpoint.
#   sbatch --export=ALL,SPECS="cell|model|tokz|analysis|src|out ..." depth_profile_slurm.sh
set -euo pipefail

SPECS=${SPECS:?set SPECS (newline- or space-separated  tag|model|tokenizer|analysis|src|outdir)}
SD_SIF=${SD_SIF:-"/path/to/repo/analysis.sif"}
HF_CACHE_DIR=${HF_CACHE_DIR:-"/path/to/hf_cache"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/path/to/workdir/eval_basemodel"}
WORKHDD_DIR=${WORKHDD_DIR:-"/path/to/workdir"}
FORWARD_BATCH=${FORWARD_BATCH:-1}
_SD="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"; PROJECT_ROOT="${_SD}"
[[ -f "${SLURM_SUBMIT_DIR:-}/scripts/08_measure_pi.py" ]] && PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p logs

run() {  # run <extra-binds...> -- <python args...>
    apptainer exec --nv \
        --bind "${PROJECT_ROOT}:/app" --bind "${HF_CACHE_DIR}:/hf_cache" \
        --bind "${OUTPUT_ROOT}:/output" --bind "${WORKHDD_DIR}:${WORKHDD_DIR}" ${STAGE_BIND:-} \
        --env HF_HOME=/hf_cache --env HF_HUB_OFFLINE=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=/app \
        "${SD_SIF}" "$@"
}

STAGED=""; STAGE_HOST=""; STAGE_BIND=""
stage() {  # stage <model-dir>  -> sets RUN_MODEL (+ STAGE_BIND) and remembers what is staged
    local M="$1"
    if [ "${M}" = "${STAGED}" ]; then return; fi
    [ -n "${STAGE_HOST}" ] && rm -rf "${STAGE_HOST}"
    STAGE_HOST=""; STAGE_BIND=""; RUN_MODEL="${M}"; STAGED="${M}"
    # bare `return` propagates the failing test's status, which under set -e kills the job. An HF
    # hub id (the steer arm's base models) is not a directory and must simply skip staging.
    [ -d "${M}" ] || return 0
    local need avail
    need=$(du -sb "${M}" | cut -f1); avail=$(df -B1 --output=avail /dev/shm | tail -1)
    if [ "${avail}" -gt $((need + 15 * 1024 * 1024 * 1024)) ]; then
        STAGE_HOST="/dev/shm/dp_${SLURM_JOB_ID:-$$}"
        rm -rf "${STAGE_HOST}"; mkdir -p "${STAGE_HOST}"
        echo "=== staging $(numfmt --to=iec "${need}") -> ${STAGE_HOST} ==="
        time cp -r "${M}/." "${STAGE_HOST}/"
        RUN_MODEL="/staged_model"; STAGE_BIND="--bind ${STAGE_HOST}:/staged_model"
    fi
}
trap 'rm -rf "${STAGE_HOST:-/nonexistent}"' EXIT INT TERM

for SPEC in ${SPECS}; do
    # 7th field = frac. 0 (default) registers no hook -> the bare / behavioural halves.
    # non-zero re-applies the original injection -> the hooked half, whose L>inject layers carry
    # the generation-time state (a --frac 0 pass would leave those layers un-injected).
    IFS='|' read -r TAG MODEL TOKZ ANALYSIS SRC OUTD FRAC <<< "${SPEC}"
    FRAC=${FRAC:-0}
    OUTJ="${OUTD}/pi_depth.json"
    if [ -f "${OUTJ}" ]; then echo "SKIP ${TAG}: already done"; continue; fi
    mkdir -p "${OUTD}"

    ROLL="${SRC}"
    case "${SRC}" in
        *results_thinking_false.json|*val_details/*)      # nested bare schema -> rollouts.json
            ROLL="${OUTD}/rollouts.json"
            if [ ! -f "${ROLL}" ]; then
                echo "=== [${TAG}] convert bare generation -> rollouts.json ==="
                run python /app/scripts/bare_to_rollouts.py --src "${SRC}" --out "${ROLL}" --tokenizer "${TOKZ}"
            fi ;;
    esac

    stage "${MODEL}"
    echo "=== [${TAG}] per-layer pi, frac=${FRAC} (hook $([ "${FRAC}" = "0" ] && echo OFF || echo ON))  model=${MODEL} ==="
    run python /app/scripts/08_measure_pi.py \
        --mode additive --frac "${FRAC}" --rollouts_json "${ROLL}" --model "${RUN_MODEL}" \
        --tokenizer "${TOKZ}" --base_dir "${ANALYSIS}" --layer 20 \
        --parquet /app/data/test.parquet --n_per_q 4 --forward_batch "${FORWARD_BATCH}" \
        --out_json "${OUTJ}" --dump_hidden "${OUTD}/hidden_pooled_depth.h5"
    echo "=== [${TAG}] done ==="
done
echo "ALL DONE"
