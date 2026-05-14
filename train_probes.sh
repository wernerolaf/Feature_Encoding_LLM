#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180gb
#SBATCH --partition=short
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train_probes
#SBATCH --array=0-207

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
# Assume sbatch was launched from Feature_Encoding_LLM.
FE_ROOT="${SUBMIT_DIR:-$SCRIPT_DIR}"

# Use shared scratch (via slurm_env.sh) for caches and EVAFS for outputs if configured.
ENV_FILE=""
if [ -f "${SUBMIT_DIR}/slurm_env.sh" ]; then
  ENV_FILE="${SUBMIT_DIR}/slurm_env.sh"
elif [ -f "${SCRIPT_DIR}/slurm_env.sh" ]; then
  ENV_FILE="${SCRIPT_DIR}/slurm_env.sh"
fi

if [ -n "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "[train_probes] slurm_env.sh not found; continuing without extra cache setup." >&2
fi

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" || -n "${HF_TOKEN:-}" ]]; then
  TOKEN_VAL="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
  echo "Logging into Hugging Face via token from env..."
  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "${TOKEN_VAL}" || true
  else
    huggingface-cli login --token "${TOKEN_VAL}" --add-to-git-credential || true
  fi
else
  echo "Hugging Face token not found in env; continuing without login."
fi

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATA="${DATA:-data/LLM_mini.csv}"
LABELS=(${LABELS_OVERRIDE:-i we female prosocial differ clout polite negate shehe risk gender level trait belief question type pronoun answered})
LAYERS=(${LAYERS_OVERRIDE:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31})
AE_ROOT="${AE_ROOT:-artifacts/autoencoders}"
PROBE_ROOT="${PROBE_ROOT:-artifacts/probes}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
DTYPE="${DTYPE:-bfloat16}"
LOGISTIC_PENALTY="${LOGISTIC_PENALTY:-l2}"
LOGISTIC_C="${LOGISTIC_C_OVERRIDE:-1.0}"
LOGISTIC_MAX_ITER="${LOGISTIC_MAX_ITER:-100}"
VAL_SIZE="${VAL_SIZE:-0.1}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-0}"
LABEL_CHUNK_SIZE="${LABEL_CHUNK_SIZE:-0}"
LAYER_CHUNK_SIZE="${LAYER_CHUNK_SIZE:-2}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
VERSION="${VERSION_OVERRIDE:-${TASK_ID:-}}"
START_TIME=$(date +%s)

LABEL_COUNT=${#LABELS[@]}
if [ "${LABEL_CHUNK_SIZE}" -gt 0 ]; then
  LABEL_CHUNK_COUNT=$(( (LABEL_COUNT + LABEL_CHUNK_SIZE - 1) / LABEL_CHUNK_SIZE ))
else
  LABEL_CHUNK_COUNT=1
fi

LAYER_COUNT=${#LAYERS[@]}
if [ "${LAYER_CHUNK_SIZE}" -gt 0 ]; then
  LAYER_CHUNK_COUNT=$(( (LAYER_COUNT + LAYER_CHUNK_SIZE - 1) / LAYER_CHUNK_SIZE ))
else
  LAYER_CHUNK_COUNT=1
fi

FIRST_LAYER="${LAYERS[0]}"
AE_LIST_FIRST=("baseline")
AE_FOUND_FIRST=$(cd "${FE_ROOT}" && PYTHONPATH=$(pwd) python -m scripts.list_autoencoder_artifacts \
  --model-name "$MODEL" \
  --layer "$FIRST_LAYER" \
  --output-dir "$AE_ROOT")
if [ -n "${AE_FOUND_FIRST}" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && AE_LIST_FIRST+=("$line")
  done <<< "${AE_FOUND_FIRST}"
fi
AE_COUNT=${#AE_LIST_FIRST[@]}

TOTAL_JOBS=$(( AE_COUNT * LAYER_CHUNK_COUNT * LABEL_CHUNK_COUNT ))
echo "[train_probes] total_jobs=${TOTAL_JOBS} (AE=${AE_COUNT}, layer_chunks=${LAYER_CHUNK_COUNT}, label_chunks=${LABEL_CHUNK_COUNT})"
if [ -n "${TASK_ID}" ]; then
  if [ "${TASK_ID}" -ge "${TOTAL_JOBS}" ]; then
    echo "[train_probes] SLURM_ARRAY_TASK_ID=${TASK_ID} exceeds variants=${TOTAL_JOBS} (AE=${AE_COUNT}, layer_chunks=${LAYER_CHUNK_COUNT}, label_chunks=${LABEL_CHUNK_COUNT})" >&2
    exit 1
  fi
  AE_IDX=$((TASK_ID / (LAYER_CHUNK_COUNT * LABEL_CHUNK_COUNT)))
  REM=$((TASK_ID % (LAYER_CHUNK_COUNT * LABEL_CHUNK_COUNT)))
  LAYER_CHUNK_IDX=$((REM / LABEL_CHUNK_COUNT))
  LABEL_CHUNK_IDX=$((REM % LABEL_CHUNK_COUNT))
else
  AE_IDX=0
  LAYER_CHUNK_IDX=0
  LABEL_CHUNK_IDX=0
fi

if [ "${LAYER_CHUNK_COUNT}" -gt 1 ]; then
  L_START=$((LAYER_CHUNK_IDX * LAYER_CHUNK_SIZE))
  L_REM=$((LAYER_COUNT - L_START))
  if [ "${L_REM}" -lt "${LAYER_CHUNK_SIZE}" ]; then
    L_COUNT="${L_REM}"
  else
    L_COUNT="${LAYER_CHUNK_SIZE}"
  fi
  LAYER_SUB=("${LAYERS[@]:${L_START}:${L_COUNT}}")
else
  LAYER_SUB=("${LAYERS[@]}")
fi

if [ "${LABEL_CHUNK_COUNT}" -gt 1 ]; then
  START=$((LABEL_CHUNK_IDX * LABEL_CHUNK_SIZE))
  REM=$((LABEL_COUNT - START))
  if [ "${REM}" -lt "${LABEL_CHUNK_SIZE}" ]; then
    COUNT="${REM}"
  else
    COUNT="${LABEL_CHUNK_SIZE}"
  fi
  LABEL_SUB=("${LABELS[@]:${START}:${COUNT}}")
else
  LABEL_SUB=("${LABELS[@]}")
fi

AE_SELECTED="${AE_LIST_FIRST[$AE_IDX]}"

if [ "${AE_SELECTED}" = "baseline" ]; then
  (cd "${FE_ROOT}" && PYTHONPATH=$(pwd) python -m scripts.train_probes \
    --model-name "$MODEL" \
    --data-path "$DATA" \
    --label-columns "${LABEL_SUB[@]}" \
    --layers "${LAYER_SUB[@]}" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    --standardizer identity \
    --probe-type linear \
    --logistic-penalty "$LOGISTIC_PENALTY" \
    --logistic-C "$LOGISTIC_C" \
    --logistic-max-iter "$LOGISTIC_MAX_ITER" \
    --tqdm \
    --val-size "$VAL_SIZE" \
    --test-size "$TEST_SIZE" \
    --random-state "$RANDOM_STATE" \
    ${VERSION:+--version "$VERSION"} \
    --artifact-root "$PROBE_ROOT")
else
  AE_LABEL=$(basename "$(dirname "$(dirname "${AE_SELECTED}")")")
  (cd "${FE_ROOT}" && PYTHONPATH=$(pwd) python -m scripts.train_probes \
    --model-name "$MODEL" \
    --data-path "$DATA" \
    --label-columns "${LABEL_SUB[@]}" \
    --layers "${LAYER_SUB[@]}" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    --standardizer autoencoder \
    --autoencoder-root "$AE_ROOT" \
    --autoencoder-label-filter "$AE_LABEL" \
    --autoencoder-version "latest" \
    --probe-type linear \
    --logistic-penalty "$LOGISTIC_PENALTY" \
    --logistic-C "$LOGISTIC_C" \
    --logistic-max-iter "$LOGISTIC_MAX_ITER" \
    --tqdm \
    --val-size "$VAL_SIZE" \
    --test-size "$TEST_SIZE" \
    --random-state "$RANDOM_STATE" \
    ${VERSION:+--version "$VERSION"} \
    --artifact-root "$PROBE_ROOT")
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[train_probes] total_runtime_seconds=${ELAPSED}"
