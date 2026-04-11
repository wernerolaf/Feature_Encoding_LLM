#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --partition=short
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=train_gen
#SBATCH --array=0-11
#SBATCH --exclude=dgx-2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
# Assume sbatch was launched from Feature_Encoding_LLM.
FE_ROOT="${SUBMIT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${FE_ROOT}"

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
  echo "[train_autoencoders] slurm_env.sh not found; continuing without extra cache setup." >&2
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

# --------- Config (override via env) ---------
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATA_PATH="${DATA_PATH:-data/LLM_mini.csv}"
SHEET_NAME="${SHEET_NAME:-}"  # only used for .xlsx
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/autoencoders}"
LATENT_DIMS=(${LATENT_DIMS_OVERRIDE:-2048 4096 8192 16384})
L1_GRID=(${L1_GRID_OVERRIDE:-3e-5 1e-4 3e-4})
LR_GRID=(${LR_GRID_OVERRIDE:-1e-4 2e-4})
LAYERS=(${LAYERS_OVERRIDE:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31})
EPOCHS="${EPOCHS:-10}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-8192}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DTYPE="${DTYPE:-bfloat16}"
INPUT_NORM="${INPUT_NORM:-rmsnorm}"
NORM_EPS="${NORM_EPS:-1e-5}"
L0_THRESHOLD="${L0_THRESHOLD:-1e-6}"

# Use the SLURM array index to create distinct versions/seeds.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VERSION="$((TASK_ID + 1))"
SEED="${SEED_OVERRIDE:-$TASK_ID}"

NUM_LATENT=${#LATENT_DIMS[@]}
NUM_L1=${#L1_GRID[@]}
NUM_LR=${#LR_GRID[@]}
COMBO_PER_LATENT=$((NUM_L1 * NUM_LR))
TOTAL_COMBOS=$((NUM_LATENT * COMBO_PER_LATENT))

if [ "${TASK_ID}" -ge "${TOTAL_COMBOS}" ]; then
  echo "[train_autoencoders] SLURM_ARRAY_TASK_ID=${TASK_ID} exceeds combos=${TOTAL_COMBOS}" >&2
  exit 1
fi

LATENT_IDX=$((TASK_ID / COMBO_PER_LATENT))
REM=$((TASK_ID % COMBO_PER_LATENT))
L1_IDX=$((REM / NUM_LR))
LR_IDX=$((REM % NUM_LR))

LATENT_DIM="${LATENT_DIMS[$LATENT_IDX]}"
BETA="${L1_GRID[$L1_IDX]}"
LR="${LR_GRID[$LR_IDX]}"
ARTIFACT_LABEL="latent${LATENT_DIM}_beta${BETA}_lr${LR}_norm${INPUT_NORM}"

START_TIME=$(date +%s)
echo "[train_autoencoders] model=${MODEL_NAME} data=${DATA_PATH} layers=${LAYERS[*]} latent_dim=${LATENT_DIM} beta=${BETA} lr=${LR} version=${VERSION} seed=${SEED} norm=${INPUT_NORM}"

EXTRA_SHEET_ARGS=()
if [ -n "${SHEET_NAME}" ]; then
  EXTRA_SHEET_ARGS+=(--sheet "${SHEET_NAME}")
fi

for L in "${LAYERS[@]}"; do
  echo "[train_autoencoders] layer=${L}"
  (cd "${FE_ROOT}" && python -m scripts.prepare_autoencoders \
    --model-name "${MODEL_NAME}" \
    --data-path "${DATA_PATH}" \
    "${EXTRA_SHEET_ARGS[@]}" \
    --layers "${L}" \
    --latent-dim "${LATENT_DIM}" \
    --epochs "${EPOCHS}" \
    --ae-batch-size "${AE_BATCH_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype "${DTYPE}" \
    --beta "${BETA}" \
    --lr "${LR}" \
    --input-norm "${INPUT_NORM}" \
    --norm-eps "${NORM_EPS}" \
    --l0-threshold "${L0_THRESHOLD}" \
    --tqdm \
    --seed "${SEED}" \
    --artifact-label "${ARTIFACT_LABEL}" \
    --version "${VERSION}" \
    --output-dir "${OUTPUT_DIR}")
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[train_autoencoders] total_runtime_seconds=${ELAPSED}"
