#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180gb
#SBATCH --partition=short
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train_layer_adapters
#SBATCH --array=0-31%4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

if [ -f "${SUBMIT_DIR}/scripts/train_debiasing_trl.py" ]; then
  FE_ROOT="$SUBMIT_DIR"
elif [ -f "${SUBMIT_DIR}/Feature_Encoding_LLM/scripts/train_debiasing_trl.py" ]; then
  FE_ROOT="${SUBMIT_DIR}/Feature_Encoding_LLM"
elif [ -f "${SCRIPT_DIR}/scripts/train_debiasing_trl.py" ]; then
  FE_ROOT="$SCRIPT_DIR"
else
  echo "[train_layer_adapters] Could not locate Feature_Encoding_LLM root. Submit from repo root or Feature_Encoding_LLM." >&2
  exit 1
fi

cd "$FE_ROOT"

ENV_FILE=""
if [ -f "${FE_ROOT}/slurm_env.sh" ]; then
  ENV_FILE="${FE_ROOT}/slurm_env.sh"
elif [ -f "${SUBMIT_DIR}/slurm_env.sh" ]; then
  ENV_FILE="${SUBMIT_DIR}/slurm_env.sh"
fi

if [ -n "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "[train_layer_adapters] slurm_env.sh not found; continuing without extra cache setup." >&2
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
SHEET="${SHEET:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/debias/layer_adapters}"

LAYERS_OVERRIDE="${LAYERS_OVERRIDE:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31}"
LEARNING_RATES_OVERRIDE="${LEARNING_RATES_OVERRIDE:-5e-5}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-0}"

MAX_SAMPLES="${MAX_SAMPLES:-all}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EVAL_SIZE="${EVAL_SIZE:-0.2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
NO_EVAL="${NO_EVAL:-0}"

USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ADAPTER_PATH_TEMPLATE="${ADAPTER_PATH_TEMPLATE:-}"

TEXT_COLUMN="${TEXT_COLUMN:-}"
RESPONSE_COLUMN="${RESPONSE_COLUMN:-response}"
GENDER_COLUMN="${GENDER_COLUMN:-gender}"
PRONOUN_COLUMN="${PRONOUN_COLUMN:-pronoun}"
SWAP_PROBABILITY="${SWAP_PROBABILITY:-1.0}"
PROMPT_RESPONSE_SEP="${PROMPT_RESPONSE_SEP:-$'\n'}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a LAYERS <<< "$LAYERS_OVERRIDE"
read -r -a LEARNING_RATES <<< "$LEARNING_RATES_OVERRIDE"
read -r -a SEEDS <<< "$SEEDS_OVERRIDE"

expand_layer_path() {
  local raw="$1"
  local layer_value="$2"
  local placeholder="{layer}"
  local broken_placeholder="{layer"
  raw="${raw//$placeholder/$layer_value}"
  raw="${raw//$broken_placeholder/$layer_value}"
  raw="$(printf '%s' "$raw" | tr -d '{}')"
  printf '%s' "$raw"
}

if [ "${#LAYERS[@]}" -eq 0 ]; then
  echo "[train_layer_adapters] No layers configured." >&2
  exit 1
fi

if [ "${#LEARNING_RATES[@]}" -eq 0 ]; then
  echo "[train_layer_adapters] No learning rates configured." >&2
  exit 1
fi

if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "[train_layer_adapters] No seeds configured." >&2
  exit 1
fi

run_job() {
  local layer="$1"
  local learning_rate="$2"
  local seed="$3"
  local adapter_for_layer="$ADAPTER_PATH"

  if [ -z "$adapter_for_layer" ] && [ -n "$ADAPTER_PATH_TEMPLATE" ]; then
    adapter_for_layer="$(expand_layer_path "$ADAPTER_PATH_TEMPLATE" "$layer")"
  elif [ -n "$adapter_for_layer" ]; then
    adapter_for_layer="$(expand_layer_path "$adapter_for_layer" "$layer")"
  fi

  echo "[train_layer_adapters] layer=${layer} lr=${learning_rate} seed=${seed} use_lora=${USE_LORA} adapter=${adapter_for_layer:-none}"

  local cmd=(
    python -m scripts.train_debiasing_trl
    --model-name "$MODEL"
    --data-path "$DATA"
    --response-column "$RESPONSE_COLUMN"
    --gender-column "$GENDER_COLUMN"
    --pronoun-column "$PRONOUN_COLUMN"
    --swap-probability "$SWAP_PROBABILITY"
    --max-length "$MAX_LENGTH"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --learning-rate "$learning_rate"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-steps "$WARMUP_STEPS"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --max-grad-norm "$MAX_GRAD_NORM"
    --eval-size "$EVAL_SIZE"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --output-dir "$OUTPUT_ROOT"
    --seed "$seed"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --prompt-response-sep "$PROMPT_RESPONSE_SEP"
    --train-layer "$layer"
  )

  if [ -n "$SHEET" ]; then
    cmd+=(--sheet "$SHEET")
  fi

  if [ -n "$TEXT_COLUMN" ]; then
    cmd+=(--text-column "$TEXT_COLUMN")
  fi

  if [ -n "$EVAL_BATCH_SIZE" ]; then
    cmd+=(--eval-batch-size "$EVAL_BATCH_SIZE")
  fi

  if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "all" ]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi

  if [ "$NO_EVAL" = "1" ]; then
    cmd+=(--no-eval)
  fi

  if [ "$USE_LORA" = "1" ]; then
    cmd+=(
      --use-lora
      --lora-r "$LORA_R"
      --lora-alpha "$LORA_ALPHA"
      --lora-dropout "$LORA_DROPOUT"
    )
  else
    cmd+=(--no-peft)
  fi

  if [ "$LOAD_IN_8BIT" = "1" ]; then
    cmd+=(--load-in-8bit)
  fi

  if [ "$LOAD_IN_4BIT" = "1" ]; then
    cmd+=(--load-in-4bit)
  fi

  if [ -n "$adapter_for_layer" ]; then
    cmd+=(--adapter-path "$adapter_for_layer")
  fi

  PYTHONPATH="$FE_ROOT" "${cmd[@]}"
}

declare -a JOB_MATRIX=()
for layer in "${LAYERS[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOB_MATRIX+=("${layer}|${learning_rate}|${seed}")
    done
  done
done

TOTAL_JOBS="${#JOB_MATRIX[@]}"
echo "[train_layer_adapters] total_jobs=${TOTAL_JOBS}"
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ "$SLURM_ARRAY_TASK_MAX" -lt $((TOTAL_JOBS - 1)) ]; then
  echo "[train_layer_adapters] warning: Slurm array max=${SLURM_ARRAY_TASK_MAX}, but total_jobs=${TOTAL_JOBS}. Increase #SBATCH --array to 0-$((TOTAL_JOBS - 1)) to run every job." >&2
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "[train_layer_adapters] SLURM_ARRAY_TASK_ID=${TASK_ID} out of range 0..$((TOTAL_JOBS - 1)); skipping."
    exit 0
  fi
  IFS='|' read -r layer learning_rate seed <<< "${JOB_MATRIX[$TASK_ID]}"
  run_job "$layer" "$learning_rate" "$seed"
else
  for job in "${JOB_MATRIX[@]}"; do
    IFS='|' read -r layer learning_rate seed <<< "$job"
    run_job "$layer" "$learning_rate" "$seed"
  done
fi
