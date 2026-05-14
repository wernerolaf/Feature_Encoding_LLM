#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180gb
#SBATCH --partition=short
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=feature_interaction
#SBATCH --array=0-62%8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

if [ -f "${SUBMIT_DIR}/scripts/feature_interaction.py" ]; then
  FE_ROOT="$SUBMIT_DIR"
elif [ -f "${SUBMIT_DIR}/Feature_Encoding_LLM/scripts/feature_interaction.py" ]; then
  FE_ROOT="${SUBMIT_DIR}/Feature_Encoding_LLM"
elif [ -f "${SCRIPT_DIR}/scripts/feature_interaction.py" ]; then
  FE_ROOT="$SCRIPT_DIR"
else
  echo "[feature_interaction] Could not locate Feature_Encoding_LLM root. Submit from repo root or Feature_Encoding_LLM." >&2
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
  echo "[feature_interaction] slurm_env.sh not found; continuing without extra cache setup." >&2
fi

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" || -n "${HF_TOKEN:-}" ]]; then
  TOKEN_VAL="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
  echo "Logging into Hugging Face via token from env..."
  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "${TOKEN_VAL}" || true
  else
    huggingface-cli login --token "${TOKEN_VAL}" --add-to-git-credential || true
  fi
fi

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATA="${DATA:-data/LLM_mini.csv}"
PROBE_DIR="${PROBE_DIR:-artifacts/probes}"
PROBE_VERSION="${PROBE_VERSION:-latest}"
PRIMARY_PROBE_VERSION="${PRIMARY_PROBE_VERSION:-$PROBE_VERSION}"
SECONDARY_PROBE_VERSION="${SECONDARY_PROBE_VERSION:-$PROBE_VERSION}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/experiments/feature_interaction}"
MODEL_FILTER_VALUE="${MODEL_FILTER_VALUE:-Meta-Llama-3-8B}"

PRIMARY_FEATURE="${PRIMARY_FEATURE:-gender}"
COLLECT_FEATURES_OVERRIDE="${COLLECT_FEATURES_OVERRIDE:-i we female shehe clout polite prosocial risk differ negate}"
PRIMARY_MODES_OVERRIDE="${PRIMARY_MODES_OVERRIDE:-increase decrease project}"

LAYERS_OVERRIDE="${LAYERS_OVERRIDE:-4 8 12 16 20 24 28}"
PRIMARY_STRENGTHS_OVERRIDE="${PRIMARY_STRENGTHS_OVERRIDE:-2.0 4.0 8.0}"

MAX_SAMPLES="${MAX_SAMPLES:-all}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
TOKEN_STATS_FORMAT="${TOKEN_STATS_FORMAT:-parquet}"
DO_SAMPLE="${DO_SAMPLE:-0}"
STRENGTH_UNIT="${STRENGTH_UNIT:-activation_pct}"
COLLECT_PROBE_LAYERS="${COLLECT_PROBE_LAYERS:-all}"
LOGPROB_CHUNK_SIZE="${LOGPROB_CHUNK_SIZE:-16}"
PROGRESS_STYLE="${PROGRESS_STYLE:-line}"
NO_GRADIENT_COSINES="${NO_GRADIENT_COSINES:-1}"
ACTIVATION_COSINE_COMPARISON="${ACTIVATION_COSINE_COMPARISON:-0}"
ACTIVATION_COSINE_LAYERS="${ACTIVATION_COSINE_LAYERS:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
ADAPTER_PATH_TEMPLATE="${ADAPTER_PATH_TEMPLATE:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a LAYERS <<< "$LAYERS_OVERRIDE"
read -r -a COLLECT_FEATURES <<< "$COLLECT_FEATURES_OVERRIDE"
read -r -a PRIMARY_MODES <<< "$PRIMARY_MODES_OVERRIDE"
read -r -a PRIMARY_STRENGTHS <<< "$PRIMARY_STRENGTHS_OVERRIDE"

if [ "${#LAYERS[@]}" -eq 0 ]; then
  echo "[feature_interaction] No layers configured." >&2
  exit 1
fi

if [ "${#PRIMARY_STRENGTHS[@]}" -eq 0 ]; then
  echo "[feature_interaction] No intervention strengths configured." >&2
  exit 1
fi

if [ "${#PRIMARY_MODES[@]}" -eq 0 ]; then
  echo "[feature_interaction] No intervention modes configured." >&2
  exit 1
fi

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw//:/_}"
  raw="${raw// /_}"
  raw="${raw//./p}"
  raw="${raw//- /-}"
  raw="${raw//-/m}"
  printf '%s' "$raw"
}

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

run_job() {
  local layer="$1"
  local primary_strength="$2"
  local primary_mode="$3"
  local adapter_for_layer="$ADAPTER_PATH"

  if [ -z "$adapter_for_layer" ] && [ -n "$ADAPTER_PATH_TEMPLATE" ]; then
    adapter_for_layer="$(expand_layer_path "$ADAPTER_PATH_TEMPLATE" "$layer")"
  elif [ -n "$adapter_for_layer" ]; then
    adapter_for_layer="$(expand_layer_path "$adapter_for_layer" "$layer")"
  fi

  local exp_name
  exp_name="$(
    printf '%s' \
      "${PRIMARY_FEATURE}_intervention_L${layer}_${primary_mode}_$(slugify "$primary_strength")"
  )"

  echo "[feature_interaction] layer=${layer} intervention=${PRIMARY_FEATURE} mode=${primary_mode} strength=${primary_strength}; collect_features=${COLLECT_FEATURES_OVERRIDE}"

  local cmd=(
    python -m scripts.feature_interaction
    --model-name "$MODEL"
    --data-path "$DATA"
    --model-filter-value "$MODEL_FILTER_VALUE"
    --feature "${PRIMARY_FEATURE}@${layer}:${primary_mode}:${primary_strength}:${PRIMARY_PROBE_VERSION}"
    --probe-type "$PROBE_TYPE"
    --probe-dir "$PROBE_DIR"
    --probe-version "$PROBE_VERSION"
    --strength-unit "$STRENGTH_UNIT"
    --batch-size "$BATCH_SIZE"
    --generation-batch-size "$GENERATION_BATCH_SIZE"
    --logprob-chunk-size "$LOGPROB_CHUNK_SIZE"
    --max-length "$MAX_LENGTH"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --seed "$SEED"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --token-stats-format "$TOKEN_STATS_FORMAT"
    --progress-style "$PROGRESS_STYLE"
    --output-dir "$OUTPUT_ROOT"
    --experiment-name "$exp_name"
  )

  if [ "$NO_GRADIENT_COSINES" = "1" ]; then
    cmd+=(--no-gradient-cosines)
  fi

  if [ "$ACTIVATION_COSINE_COMPARISON" = "1" ]; then
    cmd+=(--activation-cosine-comparison)
  fi

  if [ -n "$ACTIVATION_COSINE_LAYERS" ]; then
    cmd+=(--activation-cosine-layers "$ACTIVATION_COSINE_LAYERS")
  elif [ "$ACTIVATION_COSINE_COMPARISON" = "1" ] || [ -n "$adapter_for_layer" ]; then
    cmd+=(--activation-cosine-layers "$layer")
  fi

  if [ -n "$adapter_for_layer" ]; then
    if [ ! -f "${adapter_for_layer}/adapter_config.json" ]; then
      echo "[feature_interaction] Missing adapter_config.json under adapter path: ${adapter_for_layer}" >&2
      exit 1
    fi
    cmd+=(--adapter-path "$adapter_for_layer")
  fi

  if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "all" ]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi

  if [ -n "$COLLECT_PROBE_LAYERS" ]; then
    cmd+=(--collect-probe-layers "$COLLECT_PROBE_LAYERS")
  fi

  for collect_feature in "${COLLECT_FEATURES[@]}"; do
    if [ -n "$collect_feature" ]; then
      cmd+=(--collect-feature "${collect_feature}:increase:0.0:${SECONDARY_PROBE_VERSION}")
    fi
  done

  if [ "$DO_SAMPLE" = "1" ]; then
    cmd+=(--do-sample)
  fi

  PYTHONPATH="$FE_ROOT" "${cmd[@]}"
}

declare -a JOB_MATRIX=()
for layer in "${LAYERS[@]}"; do
  for primary_mode in "${PRIMARY_MODES[@]}"; do
    for primary_strength in "${PRIMARY_STRENGTHS[@]}"; do
      JOB_MATRIX+=("${layer}|${primary_strength}|${primary_mode}")
    done
  done
done

TOTAL_JOBS="${#JOB_MATRIX[@]}"
echo "[feature_interaction] total_jobs=${TOTAL_JOBS}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_JOBS" ]; then
    echo "[feature_interaction] SLURM_ARRAY_TASK_ID=${TASK_ID} out of range 0..$((TOTAL_JOBS - 1)); skipping."
    exit 0
  fi
  IFS='|' read -r layer primary_strength primary_mode <<< "${JOB_MATRIX[$TASK_ID]}"
  run_job "$layer" "$primary_strength" "$primary_mode"
else
  for job in "${JOB_MATRIX[@]}"; do
    IFS='|' read -r layer primary_strength primary_mode <<< "$job"
    run_job "$layer" "$primary_strength" "$primary_mode"
  done
fi
