#!/usr/bin/env bash
set -euo pipefail
MODEL="EleutherAI/pythia-70m-deduped"
DATA="data/LLM_nano.xlsx"
SHEET="Qwen_110B"
OUT="artifacts/experiments/gender_clout"

PYTHONPATH=$(pwd) python -m scripts.feature_interaction \
  --model-name "$MODEL" \
  --data-path "$DATA" \
  --sheet "$SHEET" \
  --feature gender@1:increase:10.1 \
  --feature Clout@1:increase:0.0 \
  --feature gender@2:increase:0.0 \
  --feature Clout@2:increase:0.0 \
  --probe-type linear \
  --probe-dir artifacts/probes \
  --batch-size 8 \
  --max-length 256 \
  --max-new-tokens 32 \
  --temperature 0.0 \
  --output-dir "$OUT"
