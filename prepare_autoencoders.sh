#!/usr/bin/env bash
set -euo pipefail
MODEL="EleutherAI/pythia-70m-deduped"
DATA="data/LLM_mini.csv"
LAYERS=(0 1 2)

PYTHONPATH=$(pwd) python -m scripts.prepare_autoencoders \
  --model-name "$MODEL" \
  --data-path "$DATA" \
  --layers "${LAYERS[@]}" \
  --latent-dim 64 \
  --epochs 20 \
  --ae-batch-size 256 \
  --batch-size 8 \
  --max-length 256 \
  --dtype float16 \
  --log-dir artifacts/autoencoders
