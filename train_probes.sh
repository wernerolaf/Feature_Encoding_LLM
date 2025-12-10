#!/usr/bin/env bash
set -euo pipefail
MODEL="EleutherAI/pythia-70m-deduped"
DATA="data/LLM_mini.xlsx"
SHEET="Qwen_110B"
LABELS=(gender Clout)
LAYERS=(0 1 2)

for L in "${LAYERS[@]}"; do
  AE="artifacts/autoencoders/EleutherAI_pythia-70m-deduped_layer${L}_autoencoder.pt"
  for LABEL in "${LABELS[@]}"; do
    PYTHONPATH=$(pwd) python -m scripts.train_probes \
      --model-name "$MODEL" \
      --data-path "$DATA" \
      --sheet "$SHEET" \
      --label-column "$LABEL" \
      --layer "$L" \
      --batch-size 8 \
      --max-length 256 \
      --dtype float16 \
      --standardizer autoencoder \
      --autoencoder-artifact "$AE" \
      --probe-type linear \
      --test-size 0.2 \
      --random-state 0 \
      --save-path "artifacts/probes/pythia70m_L${L}_${LABEL}.pkl"
  done
done
