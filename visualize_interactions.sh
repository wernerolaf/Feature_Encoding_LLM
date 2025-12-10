#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=$(pwd)

python -m scripts.visualize_interactions \
  --experiment-dir artifacts/experiments/gender_clout