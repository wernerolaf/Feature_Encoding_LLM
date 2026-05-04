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
