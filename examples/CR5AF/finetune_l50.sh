#!/usr/bin/env bash
set -euo pipefail

# ─── CR5AF v5: lookahead=50 action delta + tune-visual + top LLM layers ───
#
# Usage: bash examples/CR5AF/finetune_l50.sh

# Proxy for wandb
export https_proxy=http://192.168.16.152:7897
export http_proxy=http://192.168.16.152:7897
export WANDB_API_KEY=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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=100000
export CUDA_VISIBLE_DEVICES=1

# ─── Config ───
BASE_MODEL="/home/zpa/.cache/modelscope/nv-community/GR00T-N1.7-3B"
DATASET_PATH="/datasets/cr5af_grasp_housing_l50"
OUTPUT_DIR="/tmp/cr5af_finetune_v5"

# ─── Launch ───
.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --modality-config-path examples/CR5AF/cr5af_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "$OUTPUT_DIR" \
  --use-wandb \
  --wandb-project gr00t-cr5af-finetune \
  --experiment-name grasp-housing-v5-l50 \
  --max-steps 20000 \
  --save-steps 5000 \
  --global-batch-size 4 \
  --dataloader-num-workers 0 \
  --learning-rate 1e-5 \
  --episode-sampling-rate 0.1 \
  --tune-visual \
  --no-tune-llm \
  --tune-top-llm-layers 2
