#!/usr/bin/env bash
set -euo pipefail

# ─── Train CR5AF with visual encoder unfrozen (continued from v2 checkpoint) ───
#
# Usage: bash examples/CR5AF/finetune_tune_visual.sh

# Proxy for wandb
export https_proxy=http://192.168.16.152:7897
export http_proxy=http://192.168.16.152:7897
export WANDB_API_KEY=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

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use GPU 1 (GPU 0 has Isaac Sim)
export CUDA_VISIBLE_DEVICES=1

# ─── Config ───
# Continue from v2 checkpoint (frozen VLM, DiT only) and unfreeze visual encoder
BASE_CHECKPOINT="/tmp/cr5af_finetune_v2/grasp-housing-v2-103eps"
DATASET_PATH="/datasets/cr5af_grasp_housing_v2"
OUTPUT_DIR="/tmp/cr5af_finetune_v3"

# ─── Launch ───
.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_CHECKPOINT" \
  --dataset-path "$DATASET_PATH" \
  --modality-config-path examples/CR5AF/cr5af_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "$OUTPUT_DIR" \
  --use-wandb \
  --wandb-project gr00t-cr5af-finetune \
  --experiment-name grasp-housing-v3-tunevisual \
  --max-steps 20000 \
  --save-steps 5000 \
  --global-batch-size 8 \
  --dataloader-num-workers 0 \
  --learning-rate 1e-5 \
  --episode-sampling-rate 0.1 \
  --tune-visual \
  --no-tune-llm
