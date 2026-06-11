#!/usr/bin/env bash
set -euo pipefail

# ─── CR5AF: continue from v3, unfreeze top 4 LLM layers ───
#
# Usage: bash examples/CR5AF/finetune_tune_llm.sh

# Proxy for wandb
export https_proxy=http://192.168.16.152:7897
export http_proxy=http://192.168.16.152:7897
export WANDB_API_KEY=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

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=100000

# Use GPU 1 (GPU 0 has Isaac Sim)
export CUDA_VISIBLE_DEVICES=1

# ─── Config ───
# Continue from v3 checkpoint (tune-visual) and unfreeze top LLM layers
BASE_CHECKPOINT="/tmp/cr5af_finetune_v3/grasp-housing-v3-tunevisual"
DATASET_PATH="/datasets/cr5af_grasp_housing_v2"
OUTPUT_DIR="/tmp/cr5af_finetune_v4"

# ─── Launch ───
.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_CHECKPOINT" \
  --dataset-path "$DATASET_PATH" \
  --modality-config-path examples/CR5AF/cr5af_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "$OUTPUT_DIR" \
  --use-wandb \
  --wandb-project gr00t-cr5af-finetune \
  --experiment-name grasp-housing-v4-tunellm \
  --max-steps 10000 \
  --save-steps 5000 \
  --global-batch-size 2 \
  --dataloader-num-workers 0 \
  --learning-rate 1e-6 \
  --episode-sampling-rate 0.1 \
  --tune-visual \
  --no-tune-llm \
  --tune-top-llm-layers 2
