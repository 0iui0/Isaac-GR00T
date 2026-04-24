#!/bin/bash
export HF_HOME=$HF_HOME
source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
    --model-path /datasets/so101_pick_orange_finetune/checkpoint-2000 \
    --embodiment-tag new_embodiment \
    --port 5555 \
    --default-language "pick up the orange"
