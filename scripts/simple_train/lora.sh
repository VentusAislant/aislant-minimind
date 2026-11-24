#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 多卡
torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_lora \
    --model_path outputs/simple_full_sft \
    --data_path data/lora_medical.jsonl \
    --output_dir outputs/simple_lora \
    --use_wandb \
    "$@"

## 单卡
#python -m aislant_minimind.simple_train.train_lora_single \
#    --model_path outputs/simple_full_sft_single \
#    --data_path data/lora_medical.jsonl \
#    --output_dir outputs/simple_lora_single \
#    --use_wandb \
#    "$@"
