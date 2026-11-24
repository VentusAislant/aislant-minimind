#!/usr/bin/env bash

# 无论从哪个目录调用，都统一在项目根目录运行。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 多卡: 默认使用所有可见的卡，可以使用 CUDA_VISIBLE_DEVICES=xxx 设置可见卡。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_dpo \
    --model_path outputs/simple_full_sft \
    --data_path data/dpo.jsonl \
    --output_dir outputs/simple_dpo \
    --use_wandb \
    "$@"

## 单卡
#python -m aislant_minimind.simple_train.train_dpo_single \
#    --model_path outputs/simple_full_sft_single \
#    --data_path data/dpo.jsonl \
#    --output_dir outputs/simple_dpo_single \
#    --use_wandb \
#    "$@"
