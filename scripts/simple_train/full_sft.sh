#!/usr/bin/env bash

# 无论从哪个目录调用，都统一在项目根目录运行。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 多卡: 默认使用所有可见的卡，可以使用 CUDA_VISIBLE_DEVICES=xxx 设置可见卡。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_full_sft \
    --model_path outputs/simple_pretrain \
    --data_path data/sft_t2t_mini.jsonl \
    --output_dir outputs/simple_full_sft \
    --use_wandb \
    "$@"

## 单卡
#python -m aislant_minimind.simple_train.train_full_sft_single \
#    --model_path outputs/simple_pretrain_single \
#    --data_path data/sft_t2t_mini.jsonl \
#    --output_dir outputs/simple_full_sft_single \
#    --use_wandb \
#    "$@"
