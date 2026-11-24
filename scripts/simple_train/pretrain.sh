#!/usr/bin/env bash

# 无论从哪个目录调用，都统一在项目根目录运行。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 多卡: 默认使用所有可见的卡, 可以使用 CUDA_VISIBLE_DEVICES=xxx 来设置可见卡
# 提前设置每个训练进程使用的 CPU 线程数，避免 torchrun 自动设置时打印 W1124 警告。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_pretrain \
    --tokenizer_path model_weights/tokenizers/aislant_minimind \
    --data_path data/pretrain_t2t_mini.jsonl \
    --output_dir outputs/simple_pretrain  \
    --use_wandb \
    "$@"

## 单卡
#python -m aislant_minimind.simple_train.train_pretrain_single \
#    --tokenizer_path model_weights/tokenizers/aislant_minimind \
#    --data_path data/pretrain_t2t_mini.jsonl \
#    --output_dir outputs/simple_pretrain_single \
#    --use_wandb \
#    "$@"
