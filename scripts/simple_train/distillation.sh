#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 多卡
torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_distillation \
    --student_model_path outputs/simple_full_sft \
    --teacher_model_path outputs/teacher_model \
    --data_path data/sft_t2t_mini.jsonl \
    --output_dir outputs/simple_distillation \
    --use_wandb \
    "$@"

## 单卡
#python -m aislant_minimind.simple_train.train_distillation_single \
#    --student_model_path outputs/simple_full_sft_single \
#    --teacher_model_path outputs/teacher_model \
#    --data_path data/sft_t2t_mini.jsonl \
#    --output_dir outputs/simple_distillation_single \
#    --use_wandb \
#    "$@"
