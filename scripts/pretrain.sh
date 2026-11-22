#!/usr/bin/env bash

set -euo pipefail

# 统一从项目根目录运行，保证下面的相对路径稳定。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m aislant_minimind.trainer.train_pretrain \
  --tokenizer_path model_weights/tokenizers/llama \
  --data_path data/pretrain_t2t_mini.jsonl \
  --output_dir outputs/pretrain \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 4 \
  --max_seq_len 340 \
  --epochs 2 \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --accumulation_steps 8 \
  --grad_clip 1.0 \
  --dtype bfloat16 \
  --device auto \
  --num_workers 8 \
  --seed 42 \
  --lr_scheduler cosine \
  --warmup_ratio 0.0 \
  --min_lr_ratio 0.1 \
  --log_interval 100 \
  --save_strategy steps \
  --save_interval 1000 \
  "$@"
