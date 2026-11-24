#!/usr/bin/env bash

set -euo pipefail

# 无论从哪个目录调用，都统一在项目根目录运行。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 从语料训练全新的 Aislant MiniMind tokenizer。
python -m aislant_minimind.model.tokenizer.train_tokenizer \
    --data_paths data/pretrain_t2t_mini.jsonl data/sft_t2t_mini.jsonl \
    --output_dir model_weights/tokenizers/aislant_minimind \
    --vocab_size 6400 \
    "$@"
