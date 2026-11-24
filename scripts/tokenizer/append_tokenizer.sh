#!/usr/bin/env bash

set -euo pipefail

# 无论从哪个目录调用，都统一在项目根目录运行。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 在原始 Llama 2 tokenizer 后追加项目 token。
python -m aislant_minimind.model.tokenizer.append_tokenizer \
    --base_tokenizer_path model_weights/tokenizers/Llama-2-7b-hf \
    --output_dir model_weights/tokenizers/aislant_minimind_llama \
    "$@"

