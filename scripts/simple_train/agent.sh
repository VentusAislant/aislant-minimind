#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_agent \
    --model_path outputs/simple_full_sft \
    --data_path data/agent_rl.jsonl \
    --reward_model_path model_weights/reward_models/internlm2-1_8b-reward \
    --output_dir outputs/simple_agent \
    --use_wandb "$@"

## 单卡：将上面的 torchrun 命令换成以下命令，其余参数保持一致。
#python -m aislant_minimind.simple_train.train_agent_single ...
