#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_PATH="tests/data/pretrain_resume_80.jsonl"
TOKENIZER_PATH="model_weights/tokenizers/Llama-2-7b-hf"

# 这是端到端流程测试，使用小模型缩短运行时间。
# 80 条数据 / 2 张卡 / batch_size 4 = 每个 rank 每个 epoch 10 个 batch。
COMMON_ARGS=(
  --tokenizer_path "${TOKENIZER_PATH}"
  --data_path "${DATA_PATH}"
  --hidden_size 128
  --num_hidden_layers 2
  --num_attention_heads 4
  --num_key_value_heads 2
  --max_seq_len 128
  --epochs 2
  --batch_size 4
  --learning_rate 5e-4
  --accumulation_steps 4
  --grad_clip 1.0
  --dtype bfloat16
  --device auto
  --num_workers 2
  --seed 42
  --lr_scheduler cosine
  --warmup_ratio 0.0
  --min_lr_ratio 0.1
  --log_interval 1
  --save_strategy steps
  --save_interval 1
  --use_wandb false
)

assert_equal() {
  local actual="$1"
  local expected="$2"
  local message="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "检查失败: ${message}，期望 ${expected}，实际 ${actual}" >&2
    exit 1
  fi
}

assert_log_count() {
  local log_file="$1"
  local expected="$2"
  local actual
  actual="$(grep -cE 'Epoch \[[0-9]+/2\] Step \[[0-9]+/10\]' "${log_file}" || true)"
  assert_equal "${actual}" "${expected}" "${log_file} 的 Step 日志行数"
}

for output_dir in out_base out_resume out_resume3; do
  if [[ -e "${output_dir}" ]]; then
    echo "测试输出目录已存在，请先移动或删除后重试: ${output_dir}" >&2
    exit 1
  fi
done

echo "[1/3] 完整基线训练"
torchrun --standalone --nproc_per_node=2 \
  -m aislant_minimind.trainer.train_pretrain \
  "${COMMON_ARGS[@]}" \
  --output_dir out_base \
  2>&1 | tee baseline.log

for step in 1 2 3 4 5 6; do
  checkpoint_dir="out_base/checkpoints/checkpoint-${step}"
  [[ -f "${checkpoint_dir}/trainer_state.pt" ]] || {
    echo "检查失败: 缺少 ${checkpoint_dir}/trainer_state.pt" >&2
    exit 1
  }
  compgen -G "${checkpoint_dir}/*.safetensors" >/dev/null || {
    echo "检查失败: ${checkpoint_dir} 中没有 safetensors 权重" >&2
    exit 1
  }
done
assert_log_count baseline.log 20

echo "[2/3] 从 epoch 中间 checkpoint-2 续训"
torchrun --standalone --nproc_per_node=2 \
  -m aislant_minimind.trainer.train_pretrain \
  "${COMMON_ARGS[@]}" \
  --output_dir out_resume \
  --resume_from_checkpoint out_base/checkpoints/checkpoint-2 \
  2>&1 | tee resume_mid_epoch.log

grep -qE '续训: epoch 1, step 8, global_step 2' resume_mid_epoch.log || {
  echo "检查失败: 中途续训日志中的恢复位置不正确" >&2
  exit 1
}
grep -qE 'Epoch \[1/2\]: 跳过前 8 个 step，从 step 9 开始' resume_mid_epoch.log || {
  echo "检查失败: 中途续训日志中没有正确的跳过信息" >&2
  exit 1
}
assert_log_count resume_mid_epoch.log 12

echo "[3/3] 从 epoch 边界 checkpoint-3 续训"
torchrun --standalone --nproc_per_node=2 \
  -m aislant_minimind.trainer.train_pretrain \
  "${COMMON_ARGS[@]}" \
  --output_dir out_resume3 \
  --resume_from_checkpoint out_base/checkpoints/checkpoint-3 \
  2>&1 | tee resume_epoch_boundary.log

grep -qE '续训: epoch 2, step 0, global_step 3' resume_epoch_boundary.log || {
  echo "检查失败: epoch 边界续训日志中的恢复位置不正确" >&2
  exit 1
}
if grep -q '跳过前' resume_epoch_boundary.log; then
  echo "检查失败: epoch 边界续训不应出现跳过日志" >&2
  exit 1
fi
assert_log_count resume_epoch_boundary.log 10

echo "三组训练及其验收检查均已通过。"
