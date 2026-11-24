#!/usr/bin/env bash
# 多卡: 默认使用所有可见的卡, 可以使用 CUDA_VISIBLE_DEVICES=xxx 来设置可见卡
torchrun --nproc_per_node=gpu -m aislant_minimind.simple_train.train_pretrain \
    --tokenizer_path model_weights/tokenizers/Llama-2-7b-hf \
    --data_path data/pretrain_t2t_mini.jsonl \
    --output_dir outputs/simple_pretrain  \
    -- use_wandb

## 单卡
#python -m aislant_minimind.simple_train.train_pretrain_single \
#    --tokenizer_path model_weights/tokenizers/Llama-2-7b-hf \
#    --data_path data/pretrain_t2t_mini.jsonl \
#    --output_dir outputs/simple_pretrain_single \
#    --use_wandb
