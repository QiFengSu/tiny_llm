#!/bin/bash
set -e
export PATH="$PATH:/share/miniconda3/bin"
source activate
conda activate sqf_llm_train
export CUDA_VISIBLE_DEVICES=1,2,3
# 启动参数
OUT_DIR="./result/20250928_pretrain_v2"
DATA_PATH="./data/pretrain_hq.jsonl"

# 使用 accelerate 启动（单机多卡）
accelerate launch --config_file /nlp_group/suqifeng/minimind/train_script/accelerate_config.yaml \
  /nlp_group/suqifeng/minimind/trainer/pretrain.py \
  --out_dir ${OUT_DIR} \
  --data_path ${DATA_PATH} \
  --epochs 3 \
  --batch_size 128 \
  --eval_batch_size 128 \
  --learning_rate 5e-5 \
  --dtype bfloat16 \
  --accumulation_steps 1 \
  --grad_clip 1.0 \
  --log_interval 10 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 512 \
  --num_workers 8 \
  --log_interval 10 \
  --save_steps 0.5 \