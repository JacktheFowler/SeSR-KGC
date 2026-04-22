#!/bin/bash

export EXP_NAME=FB13_bert-base
# 指定使用 8 张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

mkdir -p ${EXP_ROOT}/cache_${EXP_NAME}
mkdir -p ${EXP_ROOT}/out_${EXP_NAME}

python -m torch.distributed.launch --nproc_per_node=8 run_triplet_classification.py \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir ./data/FB13 \
    --data_cache_dir ${EXP_ROOT}/cache_${EXP_NAME} \
    --model_cache_dir ${MODEL_CACHE_DIR} \
    --output_dir ${EXP_ROOT}/out_${EXP_NAME} \
    --model_name_or_path bert-base-cased \
    --pooling_model \
    --num_neg 1 \
    --only_corrupt_entity \
    --margin 7 \
    --no_mid \
    --max_seq_length 192 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 2048 \
    --learning_rate 3e-5 \
    --adam_epsilon 1e-6 \
    --num_train_epochs 5 \
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 64 \
    --save_total_limit 5 \
    --save_steps 5000 \
    --text_loss_weight 0.2 \
    --test_ratio 1.0 \
    --overwrite_output_dir \
    --seed 42 \
    --fp16 \
    --group_shuffle \
    --max_neighbors 10 \
    --structure_loss_weight 0.3 \
    --reconstruction_loss_weight 0.2 \
    --temperature 0.07
