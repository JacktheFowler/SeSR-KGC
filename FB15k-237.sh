#!/bin/bash
unzip ./data/FB15k-237.zip -d ./data/FB15k-237
export EXP_NAME=FB15k-237_bert-base
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

mkdir -p ${EXP_ROOT}/cache_${EXP_NAME}
mkdir -p ${EXP_ROOT}/out_${EXP_NAME}

python run_link_prediction.py \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir ./data/FB15k-237 \
    --data_cache_dir ${EXP_ROOT}/cache_${EXP_NAME} \
    --model_cache_dir ${MODEL_CACHE_DIR} \
    --output_dir ${EXP_ROOT}/out_${EXP_NAME} \
    --model_name_or_path bert-base-cased \
    --pooling_model \
    --num_neg 10 \
    --only_corrupt_entity \
    --margin 7 \
    --no_mid \
    --max_seq_length 192\
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 3e-5 \
    --adam_epsilon 1e-6 \
    --num_train_epochs 5\
    --warmup_steps 3000 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1\
    --save_total_limit 5 \
    --save_steps 5000 \
    --text_loss_weight 0.2 \
    --test_ratio 1.0 \
    --overwrite_output_dir \
    --seed 42 \
    --group_shuffle \
    --max_neighbors 10\
    --structure_loss_weight 0.3 \
    --reconstruction_loss_weight 0.2 \
    --temperature 0.07
