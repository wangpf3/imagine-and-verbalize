#!/bin/bash

save_dir=$1
dataset=$2
eval_split=$3
eval_batch_size=16
ckpt_num=-1


num_beams=10
top_k=0
top_p=1.0
num_return_sequences=1



num_gpu=1
# export SLURM_NTASKS=$num_gpu
# srun --gres=gpu:6000:${num_gpu} --qos prior python \
export CUDA_VISIBLE_DEVICES=$4
nohup python -u \
    main.py \
    --is_training \
    --textualization \
    --eval_batch_size $eval_batch_size \
    --inference \
    --parse_output \
    --overwrite_output \
    --eval_split $eval_split \
    --ckpt_num $ckpt_num \
    --num_beams $num_beams \
    --top_k $top_k \
    --top_p $top_p \
    --num_return_sequences $num_return_sequences \
    --dataset $dataset \
    --save_dir $save_dir \
    > ${save_dir}/inference_${dataset}_${eval_split}_ckpt${ckpt_num}.log 2>&1 &
