#!/bin/bash

num_gpu=1

mask_ratio=0
dataset="roc_concept2story"
model_type="t5-large"
separation='rel'
batch_size=8
grad_step=2
learning_rate=1e-4
warmup_ratio=0.01
weight_decay=1e-2
smooth_factor=0
num_epoch=5
gpu_device=0
num_workers=0
num_beams=10
pretrain_dir=$1
eval_split='test'
eval_batch_size=16
ckpt_num=-1

save_dir="./checkpoints/${dataset}_${model_type}_${separation}_uniform-mask${mask_ratio}_bs${batch_size}-${num_gpu}_gs${grad_step}_lr${learning_rate}_wp${warmup_ratio}_wd${weight_decay}_sm${smooth_factor}_e${num_epoch}"
mkdir -p $save_dir

# export SLURM_NTASKS=$num_gpu
# srun --gres=gpu:${num_gpu} python \
export CUDA_VISIBLE_DEVICES=0
export NGPU=$num_gpu
nohup python -m torch.distributed.launch --nproc_per_node=$num_gpu \
    main.py \
    --do_train \
    --textualization \
    --ckpt_num $ckpt_num \
    --eval_batch_size $eval_batch_size \
    --parse_output \
    --overwrite_output \
    --dataset $dataset \
    --save_dir $save_dir \
    --pretrain_dir $pretrain_dir \
    --model_type $model_type \
    --separation $separation \
    --mask_ratio $mask_ratio \
    --batch_size $batch_size \
    --grad_step $grad_step \
    --warmup_ratio $warmup_ratio \
    --weight_decay $weight_decay \
    --smooth_factor $smooth_factor \
    --learning_rate $learning_rate \
    --num_epoch $num_epoch \
    --gpu_device $gpu_device \
    --num_workers $num_workers \
    > ${save_dir}/debug.log 2>&1 &
