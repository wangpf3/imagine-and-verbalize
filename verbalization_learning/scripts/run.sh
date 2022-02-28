#!/bin/bash

project_dir='.'


dataset="commongen_inhouse"
eval_split='dev,test'
method="ng2text"
model_type="t5-base"
max_enc_length=128
max_dec_length=128
train_batch_size=16
eval_batch_size=32
grad_step=1
learning_rate=1e-4
weight_decay=1e-2
num_epoch=5
num_beams=10
episodes='41-43'
data_seed=42
train_ratio=0
graph_source_alpha=0.5

# trained imagination checkpoint
graph_generator_dir=$1

save_dir="${project_dir}/checkpoints/${dataset}_${train_ratio}_${method}_mixGT${graph_source_alpha}_${model_type}_bs${train_batch_size}-${num_gpu}_gs${grad_step}_lr${learning_rate}_wd${weight_decay}_e${num_epoch}_seed${episodes}"
mkdir -p $save_dir

# export SLURM_NTASKS=$num_gpu
# srun --gres=gpu:${num_gpu} python \
num_gpu=1
export CUDA_VISIBLE_DEVICES=0
export NGPU=$num_gpu
nohup python -m torch.distributed.launch --nproc_per_node=$num_gpu \
	main.py \
	--do_train \
	--inference \
	--evaluate \
	--episodes $episodes \
    --data_seed $data_seed \
	--train_ratio $train_ratio \
	--project_dir $project_dir \
	--graph_generator_dir $graph_generator_dir \
	--eval_split $eval_split \
	--overwrite_output \
	--method $method \
	--graph_source_alpha $graph_source_alpha \
    --dataset $dataset \
    --save_dir $save_dir \
	--model_type $model_type \
	--max_enc_length $max_enc_length \
	--max_dec_length $max_dec_length \
	--train_batch_size $train_batch_size \
	--eval_batch_size $eval_batch_size \
	--grad_step $grad_step \
	--learning_rate $learning_rate \
	--weight_decay $weight_decay \
	--num_epoch $num_epoch \
	--num_beams $num_beams \
    > ${save_dir}/debug.log 2>&1 &
