#!/bin/bash
#SBATCH --job-name=LLM_NER_finetune
#SBATCH -p gpu_long
#SBATCH -t 12:00:00
#SBATCH --gres gpu:a6000:1

# export WANDB_PROJECT=LLM_NER_finetune

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
dataset='conll2003'
train_file=~/data/ner/${dataset}/train.jsonl
val_file=~/data/ner/${dataset}/validation.jsonl
test_file=~/data/ner/${dataset}/test.jsonl
output_dir=save_models/${dataset}/${model_name}
format='single' # single or multi
extend_context=True
language='en' # en or ja
seed=0

mkdir -p $output_dir
uv run torchrun --nproc_per_node 1 src/train.py
    --do_train \
    --do_eval \
    --model $model_name \
    --train_file $train_file \
    --validation_file $val_file \
    --test_file $test_file \
    --format $format \
    --language $language \
    --extend_context $extend_context \
    --output_dir $output_dir \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --run_name ${model_name}_${seed} \
    --seed $seed



