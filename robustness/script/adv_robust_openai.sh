#!/bin/zsh

model=$1	# text-curie-001, text-davinci-003
dataset=$2	# alpaca, stable-vicuna, vicuna, original

python ../src/run_openai.py --mode adv_robust --model $model --task sst2 --dataset $dataset

python ../src/run_openai.py --mode adv_robust --model $model --task qqp --dataset $dataset

python ../src/run_openai.py --mode adv_robust --model $model --task mnli --dataset $dataset

python ../src/run_openai.py --mode adv_robust --model $model --task mnli-mm --dataset $dataset

python ../src/run_openai.py --mode adv_robust --model $model --task qnli --dataset $dataset

python ../src/run_openai.py --mode adv_robust --model $model --task rte --dataset $dataset
