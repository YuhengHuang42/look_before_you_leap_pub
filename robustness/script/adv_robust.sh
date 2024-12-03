#!/bin/zsh

uncertainty_method=$1	# raw_inference, perturbation_uncertainty, sample_uncertainty
model=$2	# llama, llama2
dataset=$3	# alpaca, stable-vicuna, vicuna, original

python ../src/adv_robust.py --uncertainty_method $uncertainty_method --mode adv_robust --model $model --task sst2 --dataset $dataset

python ../src/adv_robust.py --uncertainty_method $uncertainty_method --mode adv_robust --model $model --task qqp --dataset $dataset

python ../src/adv_robust.py --uncertainty_method $uncertainty_method --mode adv_robust --model $model --task mnli --dataset $dataset

python ../src/adv_robust.py --uncertainty_method $uncertainty_method --mode adv_robust --model $model --task mnli-mm --dataset $dataset

python ../src/adv_robust.py --uncertainty_method $uncertainty_method --mode adv_robust --model $model --task qnli --dataset $dataset

python ../src/adv_robust.py --uncertainty_method $uncertainty_method --mode adv_robust --model $model --task rte --dataset $dataset


