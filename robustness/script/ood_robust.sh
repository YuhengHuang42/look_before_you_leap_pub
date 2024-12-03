#!/bin/zsh

uncertainty_method=$1	# raw_inference, perturbation_uncertainty, sample_uncertainty
model=$2	# gpt2, llama
task=$3		# style, knowledge

if [[ $task == "style" ]]; then
	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "original" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "augment" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "shake_w" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "shake_p0" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "shake_p0.6" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "tweet_p0" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "tweet_p0.6" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "bible_p0" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "bible_p0.6" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "romantic_p0" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "romatic_p0.6" --few_shot_num 0 --idk


elif [[ $task == "knowledge" ]]; then
	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "qa_2020" --few_shot_num 0 --idk

	python ../src/ood_robust.py --uncertainty_method $uncertainty_method --mode ood_robust --model $model --task $task --dataset "qa_2023" --few_shot_num 0 --idk
	
fi
