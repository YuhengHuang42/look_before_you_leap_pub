#!/bin/zsh
#!/usr/bin//usr/bin/python33

model=$1        # text-curie-001, text-davinci-003
task=$2         # style, knowledge

if [[ $task == "style" ]]; then
        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "original" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "augment" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "shake_w" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "shake_p0" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "shake_p0.6" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "tweet_p0" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "tweet_p0.6" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "bible_p0" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "bible_p0.6" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "romantic_p0" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "romantic_p0.6" --few_shot_num 0 --idk


elif [[ $task == "knowledge" ]]; then
        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "qa_2020" --few_shot_num 0 --idk

        python ../src/run_openai.py --mode ood_robust --model $model --task $task --dataset "qa_2023" --few_shot_num 0 --idk

fi
