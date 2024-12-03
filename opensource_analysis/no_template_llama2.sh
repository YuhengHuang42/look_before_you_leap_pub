nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/wiki/greedy/notemplate \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --model llama2 --mode qa \
    --llm_device "cuda:0" --log_file log/llama2_greedy_wiki_category.log --disable_template > llama2_greedy_wiki_category_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/eli5_category/greedy/notemplate \
    --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv --model llama2 --mode qa \
    --llm_device "cuda:1" --log_file log/llama2_greedy_eli5_category.log --disable_template > llama2_greedy_eli5_category_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/cnn_daily/greedy/notemplate \
    --llm_device "cuda:2" --model llama2 --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv \
    --log_file log/llama2_greedy_cnndaily.log --mode summarization --disable_template > llama2_greedy_cnndaily_error.log 2>&1 &
    
nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/wmt14_test/greedy/notemplate \
    --llm_device "cuda:3" --model llama2 --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv \
    --log_file log/llama2_greedy_wmt14.log --mode translation --disable_template > llama2_greedy_wmt14_error.log 2>&1 &


nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/notemplate_sample_uncertain_wiki \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --mode qa \
    --log_file log/llama2_sample_uncertain_wiki.log --disable_template > llama2_sample_uncertain_wiki_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/notemplate_sample_uncertain_wmt14 \
    --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv --mode translation \
    --log_file log/llama2_sample_uncertain_wmt14.log --disable_template > llama2_sample_uncertain_wmt14_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/notemplate_sample_uncertain_eli5_category \
    --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv --mode qa \
    --log_file log/llama2_sample_uncertain_eli5_category.log --disable_template > llama2_sample_uncertain_eli5_category_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama2/notemplate_sample_uncertain_cnn_dailymail" \
    --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv --mode summarization \
    --log_file log/llama2_sample_uncertain_cnn_dailymail.log --disable_template > llama2_sample_uncertain_cnn_dailymail_error.log 2>&1 &


nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama2/cnn_daily/greedy/raw_inference" --devices "cuda:0" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama2/cnn_daily_alter"  --mode summarization \
    --model llama2 --log_file "log/uncertainty_llama2_cnn_daily.log" > uncertainty_llama2_cnn_daily_error.log 2>&1 &
    
nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama2/eli5_category/greedy/raw_inference" --devices "cuda:1" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama2/eli5_category_alter"  --mode qa \
    --model llama2 --log_file "log/uncertainty_llama2_eli5_category_alter.log" > uncertainty_llama2_eli5_category_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama2/wiki/greedy/raw_inference" --devices "cuda:1"  \
    --output_path "/data/huangyuheng/QA/uncertainty/llama2/wiki_alter"  --mode qa \
    --model llama2 --log_file "log/uncertainty_llama2_wiki_alter.log" > uncertainty_llama2_wiki_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama2/wmt14_test/greedy/raw_inference" --devices "cuda:1" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama2/wmt14_alter"  --mode translation  \
    --model llama2 --log_file "log/uncertainty_llama2_wmt14_alter.log" > uncertainty_llama2_wmt14_alter_error.log 2>&1 &