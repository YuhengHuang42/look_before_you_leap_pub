nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/wiki/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --model llama2 --mode qa \
    --llm_device "cuda:1" --log_file log/llama2_greedy_wiki_category.log > llama2_greedy_wiki_category_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/eli5_category/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv --model llama2 --mode qa \
    --llm_device "cuda:2" --log_file log/llama2_greedy_eli5_category.log > llama2_greedy_eli5_category_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/cnn_daily/greedy/raw_inference \
    --llm_device "cuda:3" --model llama2 --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv \
    --log_file log/llama2_greedy_cnndaily.log --mode summarization > llama2_greedy_cnndaily_error.log 2>&1 &
    
nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama2/wmt14_test/greedy/raw_inference \
    --llm_device "cuda:0" --model llama2 --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv \
    --log_file log/llama2_greedy_wmt14.log --mode translation > llama2_greedy_wmt14_error.log 2>&1 &
