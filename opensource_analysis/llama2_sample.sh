#python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
#    --output_path /data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_wiki \
#    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --mode qa \
#    --log_file log/llama2_sample_uncertain_wiki.log 

python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_wmt14 \
    --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv --mode translation \
    --log_file log/llama2_sample_uncertain_wmt14.log 

python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_eli5_category \
    --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv --mode qa \
    --log_file log/llama2_sample_uncertain_eli5_category.log 