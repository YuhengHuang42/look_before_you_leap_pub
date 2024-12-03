import argparse
import shelve
import spacy
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import logging
import os

import sys
sys.path.append('../')
from eval_utils import Evaluator
import abstract_state
import utils
from utils import LLAMA2_PROMPT_LENGTH

'''
python llm_evaluation.py --output_path /data/huangyuheng/QA/gpt2/wiki/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/gpt2/eli5_category/greedy/raw_inference \
    --llm_device "cuda:1" --eval_device "cuda:1" --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv \
    --log_file log/gpt2_greedy_eli5_category.log --mode qa > gpt2_greedy_eli5_category_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/gpt2/cnn_daily/greedy/raw_inference \
    --llm_device "cuda:2" --eval_device "cuda:2" --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv \
    --log_file log/gpt2_greedy_cnndaily.log --mode summarization > gpt2_greedy_cnndaily_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/gpt2/wmt14_test/greedy/raw_inference \
    --llm_device "cuda:2" --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv \
    --log_file log/gpt2_greedy_wmt14.log --mode translation > gpt2_greedy_wmt14_error.log 2>&1 &
    
# LLAMA
 nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama/shortQA/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/natural_qa_validation.csv --model llama \
    --llm_device "cuda:0" --eval_device "cuda:0" --log_file llama_greedy_shortQA.log \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ > llama_greedy_shortQA_error.log 2>&1 &
    
 nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama/eli5_category/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv --model llama \
    --llm_device "cuda:0" --eval_device "cuda:0" --log_file log/llama_greedy_eli5_category.log \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ > llama_greedy_eli5_category_error.log 2>&1 &

 nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama/wiki/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --model llama --mode qa \
    --llm_device "cuda:2" --eval_device "cuda:2" --log_file log/llama_greedy_wiki_category.log  \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ > llama_greedy_wiki_category_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama/cnn_daily/greedy/raw_inference \
    --llm_device "cuda:3" --eval_device "cuda:3" --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ --model llama \
    --log_file log/llama_greedy_cnndaily.log --mode summarization > llama_greedy_cnndaily_error.log 2>&1 &

nohup python -u llm_evaluation.py --output_path /data/huangyuheng/QA/llama/wmt14_test/greedy/raw_inference \
    --llm_device "cuda:3" --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ --model llama \
    --log_file log/llama_greedy_wmt14.log --mode translation > llama_greedy_wmt14_error.log 2>&1 &
    
# LLAMA2

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
    --llm_device "cuda:1" --model llama2 --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv \
    --log_file log/llama2_greedy_wmt14.log --mode translation > llama2_greedy_wmt14_error.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 python -u llm_evaluation.py --output_path  /data/huangyuheng/QA/phi3/cnn_daily/greedy/raw_inference \
                                --model phi3 --mode summarization --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv \
                                --parallel True --log_file log/phi3_greedy_cnn_daily.log
'''

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "llama", "llama2", "llama3", "phi3", "gemma2"])
    parser.add_argument("--model_path", type=str, default=None, help="path to the model. Only used for LLAMA")
    parser.add_argument("--llm_device", type=str, default="cuda")
   # parser.add_argument("--eval_device", type=str, default="cuda:1")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--simplified", type=bool, default=True)
    parser.add_argument("--dataset_path", type=str, default="eli5_category_test.csv")
    parser.add_argument("--print_freq", type=int, default=10, help="Every length/n evaluation will be printed")
    #parser.add_argument("--only_log_this", default=False, action="store_true", 
    #                    help="Only log the evaluation of this file")
    parser.add_argument("--log_file", type=str, default="llm_evaluation.log")
    parser.add_argument("--mode", type=str, default='qa', choices=["qa", "summarization", "translation"])
    parser.add_argument("--disable_template", default=False, action="store_true")
    parser.add_argument("--parallel", default=False, type=bool, help="Whether to use parallel processing")
    args = parser.parse_args()
    
    #if args.only_log_this:
    #    file_filter = utils.FileFilter(os.path.basename(__file__))
    #else:
    #    file_filter = None
    file_filter = None
    logger, file_handler = utils.setupLogging(args.log_file, file_filter)
    logger.info("Running %s" % " ".join(sys.argv))
    
    '''
    if args.model == "gpt2":
        logger.info("Using GPT-2 model")
    elif args.model == "llama":
        logger.info("Using LLAMA model")
    elif args.model == "llama2":
        logger.info("Loading LLAMA2 from huggingface")
    elif args.model == "llama3":
        logger.info("Loading LLAMA3 from huggingface")
    else:
        raise Exception("Model not supported")
    '''
    wrapper_model, tokenizer = utils.load_opensource_model(args.model, args.model_path, parallel=args.parallel)
    if args.parallel == False:
        wrapper_model.model.to(args.llm_device)
        
    #nlp = spacy.load(args.spacy_model)
    nlp = None
    sentencemodel = SentenceTransformer(args.sentence_model)
    sentencemodel.to(args.llm_device)
    evaluator = Evaluator(nlp, sentencemodel, mode=args.mode)
    
    new_df = pd.read_csv(args.dataset_path)
    dataset_len = len(new_df)
    
    if args.mode == "qa":
        mode_specific_args = {"max_new_tokens": 100}
        #if not args.disable_template:
        #    if args.model == "llama2":
        #        mode_specific_args["max_length"] += LLAMA2_PROMPT_LENGTH
    elif args.mode == "summarization" or args.mode == "translation":
        mode_specific_args = {"max_new_tokens": 100}
        #if args.model == "llama2" and not args.disable_template:
        #    mode_specific_args["max_new_tokens"] += LLAMA2_PROMPT_LENGTH
    template_processor = utils.TemplatePrcessor(args.model, args.disable_template, tokenizer=tokenizer)
    
    save_path = args.output_path
    ans_recored = shelve.open(save_path)
    print_threshold = dataset_len // args.print_freq
    for index, row in new_df.iterrows():
        question = row["question"]
        question_with_tempalte = template_processor.wrap_in_template(question)
        answer = row["answer"]
        infer_output = wrapper_model.forward(question_with_tempalte, mode_specific_args=mode_specific_args)
        text_output = evaluator.post_process(infer_output['decoded_text'][0], tokenizer.eos_token)
        infer_output["evaluation"] = evaluator.compute_passage_score(text_output, answer)
        infer_output["evaluation"]["task_metric"] = evaluator.compute_task_specific_score(text_output, answer)
        if args.simplified:
            infer_output = abstract_state.get_topk_token_output(infer_output, tokenizer)
            infer_output["entropy"] = utils.compute_entropy(infer_output['scores'])
            infer_output.pop('scores')
        infer_output['evaluation']['input'] = question_with_tempalte
        infer_output['evaluation']['gt'] = row["answer"]
        ans_recored[str(index)] = infer_output
        if index % print_threshold == 0:
            logger.info(f"Processed {index}/{dataset_len} samples")
    
    ans_recored.close()
    logger.info("Processing finished, saving to: {}".format(save_path))
    file_handler.close()
        
    
    
