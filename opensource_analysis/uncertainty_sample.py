import argparse
import shelve
import torch
import pandas as pd
import logging
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import time
import shutil
import torch.multiprocessing as mp
import gc
import tqdm

import sys
sys.path.append('../')
#import abstract_state
import utils
from utils import divide_list_equally
from utils import LLAMA2_PROMPT_LENGTH

TEMPERATURE = 0.7
TOPP = 0.95
TOPK = 0
MAX_CODE_LENGTH = 512

def main_evaluation_func(wrapper_model, mode, dataset, num_sample,
                         iterate_list, save_path, device, logger_file, 
                         print_threshold, mode_specific_args, template_processor):
    logger, file_handler = utils.setupLogging(logger_file, multiprocess=True)
    ans_recored = shelve.open(save_path)
    if mode == "code":
        task = mode_specific_args["task"]
        for id, index in enumerate(tqdm.tqdm(iterate_list)):
            mode_specific_args["sample_idx"] = index
            prompt = task.get_prompt(dataset[index])
            reference = task.get_reference(dataset[index])
            # One can also utilize `num_return_sequences`
            # parameter here. However, our GPU memory 
            # is limited for us to do so.
            single_entry = []
            #ans_recored[str(index)] = single_entry
            for _ in range(num_sample):
                infer_output = wrapper_model.forward(prompt, mode="code", mode_specific_args=mode_specific_args)
                text_output = infer_output['final_result']
                infer_output["evaluation"] = task.process_results([text_output], [reference])
                infer_output["entropy"] = utils.compute_entropy(infer_output['scores'])
                infer_output.pop('scores')
                #single_entry = ans_recored[str(index)]
                single_entry.append(infer_output)
                #ans_recored[str(index)] = single_entry
                #del infer_output
                #del single_entry
                #gc.collect()
                #torch.cuda.empty_cache()
            ans_recored[str(index)] = single_entry
            if int(id) % print_threshold == 0:
                logging_str = f"In device:{str(device)}, Processed {str(id)}/{str(len(iterate_list))} samples"
                logger.info(logging_str)
    else:
        for id, index in enumerate(tqdm.tqdm(iterate_list)):
            question = template_processor.wrap_in_template(dataset.iloc[index]["question"])
            #answer = dataset.iloc[index]["answer"]
            single_entry = []
            for _ in range(num_sample):
                infer_output = wrapper_model.forward(question, mode_specific_args=mode_specific_args)
                infer_output["entropy"] = utils.compute_entropy(infer_output['scores'])
                infer_output.pop('scores')
                single_entry.append(infer_output)
            ans_recored[str(index)] = single_entry
            if int(id) % print_threshold == 0:
                logging_str = f"In device:{str(device)}, Processed {str(id)}/{str(len(iterate_list))} samples"
                logger.info(logging_str)
    ans_recored.close()

'''
nohup python3 uncertainty_sample.py --model gpt2 --devices "cuda:0" \
    --output_path /data/huangyuheng/QA/uncertainty/gpt2/sample_uncertain_wiki \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --mode qa \
    --log_file log/gpt2_sample_uncertain_wiki.log > gpt2_sample_uncertain_wiki_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model gpt2 --devices "cuda:0" \
    --output_path /data/huangyuheng/QA/uncertainty/gpt2/sample_uncertain_wmt14 \
    --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv --mode translation \
    --log_file log/gpt2_sample_uncertain_wmt14.log > gpt2_sample_uncertain_wmt14_error.log 2>&1 &

nohup python -u uncertainty_sample.py --model gpt2 --output_path /data/huangyuheng/QA/uncertainty/gpt2/sample_uncertain_eli5_category \
    --devices "cuda:0" --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv \
    --log_file log/gpt2_sample_uncertain_eli5_category.log --mode qa > gpt2_sample_uncertain_eli5_category_error.log 2>&1 &
    
nohup python3 uncertainty_sample.py --model llama --devices "cuda:0" \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --output_path /data/huangyuheng/QA/uncertainty/llama/sample_uncertain_wiki \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --mode qa \
    --log_file log/llama_sample_uncertain_wiki.log > llama_sample_uncertain_wiki_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama --devices "cuda:0" \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --output_path /data/huangyuheng/QA/uncertainty/llama/sample_uncertain_wmt14 \
    --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv  --mode translation \
    --log_file log/llama_sample_uncertain_wmt14.log > llama_sample_uncertain_wmt14_error.log 2>&1 &

nohup python -u uncertainty_sample.py --model llama --output_path /data/huangyuheng/QA/uncertainty/llama/sample_uncertain_eli5_category \
    --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --devices "cuda:0" --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv \
    --log_file log/llama_sample_uncertain_eli5_category.log --mode qa > llama_sample_uncertain_eli5_category_error.log 2>&1 &

python3 -u uncertainty_sample.py --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/gpt2/sample_uncertain_cnn_dailymail"  --mode summarization \
        --log_file "log/gpt2_sample_uncertain_cnn_dailymail.log" --model gpt2

python3 -u uncertainty_sample.py --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv  --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama/sample_uncertain_cnn_dailymail"  --mode summarization --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --model llama --log_file "log/llama_sample_uncertain_cnn_dailymail.log" 

# LLAMA2

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_wiki \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv --mode qa \
    --log_file log/llama2_sample_uncertain_wiki.log > llama2_sample_uncertain_wiki_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_wmt14 \
    --dataset_path /data/huangyuheng/dataset/llm/wmt14_test.csv --mode translation \
    --log_file log/llama2_sample_uncertain_wmt14.log > llama2_sample_uncertain_wmt14_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path /data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_eli5_category \
    --dataset_path /data/huangyuheng/dataset/llm/eli5_category_test.csv --mode qa \
    --log_file log/llama2_sample_uncertain_eli5_category.log > llama2_sample_uncertain_eli5_category_error.log 2>&1 &

nohup python3 uncertainty_sample.py --model llama2 --devices "cuda:0" "cuda:1" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama2/sample_uncertain_cnn_dailymail" \
    --dataset_path /data/huangyuheng/dataset/llm/cnn_dailymail_3_test.csv --mode summarization \
    --log_file log/llama2_sample_uncertain_cnn_dailymail.log > llama2_sample_uncertain_cnn_dailymail_error.log 2>&1 &

# Code
python3 -u uncertainty_sample.py --model codegen --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path result/sample_uncertain/sample_uncertain_codegen_mbpp --dataset mbpp \
    --log_file log/sample_uncertain_codegen_mbpp.log --mode code

python3 -u uncertainty_sample.py --model santacoder --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path result/sample_uncertain/sample_uncertain_santacoder_mbpp --dataset mbpp \
    --log_file log/sample_uncertain_santacoder_mbpp.log --mode code

python3 -u uncertainty_sample.py --model incoder --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path result/sample_uncertain/sample_uncertain_incoder_mbpp --dataset mbpp \
    --log_file log/sample_uncertain_incoder_mbpp.log --mode code

python3 -u uncertainty_sample.py --model codegen --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path result/sample_uncertain/sample_uncertain_codegen_humaneval --dataset humaneval \
    --log_file log/sample_uncertain_codegen_humaneval.log --mode code

python3 -u uncertainty_sample.py --model santacoder --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path result/sample_uncertain/sample_uncertain_santacoder_humaneval --dataset humaneval \
    --log_file log/sample_uncertain_santacoder_humaneval.log --mode code

python3 -u uncertainty_sample.py --model incoder --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path result/sample_uncertain/sample_uncertain_incoder_humaneval --dataset humaneval \
    --log_file log/sample_uncertain_incoder_humaneval.log --mode code

CUDA_VISIBLE_DEVICES=0,1 nohup python3 -u uncertainty_sample.py --model codellama --parallel True \
    --output_path /data/huangyuheng/code/uncertainty/result/sample_uncertain/sample_uncertain_codellama_humaneval --dataset humaneval \
    --log_file log/sample_uncertain_codellama_humaneval.log --mode code > codellama_sample_uncertain_humaneval_error.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup python3 -u uncertainty_sample.py --model codellama --parallel True \
    --output_path /data/huangyuheng/code/uncertainty/result/sample_uncertain/sample_uncertain_codellama_mbpp --dataset mbpp \
    --log_file log/sample_uncertain_codellama_mbpp.log --mode code > codellama_sample_uncertain_mbpp_error.log 2>&1 &
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="santacoder", 
                        choices=["codegen", "santacoder", 
                                 "incoder", "gpt2", "llama", "llama2", "codellama",
                                 "llama3", "phi3", "gemma2", "deepseekcoder", 
                                 "codeqwen1.5", "starcoder2"]
                        )
    parser.add_argument("--model_path", type=str, default=None, help="path to the model. Only used for LLAMA-1")
    parser.add_argument('--devices', nargs='*', type=str, help='List of Possible devices', default=["cuda:0"])
    parser.add_argument("--output_path", type=str, default="debug.output")
    parser.add_argument("--simplified", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["mbpp", "humaneval"]) # Only used in code mode
    parser.add_argument("--print_freq", type=int, default=10, help="Every length/n evaluation will be printed")
    parser.add_argument("--log_file", type=str, default="debug.log")
    parser.add_argument("--per_sample_num", type=int, default=5,
                        help="The number of samples to generate for each input")
    parser.add_argument("--mode", type=str, default='code', choices=["qa", "summarization", "translation", "code"])
    parser.add_argument("--dataset_path", type=str, default=None) # Only used in NLP mode
    parser.add_argument("--parallel", default=False, type=bool, help="whether to use all GPUs for a single model")
    parser.add_argument("--disable_template", default=False, action="store_true")
    parser.add_argument('--resume_dir', type=str, default=None)
    args = parser.parse_args()
    
    #if args.only_log_this:
    #    file_filter = utils.FileFilter(os.path.basename(__file__))
    #else:
    #    file_filter = None
    file_filter = None
    logger, file_handler = utils.setupLogging(args.log_file, file_filter)
    logger.info("Running %s" % " ".join(sys.argv))
    
    logger.info(f"Using {args.model} model")
    
    if args.mode == "code":
        if args.dataset == "mbpp":
            from code_analysis import mbpp
            logger.info("Evaluating MBPP")
            task = mbpp.MBPP()
        elif args.dataset == "humaneval":
            from code_analysis import human_eval
            logger.info("Evaluating Human Eval")
            task = human_eval.HumanEval()
        dataset = task.get_dataset()
        dataset_len = len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(dataset_len)]
        mode_specific_args = {}
        mode_specific_args["task"] = task
        mode_specific_args["prefix"] = ""
        mode_specific_args["max_length"] = MAX_CODE_LENGTH # Refer to https://github.com/bigcode-project/bigcode-evaluation-harness/tree/3c8c685b6c162f034e7e0215b19cb75917ba6672
        mode_specific_args["postprocess"] = True
        mode_specific_args["batch_size"] = 1
        if args.model == "incoder":
            mode_specific_args["prefix"] = "<| file ext=.py |>\n"
        elif args.model == "codegen":
            mode_specific_args["prefix"] = "# Import libraries.\n import numpy as np"
        
        iterate_list = [i for i in range(dataset_len)]
    else:
        dataset = pd.read_csv(args.dataset_path)
        iterate_list = [i for i in range(len(dataset))]
    if args.mode == "qa":
        mode_specific_args = {"max_length": 100}
        if args.model == "llama2" and not args.disable_template:
            mode_specific_args["max_length"] += LLAMA2_PROMPT_LENGTH
    elif args.mode == "summarization" or args.mode == "translation":
        mode_specific_args = {"max_new_tokens": 100}
        if args.model == "llama2" and not args.disable_template:
            mode_specific_args["max_new_tokens"] += LLAMA2_PROMPT_LENGTH
        mode_specific_args["temperature"] = TEMPERATURE
        mode_specific_args["do_sample"] = True
        mode_specific_args["top_p"] = TOPP
        mode_specific_args["top_k"] = TOPK
        
            
    save_path = args.output_path
    assert not save_path.endswith("/")
    temp_dir = save_path + "_temp"
    temp_dir_list = list()
    os.makedirs(temp_dir, exist_ok=True)
    assert len(iterate_list) > 0
    if args.resume_dir is not None:
        assert args.resume_dir != temp_dir
        starting_idx, resume_path = utils.load_shelve_and_resume(args.resume_dir)
        iterate_list = iterate_list[starting_idx:]
    if args.parallel is True:
        logger.info("Begin using data parallel")
    devices = args.devices if args.parallel is False else ["cuda"] # TODO: Refactor this. 
    logger.info("Devices: {}".format(devices))
    divided_list = divide_list_equally(iterate_list, len(devices))
    processes = []
    #mp.set_start_method('spawn')
    if args.parallel is False:
        ctx = mp.get_context("spawn")
        for idx, device in enumerate(devices):
            #nlp = spacy.load(args.spacy_model)
            wrapper_model, tokenizer = utils.load_opensource_model(args.model, args.model_path, args.parallel)
            template_processor = utils.TemplatePrcessor(args.model, args.disable_template, tokenizer=tokenizer)
            if args.mode == "code":
                mode_specific_args["gen_kwargs"] = utils.generate_gen_kwargs(True, TEMPERATURE, TOPP, top_k=TOPK, max_length=MAX_CODE_LENGTH, task=task, tokenizer=tokenizer)
            #logger.info(mode_specific_args)
            wrapper_model.model.to(device)
            #else:
            #    wrapper_model.model = torch.nn.DataParallel(wrapper_model.model)
            print_threshold = len(divided_list[idx]) // args.print_freq
            logger.info(f"Running on device {device}, print threshold: {print_threshold}")
            save_path = os.path.join(temp_dir, str(idx))
            temp_dir_list.append(save_path)
            if len(devices) == 1:
                main_evaluation_func(wrapper_model, args.mode, dataset, args.per_sample_num,
                                                        divided_list[idx], save_path, str(device), 
                                                        args.log_file, print_threshold, mode_specific_args, template_processor)
            else:
                process = ctx.Process(target=main_evaluation_func, 
                                                args=(wrapper_model, args.mode, dataset, args.per_sample_num,
                                                        divided_list[idx], save_path, str(device), 
                                                        args.log_file, print_threshold, mode_specific_args, template_processor))
                process.start()
                processes.append(process)

        if len(devices) > 1:
            for process in processes:
                process.join()
    else:
        wrapper_model, tokenizer = utils.load_opensource_model(args.model, args.model_path, args.parallel)
        template_processor = utils.TemplatePrcessor(args.model, args.disable_template, tokenizer=tokenizer)
        if args.mode == "code":
            mode_specific_args["gen_kwargs"] = utils.generate_gen_kwargs(True, TEMPERATURE, TOPP, top_k=TOPK, max_length=MAX_CODE_LENGTH, task=task, tokenizer=tokenizer)
        print_threshold = len(iterate_list) // args.print_freq
        logger.info(f"Running on device {wrapper_model.model.device}, print threshold: {print_threshold}")
        save_path = os.path.join(temp_dir, str(0))
        temp_dir_list.append(save_path)
        main_evaluation_func(wrapper_model, args.mode, dataset, args.per_sample_num, iterate_list,
                             save_path, wrapper_model.model.device, args.log_file,
                             print_threshold, mode_specific_args, template_processor)
    final_result = shelve.open(args.output_path)
    if args.resume_dir:
        temp_dir_list.append(resume_path)
    for temp_save in temp_dir_list:
        temp = shelve.open(temp_save)
        for key in temp:
            final_result[key] = temp[key]
        temp.close()
    final_result.close()
    
    logger.info("Processing finished, saving to: {}".format(args.output_path))
    time.sleep(5) # wait for the file to be closed
    shutil.rmtree(temp_dir)
    file_handler.close()
