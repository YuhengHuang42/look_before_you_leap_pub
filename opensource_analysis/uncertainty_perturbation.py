import pickle
import tqdm
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2TokenizerFast
import spacy
import shelve
#from sqlitedict import SqliteDict
import pandas as pd
import logging
import os
#import multiprocessing
import torch.multiprocessing as mp
import torch
import shutil
import time

import sys
sys.path.append("../")
import abstract_state
from utils import compute_entropy, LLMInference, divide_list_equally
import utils

from uncertainty_sample import MAX_CODE_LENGTH, TOPP, TOPK

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_diff_two_gram(entropy_list):
    """
    Iterate through the states and find the difference between two consecutive states.
    """
    recorder = list()
    state_num = len(entropy_list)
    for idx, state in enumerate(entropy_list):
        if idx == (state_num - 1):
            continue
        recorder.append([(idx, idx+1), (entropy_list[idx + 1] - entropy_list[idx])])
    return recorder

def find_changing_point_sentence(so):
    """
    Return [sentence_id, token_in_sentence_id]
    """
    result = list()
    for idx, sentence_state in enumerate(so.sentence_state):
        pair = sorted(find_diff_two_gram(so.sentence_state[idx]), key=lambda x:x[1])[-1][0]
        result.append((idx, pair[1]))
    return result

def find_changing_point_doc(entropy_list, topk=1):
    """
    Return [token_id], [Diff_in_entropy]
    """
    pairs = sorted(find_diff_two_gram(entropy_list), key=lambda x:x[1])[-topk:]
    return [i[0][1] for i in pairs], [i[1].item() for i in pairs]

# Find one position alternatives instead of multiple


def find_alternatives_sentence(model, original_info, changing_points: list[tuple[int, int]], so, question: str):
    """
    Find alternatives for each changing point at the sentence level.
    """
    result = dict()
    for point in changing_points:
        cur_token_idx = so.sentence_state[point[0]][point[1]].idx
        result[cur_token_idx] = list()
        prefix = flatten_list(original_info['decoded_word'][0][:cur_token_idx])
        cur_input = question + "".join(prefix)
        token_alter = original_info['llm_info'][cur_token_idx]['top_k_token']
        for token in token_alter[1:]: # Exclude the highest one.
            new_output = model.forward(cur_input + token)
            new_output['input'] = cur_input + token
            new_output["entropy"] = utils.compute_entropy(new_output['scores'])
            new_output.pop("scores")
            result[cur_token_idx].append(new_output)
    return result

def find_alternatives_doc(model, original_info, changing_points: list[int], 
                          question: str, mode_specific_args: dict, mode: str="nlp",
                          topk=4, model_type="gpt2"):
    """
    Find alternatives for each changing point at the document level.
    """
    result = dict()
    for point in changing_points:
        result[point] = list()
        prefix = flatten_list(original_info['decoded_word'][0][:point])
        if model_type == "llama" or model_type == "llama2" or model_type == "codellama":
            space_split = " "
        else:
            space_split = "" # GPT-2 has space in words. Code model does not need extra space.
        process_prefix = space_split.join(prefix)
        cur_input = question + process_prefix
        token_alter = original_info['llm_info'][point]['top_k_token']
        for token in token_alter[1:][:topk]:
            new_output = model.forward(cur_input + token, mode_specific_args=mode_specific_args, mode=mode) # It might be OK to not add the space_split (Byte-Pair Encoding)
            new_output['input'] = cur_input + space_split + token 
            new_output["entropy"] = utils.compute_entropy(new_output['scores'])
            new_output.pop("scores")
            result[point].append(new_output)
    return result

def main_evaluation_func(raw_data, wrapper_model, 
                         iterate_list, save_path, device, logger_file, 
                         print_threshold, mode_specific_args, compute_changingpoint=True,
                         mode="nlp", model_type="gpt2"):
    logger, file_handler = utils.setupLogging(logger_file, multiprocess=True)
    ans_recored = shelve.open(save_path)
    for id, index in enumerate(tqdm.tqdm(iterate_list)):
        # state construction
        question = raw_data[index]['evaluation']['input']
        model_output = raw_data[index]
        #mode_specific_info = {"tokenizer": tokenizer, "model_output": model_output}
        #states = abstract_state.get_model_state(nlp, "opensource", mode_specific_info)
        #if states[0] is None:
        #    raise Exception
        #so = abstract_state.StateOverview(states, 4, ("NOUN", "VERB"), 0.3, "sentence", 
        #                                nlp, disable_alter_comp=True, disable_child_comp=True)
        #max_entropy_states = max(states, key=lambda x:x.entropy)
        #min_entropy_states = min(states, key=lambda x:x.entropy)
        entropy_list = model_output['entropy']
        max_entropy_idx = torch.argmax(entropy_list).item()
        min_entropy_idx = torch.argmin(entropy_list).item()
        if compute_changingpoint:
            changing_points = find_changing_point_doc(entropy_list, 1)
        else:
            changing_points = [[], []]
        target_points = [max_entropy_idx, min_entropy_idx] + changing_points[0]
        target_points = list(set(target_points))
        if mode == "code":
            mode_specific_args["sample_idx"] = int(index)
        alter_result = find_alternatives_doc(wrapper_model, model_output, 
                                             target_points, question, mode_specific_args, mode=mode, model_type=model_type)
        alter_result['diff_entropy'] = changing_points
        alter_result['entropy'] = {"max": (max_entropy_idx, entropy_list[max_entropy_idx].item()), 
                                   "min": (min_entropy_idx, entropy_list[min_entropy_idx].item())}
        ans_recored[index] = alter_result
        if id % print_threshold == 0:
            logging_str = f"In device:{str(device)}, Processed {str(id)}/{str(len(iterate_list))} samples"
            logger.info(logging_str)
    ans_recored.close()

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

'''
nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/gpt2/wiki/greedy/raw_inference" --devices "cuda:0" \
    --output_path "/data/huangyuheng/QA/uncertainty/gpt2/wiki_alter" --mode qa \
        --log_file "log/uncertainty_wiki_alter.log"--model gpt2  > uncertainty_wiki_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/gpt2/shortQA/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/shortQA_alter" \
        --log_file "log/uncertainty_shortQA_alter.log" > uncertainty_shortQA_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/gpt2/shortQA/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/shortQA_alter_new" --compute_changingpoint False \
        --log_file "log/uncertainty_shortQA_alter_new.log" > uncertainty_shortQA_alter_new_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/gpt2/cnn_daily/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/gpt2_cnn_daily_alter"  --mode summarization \
        --log_file "log/uncertainty_gpt2_cnn_daily.log" > uncertainty_gpt2_cnn_daily_error.log 2>&1 &

python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/gpt2/wmt14_test/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/gpt2/wmt14_alter"  --mode translation \
    --model gpt2 --log_file "log/uncertainty_gpt2_wmt14_alter.log"

python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/gpt2/eli5_category/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/gpt2/eli5_alter"  --mode qa \
    --model gpt2 --log_file "log/uncertainty_gpt2_eli5_alter.log"

# ======
nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama/cnn_daily/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama_cnn_daily_alter"  --mode summarization --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --model llama --log_file "log/uncertainty_llama_cnn_daily.log" > uncertainty_llama_cnn_daily_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama/eli5_category/greedy/raw_inference" --devices "cuda:0" "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama/eli5_category_alter"  --mode qa --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --model llama --log_file "log/uncertainty_llama_eli5_category_alter.log" > uncertainty_llama_eli5_category_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama/wiki/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama/wiki_alter"  --mode qa --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --model llama --log_file "log/uncertainty_llama_wiki_alter.log" > uncertainty_llama_wiki_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/QA/llama/wmt14_test/greedy/raw_inference" --devices "cuda:1" "cuda:2" "cuda:3" \
    --output_path "/data/huangyuheng/QA/uncertainty/llama/wmt14_alter"  --mode translation --model_path /home/huangyuheng/LLM/cache/7B_hf/ \
    --model llama --log_file "log/uncertainty_llama_wmt14_alter.log" > uncertainty_llama_wmt14_alter_error.log 2>&1 &
'''

# ==== LLAMA2
"""
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
"""

"""
python3 -u uncertainty_perturbation.py --input_data "/home/huangyuheng/LLM_analysis/data/code/codegen/humaneval/greedy" \
    --model codegen --output_path /home/huangyuheng/LLM_analysis/opensource_analysis/result/code/codegen/codegen_humaneval_alter \
    --log_file log/codegen_humaneval_alter.log --mode code --parallel True --dataset humaneval

python3 -u uncertainty_perturbation.py --input_data "/home/huangyuheng/LLM_analysis/data/code/santacoder/humaneval/greedy" \
    --model santacoder --output_path /home/huangyuheng/LLM_analysis/opensource_analysis/result/code/santacoder/santacoder_humaneval_alter \
    --log_file log/santacoder_humaneval_alter.log --mode code --devices "cuda:0" "cuda:1" --dataset humaneval

python3 -u uncertainty_perturbation.py --input_data "/home/huangyuheng/LLM_analysis/data/code/incoder/humaneval/greedy" \
    --model incoder --output_path /home/huangyuheng/LLM_analysis/opensource_analysis/result/code/incoder/incoder_humaneval_alter \
    --log_file log/incoder_humaneval_alter.log --mode code --parallel True --dataset humaneval
"""

# ==== codellama
"""
nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/code/codellama/humaneval/greedy/raw_inference" \
        --model codellama --output_path /data/huangyuheng/code/uncertainty/result/perturbation/codellama_humaneval_alter \
        --log_file log/codellama_humaneval_alter.log --mode code --devices "cuda:0" --dataset humaneval > uncertainty_codellama_humaneval_alter_error.log 2>&1 &

nohup python3 -u uncertainty_perturbation.py --input_data "/data/huangyuheng/code/codellama/mbpp/greedy/raw_inference" \
        --model codellama --output_path /data/huangyuheng/code/uncertainty/result/perturbation/codellama_mbpp_alter \
        --log_file log/codellama_mbpp_alter.log --mode code --devices "cuda:1" --dataset mbpp > uncertainty_codellama_mbpp_alter_error.log 2>&1 &
        
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="The evaluation data returned by llm_evaluation.py")
    parser.add_argument('--devices', nargs='*', type=str, help='List of Possible devices')
    #parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--model", type=str, default="santacoder", 
                        choices=["codegen", "santacoder", 
                                 "incoder", "gpt2", "llama", "llama2", "codellama",
                                 "llama3", "phi3", "gemma2", "deepseekcoder", 
                                 "codeqwen1.5", "starcoder2"]
                        )
    parser.add_argument("--model_path", type=str, default=None, help="path to the model. Only used for LLAMA")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--only_log_this", default=False, action="store_true", 
                        help="Only log the evaluation of this file")
    parser.add_argument("--print_freq", type=int, default=10, help="Every length/n evaluation will be printed")
    parser.add_argument("--log_file", type=str, default="llm_evaluation.log")
    parser.add_argument("--compute_changingpoint", type=bool, default=True)
    parser.add_argument("--mode", type=str, default='qa', choices=["qa", "summarization", "translation", "code"])
    parser.add_argument("--parallel", default=False, type=bool, help="whether to use all GPUs for a single model")
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["mbpp", "humaneval"]) # Only used in code mode
    parser.add_argument('--resume_dir', type=str, default=None)
    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Setup logging
    if args.only_log_this:
        file_filter = utils.FileFilter(os.path.basename(__file__))
    else:
        file_filter = None
    logger, file_handler = utils.setupLogging(args.log_file, file_filter, multiprocess=True)
    logger.info("Running %s" % " ".join(sys.argv))
    
    # Load data
    if args.input_data.endswith("pickle"):
        with open(args.input_data, "rb") as ifile:
            raw_data = pickle.loads(ifile.read())
    else:
        temp_raw_data = shelve.open(args.input_data)
        raw_data = dict(temp_raw_data)
        temp_raw_data.close()

    # Load model
    logger.info(f"Using {args.model} model")

    save_path = args.output_path
    assert not save_path.endswith("/")
    temp_dir = save_path + "_temp"
    temp_dir_list = list()
    os.makedirs(temp_dir, exist_ok=True)
    iterate_list = [str(i) for i in raw_data]
    assert len(iterate_list) > 0
    if args.parallel is True:
        logger.info("Begin using data parallel")
    if args.resume_dir is not None:
        assert args.resume_dir != temp_dir
        starting_idx, resume_path = utils.load_shelve_and_resume(args.resume_dir)
        iterate_list = iterate_list[starting_idx:]
    devices = args.devices if args.parallel is False else ["cuda"] # TODO: Refactor this. 
    divided_list = divide_list_equally(iterate_list, len(devices))
    processes = []

    if args.mode == "qa":
        mode_specific_args = {"max_new_tokens": 100}
        mode = "nlp"
    elif args.mode == "summarization" or args.mode == "translation":
        mode_specific_args = {"max_new_tokens": 100}
        mode = "nlp"
    elif args.mode == "code":
        mode = "code"
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
        
    if args.parallel is False:
        ctx = mp.get_context("spawn")
        for idx, device in enumerate(args.devices):
            #nlp = spacy.load(args.spacy_model)
            wrapper_model, tokenizer = utils.load_opensource_model(args.model, args.model_path)
            if args.mode == "code":
                mode_specific_args["gen_kwargs"] = utils.generate_gen_kwargs(False, 0, TOPP, top_k=TOPK, 
                                                                             max_length=MAX_CODE_LENGTH, task=task, tokenizer=tokenizer)
            wrapper_model.model.to(device)
            print_threshold = len(divided_list[idx]) // args.print_freq
            logger.info(f"Running on device {device}, print threshold: {print_threshold}")
            save_path = os.path.join(temp_dir, str(idx))
            temp_dir_list.append(save_path)
            process = ctx.Process(target=main_evaluation_func, 
                                            args=(raw_data, wrapper_model, 
                                                    divided_list[idx], save_path, str(device), 
                                                    args.log_file, print_threshold, mode_specific_args,
                                                    args.compute_changingpoint, mode, args.model))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    else:
        wrapper_model, tokenizer = utils.load_opensource_model(args.model, args.model_path, args.parallel)
        print_threshold = len(iterate_list) // args.print_freq
        logger.info(f"Running on device {wrapper_model.model.device}, print threshold: {print_threshold}")
        save_path = os.path.join(temp_dir, str(0))
        temp_dir_list.append(save_path)
        if args.mode == "code":
            mode_specific_args["gen_kwargs"] = utils.generate_gen_kwargs(False, 0, TOPP, top_k=TOPK, 
                                                                            max_length=MAX_CODE_LENGTH, task=task, tokenizer=tokenizer)
        main_evaluation_func(raw_data, wrapper_model, iterate_list, save_path, wrapper_model.model.device,
                            args.log_file, print_threshold, mode_specific_args,
                            args.compute_changingpoint, mode=mode, model_type=args.model)
    # Merge the result
    if args.resume_dir:
        temp_dir_list.append(resume_path)
    final_result = shelve.open(args.output_path)
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