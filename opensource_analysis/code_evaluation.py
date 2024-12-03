import argparse
import shelve
import torch
import pandas as pd
import logging
import os
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append('../')
import abstract_state
import utils

'''
python llm_evaluation.py --output_path /data/huangyuheng/QA/gpt2/wiki/greedy/raw_inference \
    --dataset_path /data/huangyuheng/dataset/llm/wiki_qa_test_new.csv

nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/incoder/humaneval/greedy/raw_inference \
    --llm_device "cuda:0" --dataset humaneval --model incoder \
    --log_file log/code_incoder_humaneval.log > code_incoder_humaneval_error.log 2>&1 &

nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/incoder/mbpp/greedy/raw_inference \
    --llm_device "cuda:0" --dataset mbpp --model incoder \
    --log_file log/code_incoder_mbpp.log > code_incoder_mbpp_error.log 2>&1 &

nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/codegen/mbpp/greedy/raw_inference \
    --llm_device "cuda:0" --dataset mbpp --model codegen \
    --log_file log/code_codegen_mbpp.log > code_codegen_mbpp_error.log 2>&1 &

nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/codegen/humaneval/greedy/raw_inference \
    --llm_device "cuda:0" --dataset humaneval --model codegen \
    --log_file log/code_codegen_humaneval.log > code_codegen_humaneval_error.log 2>&1 &
    
nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/santacoder/mbpp/greedy/raw_inference \
    --llm_device "cuda:0" --dataset mbpp --model santacoder \
    --log_file log/code_santacoder_mbpp.log > code_santacoder_mbpp_error.log 2>&1 &

nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/santacoder/humaneval/greedy/raw_inference \
    --llm_device "cuda:0" --dataset humaneval --model santacoder \
    --log_file log/code_santacoder_humaneval.log > code_santacoder_humaneval_error.log 2>&1 &

nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/codellama/humaneval/greedy/raw_inference \
    --llm_device "cuda:0" --dataset humaneval --model codellama \
    --log_file log/code_codellama_humaneval.log > code_codellama_humaneval_error.log 2>&1 &
    
nohup python -u code_evaluation.py --output_path /data/huangyuheng/code/codellama/mbpp/greedy/raw_inference \
    --llm_device "cuda:1" --dataset mbpp --model codellama \
    --log_file log/code_codellama_mbpp.log > code_codellama_mbpp_error.log 2>&1 &
'''

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="codegen", choices=
                        ["codegen", "santacoder", "incoder", 
                         "codellama", "llama3", "deepseekcoder", 
                         "codeqwen1.5", "starcoder2"
                         ]
                        )
    parser.add_argument("--llm_device", type=str, default="cuda:0")
   # parser.add_argument("--eval_device", type=str, default="cuda:1")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--simplified", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default="mbpp", choices=["mbpp", "humaneval"])
    parser.add_argument("--print_freq", type=int, default=10, help="Every length/n evaluation will be printed")
    parser.add_argument("--log_file", type=str, default="llm_evaluation.log")
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
    if args.model == "codegen":
        logger.info("Using codegen model")
    elif args.model == "santacoder":
        logger.info("Using santacoder model")
    elif args.model == "incoder":
        logger.info("Using incoder model")
    elif args.model == "codellama":
        logger.info("Using CodeLlama-7b-Python")
    else:
        raise Exception("Model not supported")
    '''
    
    wrapper_model, tokenizer = utils.load_opensource_model(args.model, parallel=args.parallel)
    if args.parallel == False:
        wrapper_model.model.to(args.llm_device)
        
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
    mode_specific_args["max_length"] = 650 # Refer to https://github.com/bigcode-project/bigcode-evaluation-harness/tree/3c8c685b6c162f034e7e0215b19cb75917ba6672
    mode_specific_args["postprocess"] = True
    mode_specific_args["batch_size"] = 1
    if args.model == "incoder":
        mode_specific_args["prefix"] = "<| file ext=.py |>\n"
    elif args.model == "codegen":
        mode_specific_args["prefix"] = "# Import libraries.\n import numpy as np"
    mode_specific_args["gen_kwargs"] = utils.generate_gen_kwargs(False, 0, 0.95, top_k=0, max_length=mode_specific_args["max_length"], task=task, tokenizer=tokenizer)
        
    save_path = args.output_path
    ans_recored = shelve.open(save_path)
    print_threshold = dataset_len // args.print_freq
    for idx in range(dataset_len):
        mode_specific_args["sample_idx"] = idx
        prompt = task.get_prompt(dataset[idx])
        reference = references[idx]
        infer_output = wrapper_model.forward(prompt, mode="code", mode_specific_args=mode_specific_args)
        #text_output = task.postprocess_generation(infer_output['decoded_text'][0], idx)
        text_output = infer_output['final_result']
        infer_output["evaluation"] = task.process_results([text_output], [reference])
        if args.simplified:
            infer_output = abstract_state.get_topk_token_output(infer_output, tokenizer)
            infer_output["entropy"] = utils.compute_entropy(infer_output['scores'])
            infer_output.pop('scores')
        infer_output['evaluation']['input'] = prompt
        infer_output['evaluation']['gt'] = reference
        ans_recored[str(idx)] = infer_output
        if idx % print_threshold == 0:
            logger.info(f"Processed {idx}/{dataset_len} samples")
    
    ans_recored.close()
    logger.info("Processing finished, saving to: {}".format(save_path))
    file_handler.close()
        
    
    
