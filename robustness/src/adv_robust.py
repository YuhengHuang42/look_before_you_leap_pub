import json
import re
import os
import sys
import shelve
import torch
import argparse
from tqdm import tqdm
import pandas as pd

# sys.path.append('../../../DecodingTrust')
sys.path.append('../../..')
from LLM_analysis import utils, abstract_state
from LLM_analysis.opensource_analysis.uncertainty_sample import TEMPERATURE, TOPP, TOPK, main_evaluation_func
from LLM_analysis.opensource_analysis.uncertainty_evaluation import main_evaluation_func, find_changing_point_doc, find_alternatives_doc


"""
load adv_robust dataset
"""
from DecodingTrust.adv_glue_plus_plus.gpt_eval import tasks, task_to_keys, TASK2SHORTPROMPT, answer_mapping
TASK2SHORTPROMPT["mnli-mm"] = TASK2SHORTPROMPT["mnli"]

# load dataset
# ref: https://github.com/AI-secure/DecodingTrust/blob/main/adv-glue-plus-plus/gpt_eval.py#181
# task = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']
def load_dataset():
    with open(args.dataset_path) as f:
        datasets = json.load(f)
    assert args.task in tasks
    dataset = datasets[args.task]
    # results = {task: {"requests": [], "responses": [], "predictions": [], "labels": []}}
    prompts, labels = [], []
    for data in dataset:
        key1, key2 = task_to_keys[args.task]
        if args.dataset == 'original':
            if f"original_{key1}" in data:
                key1 = f"original_{key1}"
            elif f"original_{key2}" in data:
                key2 = f"original_{key2}"
            else:
                continue
        label = data['label']
        if data["method"] == "semattack":
            message = data["sentence"]
        else:
            message = f"{key1.replace('original_', '')}: {data[key1]}"
            if key2:
                message = f"{message}\n{key2.replace('original_', '')}: {data[key2]}"
        message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'Hypothesis')
        prompt = f"{TASK2SHORTPROMPT[args.task]}\n{message}"
        labels.append(label)
        prompts.append(prompt)

    # rearrange prompt
    template = {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"    
    }
    new_prompts = []
    for idx in range(len(prompts)):
        instruction, inp = prompts[idx].split('\n', 1)
        # inp = inp.replace('\n', '').split(key2)
        res = template['prompt_input'].format(instruction=instruction, input=inp)
        new_prompts.append(res)

    return new_prompts, labels


# convert score to prediction
# ref: https://github.com/AI-secure/DecodingTrust/blob/main/adv-glue-plus-plus/gpt_eval.py#L81
def response_to_labels(response, task):
    response = response.lower()
    # remove all text enclosed within ""
    for match in re.findall("\"(.*?)\"", response):
        response = response.replace(match, "")
    answer = response.split(".")[0]
    mapping = answer_mapping[task]

    count = 0
    prediction = -1
    for k, v in mapping.items():
        if k in answer:
            prediction = v
            count += 1
    if count != 1:
        # print(response, file=sys.stderr)
        prediction = -1

    return prediction


def raw_inference(wrapper_model, tokenizer, prompts, labels, iterate_list):
    mode_specific_args = {}
    mode_specific_args["max_new_tokens"] = 10

    ans_recored = shelve.open(args.output_path)
    resume = len(ans_recored)
    for index, prompt in enumerate(prompts[resume:]):
        if index%100 == 0: print(f"{index+1+resume}/{len(iterate_list)}")
        infer_output = wrapper_model.forward(prompt, mode_specific_args=mode_specific_args)
        infer_output = abstract_state.get_topk_token_output(infer_output, tokenizer)
        infer_output["entropy"] = utils.compute_entropy(infer_output['scores'])
        infer_output.pop('scores')
        infer_output["label"] = labels[index]
        infer_output["prompt"] = prompts[index]
        infer_output['prediction'] = response_to_labels(infer_output['decoded_text'][0], args.task)
        ans_recored[str(index)] = infer_output
    ans_recored.close()


def perturbation_uncertainty(wrapper_model, tokenizer, prompts, iterate_list):
    mode_specific_args = {}
    mode_specific_args["max_new_tokens"] = 10

    # load raw data
    temp_raw_data = shelve.open(args.raw_data_path)
    raw_data = dict(temp_raw_data)
    temp_raw_data.close()
    assert len(raw_data) == len(iterate_list)

    ans_recored = shelve.open(args.output_path)
    resume = len(ans_recored)
    for id, index in enumerate(iterate_list[resume:]):
        if id%100 == 0: print(f"{id+1+resume}/{len(iterate_list)}")
        prompt = prompts[index]
        model_output = raw_data[index]
        
        entropy_list = model_output['entropy']
        max_entropy_idx = torch.argmax(entropy_list).item()
        min_entropy_idx = torch.argmin(entropy_list).item()
        if args.compute_changingpoint:
            changing_points = find_changing_point_doc(entropy_list, 1)
        else:
            changing_points = [[], []]
        target_points = [max_entropy_idx, min_entropy_idx] + changing_points[0]
        target_points = list(set(target_points))
        
        alter_result = find_alternatives_doc(wrapper_model, model_output, target_points, prompt, mode_specific_args)
        alter_result['diff_entropy'] = changing_points
        alter_result['entropy'] = {"max": (max_entropy_idx, entropy_list[max_entropy_idx].item()), 
                                   "min": (min_entropy_idx, entropy_list[min_entropy_idx].item())}

        ans_recored[index] = alter_result
    ans_recored.close()


def sample_uncertainty(wrapper_model, tokenizer, prompts, iterate_list):
    mode_specific_args = {}
    mode_specific_args["max_new_tokens"] = 10
    mode_specific_args["temperature"] = TEMPERATURE
    mode_specific_args["do_sample"] = True
    mode_specific_args["top_p"] = TOPP
    mode_specific_args["top_k"] = TOPK

    ans_recored = shelve.open(args.output_path)
    resume = len(ans_recored)
    for id, index in enumerate(iterate_list[resume:]):
        if index%100 == 0: print(f"{index+1+resume}/{len(iterate_list)}")
        prompt = prompts[index]
        single_entry = []
        for _ in range(args.num_sample):
            infer_output = wrapper_model.forward(prompt, mode_specific_args=mode_specific_args)
            # print(infer_output['decoded_text'])
            infer_output["entropy"] = utils.compute_entropy(infer_output['scores'])
            infer_output.pop('scores')
            single_entry.append(infer_output)
        ans_recored[str(index)] = single_entry
    ans_recored.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncertainty_method', required=True, choices=['raw_inference', 'sample_uncertainty'])
    parser.add_argument('--mode', required=False, default='adv_robust')
    parser.add_argument('--model', required=True, choices=["llama2", "llama"], help='opensource model')
    # parser.add_argument('--model_path', required=False, default='/data/datasets/nlp/model/7B_hf')
    parser.add_argument('--task', required=True, choices=['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte'])
    parser.add_argument('--dataset', required=True, choices=['alpaca', 'stable-vicuna', 'vicuna', 'original'])
    
    parser.add_argument('--num_sample', required=False, default=5)
    parser.add_argument('--compute_changingpoint', required=False, default=True)
    parser.add_argument('--devices', required=False, default='cuda:0')
    # parser.add_argument('--no-adv', required=False, default=False)
    
    args = parser.parse_args()
    # get dataset path
    args.dataset_path = f'../../../DecodingTrust/data/adv-glue-plus-plus/data/{args.dataset}.json'
    if args.dataset == 'original': args.dataset_path = f'../../../DecodingTrust/data/adv-glue-plus-plus/data/alpaca.json'
    # get output path
    args.raw_data_path = f'../result/{args.mode}/{args.dataset}/{args.task}/{args.model}/raw_inference'
    args.output_path = f'../result/{args.mode}/{args.dataset}/{args.task}/{args.model}/{args.uncertainty_method}'
    # get model path
    if args.model == 'llama':
        args.model_path = '/data/datasets/nlp/model/7B_hf'
    elif args.model == 'llama2':
        args.model_path = '/data/zhaoshengming/llama2-hf'

    # create dir
    def create_directories_if_not_exist(file_path):
        directory_path = os.path.dirname(file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    create_directories_if_not_exist(args.output_path)

    return args


def main():
    prompts, labels = load_dataset()
    iterate_list = [i for i in range(len(prompts))]

    # resume result
    ans_recored = shelve.open(args.output_path)
    if len(ans_recored) == len(prompts):
        print('experiment has already done')
        ans_recored.close()
        return
    ans_recored.close()

    wrapper_model, tokenizer = utils.load_opensource_model(args.model, args.model_path)
    wrapper_model.model.to(args.devices)

    if args.uncertainty_method == 'raw_inference':
        raw_inference(wrapper_model, tokenizer, prompts, labels, iterate_list)
    # elif args.uncertainty_method == 'perturbation_uncertainty':
    #     perturbation_uncertainty(wrapper_model, tokenizer, prompts, iterate_list)
    elif args.uncertainty_method == 'sample_uncertainty':
        sample_uncertainty(wrapper_model, tokenizer, prompts, iterate_list)


'''
python adv_robust.py --uncertainty_method raw_inference --mode adv_robust \
    --model gpt2 --task sst2 --dataset alpaca
'''


if __name__ == '__main__':
    args = parse_args()
    print(f"run experiment: {args.uncertainty_method} | model: {args.model} | mode: {args.mode} | task: {args.task} | dataset: {args.dataset}")
    main()