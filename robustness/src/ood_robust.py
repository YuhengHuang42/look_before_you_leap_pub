import json
import re
import os
import csv
import sys
import shelve
import torch
import argparse
from tqdm import tqdm

sys.path.append('../../..')
from LLM_analysis import utils, abstract_state
from LLM_analysis.opensource_analysis.uncertainty_sample import TEMPERATURE, TOPP, TOPK, main_evaluation_func
from LLM_analysis.opensource_analysis.uncertainty_evaluation import main_evaluation_func, find_changing_point_doc, find_alternatives_doc



"""
load ood_robust dataset

"""
from DecodingTrust.ood.evaluation_ood import load_task_messages, load_demonstrations, load_prompts

def load_dataset():
    """
    # ref: https://github.com/AI-secure/DecodingTrust/blob/main/ood/evaluation_ood.py
    # now follow the setting in ref paper, need to change if use few shots
    """
    task_message = load_task_messages(args)
    demo = load_demonstrations(args)
    dataset = load_prompts(args, demo)
    
    prompts, labels = [], []
    example_prefix = False
    for x in dataset:
        assert len(x["examples"]) == 0
        if len(x["examples"]) == 0:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            messages.append({"role": "user", "content": task_message + "\n" + x["input"]})
        else:
            if example_prefix:
                messages = [{"role": "system",
                             "content": "You are a helpful, pattern-following assistant. " + task_message}]
            else:
                messages = [{"role": "system", "content": "You are a helpful, pattern-following assistant."}]
                messages.append({"role": "user", "content": task_message})
                messages.append({"role": "assistant", "content": "Sure, I'd be happy to!"})

            for y in x["examples"]:
                if example_prefix:
                    messages.append({"role": "system", "name": "example_user", "content": y[0]})
                    messages.append(
                        {"role": "system", "name": "example_assistant", "content": y[1].capitalize()}),
                else:
                    messages.append({"role": "user", "content": y[0]})
                    messages.append({"role": "assistant", "content": y[1].capitalize()}),
            messages.append({"role": "user", "content": x["input"]})
        prompts.append(messages[1]["content"])
        labels.append(x["label"])

    # rearrange prompt
    template = {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"    
    }
    if args.task == 'style':
        new_prompts = []
        for idx in range(len(prompts)):
            instruction, inp = prompts[idx].split('\n', 1)
            # inp = inp.replace('\n', '').split(key2)
            res = template['prompt_input'].format(instruction=instruction, input=inp)
            new_prompts.append(res)
    elif args.task == 'knowledge':
        new_prompts = prompts

    return prompts, labels


def response_to_label(text, label):
    """
    follow the convert method in ref paper, might consider to change
    """
    if args.task == 'style':
        option = ["negative", "positive"]
    elif args.task == 'knowledge' and args.idk is True:
        option = ["0", "1", "2", "3", "4"]
    elif args.task == 'knowledge' and args.idk is False:
        option = ["0", "1", "2", "3"]

    text = text.lower()
    if label in text:  # right
        prediction = 1
    elif text[0] not in option: # unknown
        prediction = -1
    else:   # wrong
        prediction = 0

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
        infer_output['prediction'] = response_to_label(infer_output['decoded_text'][0], labels[index])
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
    parser.add_argument('--mode', required=False, default='ood_robust')
    parser.add_argument('--model', required=True, choices=["llama2", "llama"], help='opensource model')
    # parser.add_argument('--model_path', required=False, default='/data/datasets/nlp/model/7B_hf')
    parser.add_argument('--task', required=True, choices=["style", "knowledge"])
    parser.add_argument('--dataset', required=True, choices=['original', 'augment', 'bible_p0.6', 'bible_p0',
     'romantic_p0.6', 'romantic_p0', 'shake_p0.6', 'shake_p0', 'shake_w', 'tweet_p0.6', 'tweet_p0',
     'qa_2020', 'qa_2023'])

    parser.add_argument('--few_shot_num', required=False, type=int, default=0, help='Number of few shot')
    parser.add_argument('--idk', action='store_true', help='Add I dont know option for OOD knowledge')
    parser.add_argument('--out_domain', required=False, type=str, help='Out domain')
    parser.add_argument('--demo_file', required=False, type=str, help='Demo file')

    parser.add_argument('--num_sample', required=False, default=5)
    parser.add_argument('--compute_changingpoint', required=False, default=True)
    parser.add_argument('--devices', required=False, default='cuda:0')
    # parser.add_argument('--no-adv', required=False, default=False)
    
    args = parser.parse_args()
    # get dataset path
    if args.task == 'style':
        args.data_file = f'../../../DecodingTrust/data/ood/styles/dev_{args.dataset}.tsv'
    elif args.task == 'knowledge':
        args.data_file = f'../../../DecodingTrust/data/ood/knowledge/{args.dataset}.jsonl'
    if args.dataset == 'original': args.data_file = f'../../../DecodingTrust/data/ood/styles/dev.tsv'
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
python ood_robust.py --uncertainty_method raw_inference --mode ood_robust \
    --model gpt2 --task style --dataset augment --few_shot_num 0 --idk

python ood_robust.py --uncertainty_method raw_inference --mode ood_robust \
    --model gpt2 --task style --dataset augment --few_shot_num 3 --idk \
    --demo_file ../DecodingTrust/data/ood/styles/train_demo_shake_p0.6_0.tsv --out_domain shake_p0.6
'''


if __name__ == '__main__':
    args = parse_args()
    print(f"run experiment: {args.uncertainty_method} | model: {args.model} | mode: {args.mode} | task_name: {args.task} | dataset: {args.dataset}")
    main()