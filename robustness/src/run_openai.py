import json
import re
import os
import sys
import shelve
import torch
import random
import argparse
from tqdm import tqdm
import pandas as pd

sys.path.append('../../..')
from LLM_analysis.openAI_anallysis.GPT3_utils import get_feature, get_response, get_multi_sample_response, get_pertu_response


"""
load adv_robust & ood_robust dataset

"""
from DecodingTrust.adv_glue_plus_plus.gpt_eval import tasks, task_to_keys, TASK2SHORTPROMPT, answer_mapping
from DecodingTrust.ood.evaluation_ood import load_task_messages, load_demonstrations, load_prompts
TASK2SHORTPROMPT["mnli-mm"] = TASK2SHORTPROMPT["mnli"]

# load dataset
# ref: https://github.com/AI-secure/DecodingTrust/blob/main/adv-glue-plus-plus/gpt_eval.py#181
# NO_ADV = False
# dataset_path = '../DecodingTrust/data/adv-glue-plus-plus/data/alpaca.json'
# dataset_path = '../DecodingTrust/data/adv-glue-plus-plus/data/stable-vicuna.json'
# dataset_path = '../DecodingTrust/data/adv-glue-plus-plus/data/vicuna.json'
# task = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']
def load_dataset():
    assert args.mode in ['adv_robust', 'ood_robust']
    if args.mode == 'adv_robust':
        with open(args.data_file) as f:
            datasets = json.load(f)
        assert args.task in tasks
        dataset = datasets[args.task]
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

        return prompts, labels

    elif args.mode == 'ood_robust':
        """
        # ref: https://github.com/AI-secure/DecodingTrust/blob/main/ood/evaluation_ood.py
        # now follow the setting in ref paper, need to change if use few-shots
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

        return prompts, labels


# convert score to prediction
# basically, convert unknown answer to wrong
def response_to_label(response, label):
    assert args.mode in ['adv_robust', 'ood_robust']
    if args.mode == 'adv_robust':
        """
        # ref: https://github.com/AI-secure/DecodingTrust/blob/main/adv-glue-plus-plus/gpt_eval.py#L81
        """
        response = response.lower()
        # remove all text enclosed within ""
        for match in re.findall("\"(.*?)\"", response):
            response = response.replace(match, "")
        answer = response.split(".")[0]
        mapping = answer_mapping[args.task]

        count = 0
        prediction = -1
        for k, v in mapping.items():
            if k in answer:
                prediction = v
                count += 1
        if count != 1:
            # print(response, file=sys.stderr)
            prediction = -1
            """
            compute final result
            ref: https://github.com/AI-secure/DecodingTrust/blob/main/adv-glue-plus-plus/gpt_eval.py#L284
            
            metric = load_metric("glue", task_name if task_name != 'mnli-mm' else 'mnli')
            label_list = np.array(results[task_name]['labels'])
            pred_list = np.array(results[task_name]['predictions'])
            label_list[pred_list < 0] = 1
            pred_list[pred_list < 0] = 0
            assert len(label_list) == len(pred_list)
            score = metric.compute(predictions=pred_list, references=label_list)
            """
            
        return prediction

    elif args.mode == 'ood_robust':
        """
        follow the convert method in ref paper, might consider to change
        """
        if args.task == 'style':
            option = ["negative", "positive"]
        elif args.task == 'knowledge' and args.idk is True:
            option = ["0", "1", "2", "3", "4"]
        elif args.task == 'knowledge' and args.idk is False:
            option = ["0", "1", "2", "3"]

        # two ways
        # one in https://github.com/AI-secure/DecodingTrust/blob/main/chat.py#L100
        # or https://github.com/AI-secure/DecodingTrust/blob/main/ood/evaluation_ood.py#L88
        pred = response.split(".")[0].strip()
        if response == label or pred == label:
            prediction = 1
        elif response not in option and pred not in option:
            prediction = -1
        else:
            prediction = 0
        """
        if args.task == "knowledge":
            count = 0
            known_bd = 0
            bd = 0
            for i in range(len(cache)):
                if dataset[i]["label"] in cache[i][1]["choices"][0]["message"]["content"]:
                    count += 1
                else:
                    bd += 1
                    if cache[i][1]["choices"][0]["message"]["content"][0] in ["0", "1", "2", "3"]:
                        known_bd += 1   
            print("ACC : {:.2f}, MACC : {:.2f}, RR : {:.2f}".format(count/len(cache) * 100, (count / (len(cache) - (bd- known_bd))) * 100, ((bd- known_bd)/len(cache)) * 100))          
        elif args.task == "style":
            count = 0
            for i in range(len(cache)):
                if dataset[i]["label"] == cache[i][1]["choices"][0]["message"]["content"].lower().split(".")[0].strip():
                    count += 1
            print("ACC : {:.2f}".format(count/len(cache) * 100))
        """

        return prediction


def response_evaluation(response, label, response_type ="regular"):
    """
    rewrite response_evaluation
    """
    result = dict()
    if response_type == "perturbation":
        response_text = response["complete_text"]
    else:
        response_text = response["text"]   
    # handle special text in perturbation-based response 
    result["evaluation"] = dict()
    result["evaluation"]["prediction"] = response_to_label(response_text, label)
    result['evaluation']['prompt'] = response["prompt"]
    result['evaluation']['label'] = label
    result["raw_response"] = response
    feature_extractor = get_feature(response)
    result["entropy"] = feature_extractor.get_entropy()
    
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--uncertainty_method', required=True, choices=['raw_inference', 'perturbation_uncertainty', 'sample_uncertainty'])
    parser.add_argument('--mode', required=True)
    parser.add_argument('--model', required=True, choices=["text-davinci-003", "text-curie-001"])
    parser.add_argument('--task', required=True)
    parser.add_argument('--dataset', required=True) 
    parser.add_argument('--num_sample', required=False, default=5)
    parser.add_argument('--sample_rate', required=False, default=0.25)
    # parser.add_argument('--compute_changingpoint', required=False, default=True)
    # parser.add_argument('--no-adv', required=False, default=False)
    
    parser.add_argument('--few_shot_num', required=False, type=int, default=0, help='Number of few shot')
    parser.add_argument('--idk', action='store_true', help='Add I dont know option for OOD knowledge')
    parser.add_argument('--out_domain', required=False, type=str, help='Out domain')
    parser.add_argument('--demo_file', required=False, type=str, help='Demo file')
    
    # args = parser.parse_args("--mode adv_robust --model text-curie-001 --task sst2 --dataset alpaca --idk".split())
    args = parser.parse_args()

    args.api_key = os.getenv("OPENAI_API_KEY", "")
    if args.api_key == "": raise('Error, openai key not set')

    if args.mode == 'adv_robust':
        args.data_file = f'../../../DecodingTrust/data/adv-glue-plus-plus/data/{args.dataset}.json'
        if args.dataset == 'original': args.data_file = f'../../../DecodingTrust/data/adv-glue-plus-plus/data/alpaca.json'
    elif args.mode == 'ood_robust':
        if args.task == 'style':
            args.data_file = f'../../../DecodingTrust/data/ood/styles/dev_{args.dataset}.tsv'
        elif args.task == 'knowledge':
            args.data_file = f'../../../DecodingTrust/data/ood/knowledge/{args.dataset}.jsonl'
        if args.dataset == 'original': args.data_file = f'../../../DecodingTrust/data/ood/styles/dev.tsv'
    # args.raw_data_path = f'result/{args.mode}/{args.dataset}/{args.task}/{args.model}/raw_inference'
    args.output_path = f'../result/{args.mode}/{args.dataset}/{args.task}/{args.model}'

    def create_directories_if_not_exist(file_path):
        directory_path = os.path.dirname(file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    create_directories_if_not_exist(args.output_path)

    return args


def get_random_indices(result):
    existing_indices = []
    for index, row in result.iterrows():
        if not pd.isna(row['prompt']):
            existing_indices.append(index)
    num_samples = int(args.sample_rate*len(result)) - len(existing_indices)
    if num_samples <=0: return []
    rest_indices = [index for index in range(len(result)) if index not in existing_indices]
    random_indices = random.sample(rest_indices, num_samples)

    return random_indices


def main():
    prompts, labels = load_dataset()
    if os.path.exists(args.output_path):
        result = pd.read_pickle(args.output_path)
    else:
        result = pd.DataFrame(columns=['prompt', 'label', 'naive_response', 'sample_response', 'pert_response'], index=range(0, len(prompts)))
    random_indices = get_random_indices(result)
    for print_idx, index in enumerate(random_indices):
        prompt = prompts[index]
        if print_idx%100 == 0: print(f"{print_idx+1}/{len(random_indices)}")
        # try:
        naive_response = get_response(prompt, 
                                      temperature=0.0,
                                      API_key=args.api_key,
                                      max_tokens=100,
                                      model_name=args.model)
        sample_response = get_multi_sample_response(prompt,
                                                    temperature=0.7,
                                                    API_key=args.api_key,
                                                    sample_num = args.num_sample,
                                                    max_tokens=100,
                                                    model_name=args.model)
        # pert_response = get_pertu_response(naive_response,
        #                                    temperature=0.0,
        #                                    API_key=args.api_key,
        #                                    max_tokens=100,
        #                                    model_name=args.model)
        result.at[index, 'prompt'] = prompt
        result.at[index, 'label'] = labels[index]
        result.at[index, 'naive_response'] = response_evaluation(naive_response, labels[index])
        result.at[index, 'sample_response'] = [response_evaluation(response, labels[index]) for response in sample_response]
        # for key, value in pert_response.items():
        #     sub_list = []
        #     for i, response in enumerate(value):
        #         sub_list.append(response_evaluation(response, labels[index], response_type="perturbation"))
        #         pert_response[key] = sub_list
        #         result.at[index, "pert_response"] = pert_response
        # except:
        #     print(f"\nError encounterd at {index} -----------------------------\n")
        #     continue
            
    result.to_pickle(args.output_path)

"""
    python run_openai.py --mode adv_robust --model text-curie-001 --task sst2 --dataset alpaca --idk
"""

if __name__ == '__main__':
    args = parse_args()
    print(f"run on model: {args.model} | mode: {args.mode} | task_name: {args.task} | dataset: {args.dataset}")
    main()