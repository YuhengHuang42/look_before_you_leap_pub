import numpy as np
import os
import re
import inspect
import ast
import tqdm
import torch
import shelve
import pickle
import pandas as pd

import sys
#sys.path.append("../")
import constant
import utils
from utils import compute_VR, compute_VRO
from utils import ModelSource
from nlp_result_analysis import get_baseline, cosine_dist_func

class CodeEvaluator:
    def __init__(self, task, tokenizer, model):
        """
        Two sets of methods:
            1. compute embeddings of given code
                compute_embedding()
            2. evaluate the code quality in terms of pass
                First call process_test_cases on test cases,
                then call evaluate_generation to get the score.
        """
        assert task in ["humaneval", "mbpp"]
        self.task = task
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from evaluate import load
        self.code_metric = load("code_eval")
        self.tokenizer = tokenizer
        self.model = model
        
    @staticmethod
    def extract_assertions_from_humaneval(code_str):
        # Use regex to extract all assertions
        func_name_pattern = r"check\((\w+)\)"
        matches = re.findall(func_name_pattern, code_str)
        func_name = matches[-1]
        
        assertion_pattern = r"(assert .*\n)"
        assertions = re.findall(assertion_pattern, code_str)
        
        # Replace the "candidate" with the given function name
        modified_assertions = [assertion.replace("candidate", func_name) for assertion in assertions]
        
        return modified_assertions
    
    def process_test_cases(self, gt):
        """
        Turn the ground truth into a list of test cases
        Output:
            [[Assertion 1], [Assertion 2], ...]
        """
        if self.task == "mbpp":
            result = gt.split("\n")
        elif self.task == "humaneval":
            result = self.extract_assertions_from_humaneval(gt)
        return result
    
    def evaluate_generation(self, generation, gt):
        """
        Evaluate a single generation
        """
        gt_length = len(gt)
        # First perform ast parsing
        score = 0
        try:
            _ = ast.parse(generation[0])
            score += 0.5
        except:
            # No need to perform further evaluation
            return 0
        results, detail = self.code_metric.compute(
            references=gt,
            predictions=[generation for i in range(gt_length)],
            num_workers=1,
            timeout=3 # https://huggingface.co/spaces/evaluate-metric/code_eval
        )
        return score + results["pass@1"] / 2

    def compute_single(self, x):
        tokens = self.tokenizer(x, max_length=500, truncation=True)
        input_ids = tokens["input_ids"]
        input_ids = torch.tensor(input_ids)[None,:].to(self.model.device)
        with torch.no_grad():
            embeddings = self.model(input_ids)[0]
            embeddings = embeddings.mean(dim=1)
        return embeddings.cpu().squeeze()
    
    def compute_embedding(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        temp = list()
        for x in inputs:
            y = self.compute_single(x)
            #print(y.shape)
            temp.append(y)
        #print(torch.stack(temp).shape)
        return torch.stack(temp)

    def openai_postprocess(self, single_info):
        if self.task == "humaneval":
            if "prompt" in single_info['raw_response']:
                # OLD Version API.
                return single_info['raw_response']["prompt"] + single_info['raw_response']['complete_text']
            #return single_info['raw_response']["prompt"] + single_info['raw_response']['complete_text']
            return single_info['raw_response']['complete_text']
        else:
            return single_info['raw_response']['complete_text']
    

def get_basic_info(raw_data, sep_set, evaluate_class, mode=ModelSource.OPENSOURCE):
    """
    Return performance map (key --> [performance_score, task_metric]) and
        baseline map (baseline_uncertainty --> corresponding score)
    """
    perf_map = dict()
    baseline_map = dict()
    for key in raw_data:
        if mode == ModelSource.OPENSOURCE:
            baseline = get_baseline(raw_data[key], sep_set, mode)
            baseline_map[key] = baseline
            #gen_probs = [max(i['top_k_prob'][0]) for i in raw_data[key]['llm_info']]
            gen_probs = raw_data[key]['gen_probs'].flatten()
            
            baseline_map[key]["prob_max"] = np.nanmax(gen_probs)
            baseline_map[key]["prob_mean"] = gen_probs.nanmean().item()
            baseline_map[key]["entropy_average"] = np.nanmean(raw_data[key]['entropy'].flatten())
            baseline_map[key]["entropy_max"] = np.nanmax(raw_data[key]['entropy'])
        elif mode == ModelSource.OPENAI:
            top_k_prob = raw_data[key]["raw_response"]['top_k_prob']
            gen_probs = top_k_prob.max(axis=-1)
            entropy = [utils.shannon_entropy(top_k_prob[i]) for i in range(len(top_k_prob))]
            baseline = get_baseline(raw_data[key]["raw_response"], sep_set, mode)
            baseline_map[key] = baseline
            baseline_map[key]["prob_max"] = np.nanmax(gen_probs)
            baseline_map[key]["prob_mean"] = np.nanmean(gen_probs)
            baseline_map[key]["entropy_average"] = np.nanmean(entropy)
            baseline_map[key]["entropy_max"] = np.nanmax(entropy)
    for key in raw_data:
        task_metric = raw_data[key]['evaluation']['task_metric'] if "task_metric" in raw_data[key]['evaluation'] else []
        if mode == ModelSource.OPENSOURCE:
            generation = raw_data[key]["final_result"]
        elif mode == ModelSource.OPENAI:
            if evaluate_class is not None:
                generation = [evaluate_class.openai_postprocess(raw_data[key])]
        if evaluate_class is not None:
            gt = evaluate_class.process_test_cases(raw_data[key]['evaluation']['gt'])
            performance = evaluate_class.evaluate_generation(generation, gt)
        else:
            performance = []
        if mode == ModelSource.OPENSOURCE:
            perf = raw_data[key]['evaluation']['pass@1']
        else:
            if "naive_eval" in raw_data[key]['evaluation']:
                perf = raw_data[key]['evaluation']['naive_eval']['pass@1']
            else:
                perf = raw_data[key]['evaluation']['pass@1']
        perf_map[key] = {"performance": performance, #raw_data[key]['evaluation']['pass@1'], 
                         "task_metric": task_metric,
                         "pass@1": perf}
    return (perf_map, baseline_map)

def collect_dist(inf_result, dist_func):
    # inf_result: the result returned from MC dropout
    # inf_result: [batch_size, inference_time, ...]
    batch_size = len(inf_result)
    inf_time = len(inf_result[0])
    result = []
    for batch_idx in range(batch_size):
        final_score = 0
        for first_T in range(inf_time):
            dist_all = 0
            for second_T in range(inf_time):
                if first_T == second_T:
                    continue
                dist = dist_func(
                        inf_result[batch_idx][first_T],
                        inf_result[batch_idx][second_T])
                dist_all +=  dist
            final_score += dist_all / (inf_time - 1)
        result.append(final_score / inf_time)
    return result

def get_perturbation_score(raw_data, uncertain_data, evaluator, tokenizer, dist_func, with_embed=True, use_prefix=False, mode=ModelSource.OPENSOURCE, collect_dist_flag=False):
    """
    Return VR and VRO score for the given dataset
    """
    assert mode in ModelSource
    uncertain_map = dict()
    #select_key = "decoded_text" if use_prefix else "final_result"
    for key in tqdm.tqdm(raw_data):
        if mode == ModelSource.OPENSOURCE:
            max_point = uncertain_data[key]['entropy']["max"][0]
            if len(uncertain_data[key]['diff_entropy'][0]) < 1:
                diff_entropy = max_point
            else:
                diff_entropy = uncertain_data[key]['diff_entropy'][0][0]
            min_point = uncertain_data[key]['entropy']["min"][0]
            original_text = raw_data[key]["decoded_text"] # list (len = 1)
            points = [diff_entropy, max_point, min_point]
        elif mode == ModelSource.OPENAI:
            original_text = [raw_data[key]["raw_response"]['complete_text']] # list (len = 1)
            points = list(uncertain_data[key].keys())
            max_point = "max_alter"
            min_point = "min_alter"
            diff_entropy = "max_diff_alter"
        if with_embed:
            #original_text = raw_data[key]['final_result'][0]
            origin_input = evaluator.compute_embedding(original_text) # (1, embed_size)
        else:
            #origin_input = raw_data[key]["decoded_text"] # final_result
            origin_input = original_text
        result = dict()
        for point in points:
            if mode == ModelSource.OPENSOURCE:
                if use_prefix:
                    prefix = tokenizer.decode(raw_data[key]['gen_sequences'][:point].flatten())
                else:
                    prefix = ""
                perturbed_text = [prefix + i["decoded_text"][0] for i in uncertain_data[key][point]] # ATTENTION
            elif mode == ModelSource.OPENAI:
                perturbed_text = []
                for record in uncertain_data[key][point][1:]:
                    if "response" in record['raw_response']:
                        # Last version OpenAI API response
                        target_data = record['raw_response']['response']['choices'][0]['text']
                    else:
                        target_data = record["raw_response"]["text"]
                    perturbed_text.append(target_data) # Attention
            if with_embed:
                perturbed_input = evaluator.compute_embedding(perturbed_text) # (N-1, embed_size)
                #vr_input = [np.concatenate([origin_input, perturbed_input])]
            else:
                perturbed_input = perturbed_text
                #vr_input = [origin_input + perturbed_input]
            if collect_dist_flag == False:
                vr = compute_VR([perturbed_input], dist_func)[0]
                vro = compute_VRO([perturbed_input], origin_input, dist_func)[0]
                result[point] = {"vr": vr, "vro": vro}
            else:
                ave_dist = collect_dist([perturbed_input], dist_func)[0]
                result[point] = ave_dist
        uncertain_map[key] = {"max_entropy": result[max_point], "min_entropy": result[min_point], "max_diff": result[diff_entropy]}
    return uncertain_map

def get_sample_score(raw_data, uncertain_data, evaluator, dist_func, with_embed=True, mode=ModelSource.OPENSOURCE, collect_dist_flag=False):
    """
    Return VR and VRO score for the given dataset
    """
    assert mode in ModelSource
    uncertain_map = dict()
    for key in tqdm.tqdm(raw_data):
        if mode == ModelSource.OPENSOURCE:
            original_text = raw_data[key]['decoded_text'][0]
            perturbed_text = [i['decoded_text'][0] for i in uncertain_data[key]]
        elif mode == ModelSource.OPENAI:
            original_text = raw_data[key]["raw_response"]['complete_text'] # original input
            perturbed_text = [uncertain_data[key][idx]["raw_response"]["complete_text"] for idx in range(1, len(uncertain_data[key]))]
        if with_embed:
            original_input = evaluator.compute_embedding([original_text]) # (1, embed_size)
            perturbed_input = evaluator.compute_embedding(perturbed_text)
        else:
            original_input = [original_text]
            perturbed_input = perturbed_text
        if collect_dist_flag == False:
            vr = compute_VR([perturbed_input], dist_func)[0]
            vro = compute_VRO([perturbed_input], original_input, dist_func)[0]
            uncertain_map[key] = {"vr": vr, "vro":vro}
        else:
            ave_dist = collect_dist([perturbed_input], dist_func)[0]
            uncertain_map[key] = ave_dist
    return uncertain_map

def get_dist_func(module):
    def dist_func(tgt, src):
        return 1 - module.compute(predictions = [tgt], references = [[src]])['CodeBLEU']
    return dist_func

'''
python3 code_result_analysis.py --model codellama --output_path /data/huangyuheng/code/result/codellama_uncertainty_all.pkl \
    --embedding_device cuda:0 --log_file log/codellama_uncertainty_analysis_all.log

python3 code_result_analysis.py --model codegen --output_path /data/huangyuheng/code/result/codegen_uncertainty_all.pkl \
    --embedding_device cuda:1 --log_file log/codegen_uncertainty_analysis_all.log
    
python3 code_result_analysis.py --model santacoder --output_path /data/huangyuheng/code/result/santacoder_uncertainty_all.pkl \
    --embedding_device cuda:1 --log_file log/santacoder_uncertainty_analysis_all.log

python3 code_result_analysis.py --model incoder --output_path /data/huangyuheng/code/result/incoder_uncertainty_all.pkl \
    --embedding_device cuda:0 --log_file log/incoder_uncertainty_analysis_all.log
    

python3 code_result_analysis.py --model deepseekcoder --output_path /data/huangyuheng/code/result/deepseekcoder_uncertainty_all.pkl \
    --embedding_device cuda:1 --log_file log/deepseekcoder_uncertainty_analysis_all.log
    
python3 code_result_analysis.py --model codeqwen1.5 --output_path /data/huangyuheng/code/result/codeqwen1.5_uncertainty_all.pkl \
    --embedding_device cuda:1 --log_file log/codeqwen1.5_uncertainty_analysis_all.log

python3 code_result_analysis.py --model starcoder2 --output_path /data/huangyuheng/code/result/starcoder2_uncertainty_all.pkl \
    --embedding_device cuda:1 --log_file log/starcoder2_uncertainty_analysis_all.log

python3 code_result_analysis.py --model gpt4o --output_path /data/huangyuheng/code/result/gpt4o_uncertainty_all.pkl  --log_file log/code_gpt4o_uncertainty_analysis_all.log
python3 code_result_analysis.py --model gpt4omini --output_path /data/huangyuheng/code/result/gpt4omini_uncertainty_all.pkl  --log_file log/code_gpt4omini_uncertainty_analysis_all.log

'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="codegen", choices=["codegen", "santacoder", "incoder", "gpt4o",
                                                                         "codellama", "davinci", "deepseekcoder", 
                                                                         "codeqwen1.5", "starcoder2", "gpt4omini"])
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--log_file", type=str, default="debug.log")
    parser.add_argument("--embedding_device", type=str, default="cuda:0")
    parser.add_argument("--use_prefix", type=bool, default=False)
    parser.add_argument("--collect_dist_flag", default=False, action="store_true")
    args = parser.parse_args()

    # Setup logging
    file_filter = None
    logger, file_handler = utils.setupLogging(args.log_file, file_filter, multiprocess=True)
    logger.info("Running %s" % " ".join(sys.argv))
    
    model_source = ModelSource.OPENSOURCE
    if args.model == "codegen":
        data_paths = constant.CODEGEN_RAW_DATA
    elif args.model == "santacoder":
        data_paths = constant.SANTACODER_RAW_DATA
    elif args.model == "incoder":
        data_paths = constant.INCODER_RAW_DATA
    elif args.model == "codellama":
        data_paths = constant.CODELLAMA_RAW_DATA
    elif args.model == "davinci":
        data_paths = constant.CODE_DAVINCI_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "gpt4o":
        data_paths = constant.CODE_GPT4O_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "gpt4omini":
        data_paths = constant.CODE_GPT4OMINI_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "deepseekcoder":
        data_paths = constant.DEEPSEEKCODER_RAW_DATA
    elif args.model == "codeqwen1.5":
        data_paths = constant.CODEQWEN_RAW_DATA
    elif args.model == "starcoder2":
        data_paths = constant.STARCODER2_RAW_DATA
    else:
        raise Exception("Model not supported")
    if model_source == ModelSource.OPENSOURCE:
        target_model_tokenizer = utils.load_tokenizer(args.model)
    else:
        target_model_tokenizer = None
        
    from transformers import AutoTokenizer, AutoModel
    embedding_model_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    embedding_model = AutoModel.from_pretrained("microsoft/codebert-base")
    embedding_model = embedding_model.eval()
    embedding_model = embedding_model.to(args.embedding_device)
    SEP_SET = set(["\n"])
    # Code Belu metric
    import evaluate
    module = evaluate.load("dvitel/codebleu")
    task_dist_func = get_dist_func(module)

    result = dict()
    for dataset in tqdm.tqdm(data_paths):
        original_data_path = data_paths[dataset]
        if model_source == ModelSource.OPENSOURCE:
            shelve_raw_data = shelve.open(original_data_path)
            raw_data = dict(shelve_raw_data)
            shelve_raw_data.close()
        elif model_source == ModelSource.OPENAI:
            all_data = pd.read_pickle(original_data_path)
            all_data = all_data.to_dict()
            raw_data = all_data["naive_response"]
            sample_uncertain_data = all_data["sample_response"]
            alter_uncertain_data = all_data["pert_response"]
        logger.info(f"Loading from {original_data_path}") 
        
        # Evaluate
        evaluator = CodeEvaluator(dataset, embedding_model_tokenizer, embedding_model)
        # Load data
        if model_source == ModelSource.OPENSOURCE:
            alter_path = constant.CODE_RAW_DATA_PATHS[original_data_path][0][0]
            logger.info(f"Loading perturbation uncertainty data from {alter_path}")
            sample_path = constant.CODE_RAW_DATA_PATHS[original_data_path][1][0]
            logger.info(f"Loading sample uncertainty data from {sample_path}")
        
        if args.collect_dist_flag == False:
            perf_map, baseline_map = get_basic_info(raw_data, SEP_SET, evaluator, mode=model_source)
        else:
            perf_map = None
            baseline_map = None
        
        if model_source == ModelSource.OPENSOURCE:
            shelve_sample_data = shelve.open(sample_path)
            sample_uncertain_data = dict(shelve_sample_data)
            shelve_sample_data.close()
        sample_map_cos = get_sample_score(raw_data, sample_uncertain_data, evaluator, cosine_dist_func, True, mode=model_source, collect_dist_flag=args.collect_dist_flag)
        sample_map_task = get_sample_score(raw_data, sample_uncertain_data, evaluator, task_dist_func, False, mode=model_source, collect_dist_flag=args.collect_dist_flag)
        
        if model_source == ModelSource.OPENSOURCE:
            shelve_alter_data = shelve.open(alter_path)
            alter_uncertain_data = dict(shelve_alter_data)
            shelve_alter_data.close()
        perturb_map_cos = get_perturbation_score(raw_data, alter_uncertain_data, evaluator, 
                                                 target_model_tokenizer, cosine_dist_func, True, use_prefix=args.use_prefix, mode=model_source, collect_dist_flag=args.collect_dist_flag)
        perturb_map_task = get_perturbation_score(raw_data, alter_uncertain_data, evaluator, 
                                                  target_model_tokenizer, task_dist_func, False, use_prefix=args.use_prefix, mode=model_source, collect_dist_flag=args.collect_dist_flag)
        
        result[original_data_path] = {"perf_map": perf_map, "baseline_map": baseline_map,
                                        "perturb_map_cos": perturb_map_cos, "perturb_map_task": perturb_map_task,
                                        "sample_map_cos": sample_map_cos, "sample_map_task": sample_map_task}
        
        
    with open(args.output_path, "wb") as ofile:
            pickle.dump(result, ofile)