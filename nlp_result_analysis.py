import os
import numpy as np
import sys
import shelve
import tqdm
from numpy import dot
from numpy.linalg import norm
import pickle
import tqdm
import pandas as pd
import sys
#sys.path.append("../")

import constant
import utils
from utils import compute_VR, compute_VRO
from eval_utils import Evaluator
from utils import ModelSource

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

TASK_SPEC_METRIC_MAP = {
    "eli5_category": "qa", 
    "wiki": "qa",
    "wmt14": "translation",
    "cnn_daily": "summarization"
}

def get_task_dist_func(evaluator):
    def task_dist_func(a, b):
        #try:
        similarity = evaluator.compute_task_specific_score(a, b)
        if evaluator.mode == "summarization":
            similarity = similarity["rouge-1"]["f"]
            #if isinstance(similarity, dict):
            #    similarity = similarity["rouge-l"]
            #elif isinstance(similarity, list):
            #    similarity = similarity[0]['rouge-l']['f']
        elif evaluator.mode == "qa":
            if isinstance(similarity, dict):
                similarity = similarity['f1']
        return 1 - similarity
    return task_dist_func

def cosine_dist_func(a, b):
    similarity = dot(a, b)/(norm(a)*norm(b))
    return float((1 - np.average(similarity)))

def get_sep(words, sep_set):
    words_len = len(words)
    sentence_sep = list()
    for idx, word in enumerate(words):
        if word in sep_set:
            sentence_sep.append(idx)
    if len(sentence_sep) == 0 or sentence_sep[-1] != words_len:
        sentence_sep.append(words_len) 
    return sentence_sep

def get_baseline(single_line, sep_set, mode=ModelSource.OPENSOURCE):
    if mode == ModelSource.OPENSOURCE:
        words = single_line['decoded_word'][0]
        words_len = len(words)
        gen_probs = single_line['gen_probs'].flatten()
        entropy = single_line['entropy'].flatten()
    elif mode == ModelSource.OPENAI:
        # info["raw_response"]
        words = single_line['token']
        words_len = len(words)
        top_k_prob = single_line['top_k_prob']
        gen_probs = top_k_prob.max(axis=-1)
        entropy = [utils.shannon_entropy(top_k_prob[i]) for i in range(len(top_k_prob))]
    recorder = [[], [], [], []]
    last_sep = -1
    sentence_sep = get_sep(words, sep_set)
    for sep in sentence_sep:
        if last_sep >= words_len:
            break
        sentence_probs = gen_probs[last_sep + 1: sep + 1] # include "."
        sentence_entropy = entropy[last_sep + 1: sep + 1]
        if len(sentence_probs) > 0:
            max_sen_prob = np.nanmax(- np.log(sentence_probs)) 
            recorder[0].append(max_sen_prob)
            average_sen_prob = - np.nanmean(np.log(sentence_probs))
            recorder[1].append(average_sen_prob)
        if len(sentence_entropy) > 0:
            max_sen_entropy = np.nanmax(sentence_entropy)
            recorder[2].append(max_sen_entropy)
            average_sen_entropy = np.nanmean(sentence_entropy)
            recorder[3].append(average_sen_entropy)
        last_sep = sep
    baseline = {
        "doc_max_prob": np.nanmean(recorder[0]),
        "doc_average_prob": np.nanmean(recorder[1]),
        "doc_max_ent": np.nanmean(recorder[2]),
        "doc_average_ent": np.nanmean(recorder[3])
    }
    return baseline

def get_basic_info(raw_data, sep_set, mode=ModelSource.OPENSOURCE, evaluate_metric=None):
    """
    Args:
        evaluate_metric: a function that takes (output, gt) to compute the model's performance.
    Return performance map (key --> [cosine_score, task_metric]) and
        baseline map (baseline_uncertainty --> corresponding score)
    """
    perf_map = dict()
    baseline_map = dict()
    assert mode in ModelSource
    if mode == ModelSource.OPENSOURCE:
        for key in raw_data:
            baseline = get_baseline(raw_data[key], sep_set, mode)
            baseline_map[key] = baseline
            gen_probs = raw_data[key]['gen_probs'].flatten()
            baseline_map[key]["prob_max"] = np.nanmax(gen_probs)
            baseline_map[key]["prob_mean"] = np.nanmean(gen_probs)
            baseline_map[key]["entropy_average"] = np.nanmean(raw_data[key]['entropy'].flatten())
            baseline_map[key]["entropy_max"] = np.nanmax(raw_data[key]['entropy'])
    elif mode == ModelSource.OPENAI:
        # It is acutually raw_data["naive_response"]
        for idx in range(len(raw_data)):
            # data['naive_response'][idx]
            top_k_prob = raw_data[idx]["raw_response"]['top_k_prob']
            gen_probs = top_k_prob.max(axis=-1)
            entropy = [utils.shannon_entropy(top_k_prob[i]) for i in range(len(top_k_prob))]
            baseline = get_baseline(raw_data[idx]["raw_response"], sep_set, mode)
            baseline_map[idx] = baseline
            baseline_map[idx]["prob_max"] = np.nanmax(gen_probs)
            baseline_map[idx]["prob_mean"] = np.nanmean(gen_probs)
            baseline_map[idx]["entropy_average"] = np.nanmean(entropy)
            baseline_map[idx]["entropy_max"] = np.nanmax(entropy)
    for key in raw_data:
        if evaluate_metric is None:
            task_metric = raw_data[key]['evaluation']['task_metric'] if "task_metric" in raw_data[key]['evaluation'] else []
        else:
            # Compute evaluate metrics
            if mode == ModelSource.OPENSOURCE:
                output = raw_data[key]['decoded_text'][0]
                gt = raw_data[key]['evaluation']['gt']
            elif mode == ModelSource.OPENAI:
                output = raw_data[key]['raw_response']['complete_text']
                gt = raw_data[key]['evaluation']['gt']
            task_metric = evaluate_metric(output, gt)
        perf_map[key] = {"cosine_score": raw_data[key]['evaluation']["cosine"], 
                         "task_metric": task_metric}
        
    return (perf_map, baseline_map)

def get_perturbation_score(raw_data, uncertain_data, evaluator, tokenizer, dist_func, with_embed=True, mode=ModelSource.OPENSOURCE):
    """
    Return VR and VRO score for the given dataset
    """
    assert mode in ModelSource
    uncertain_map = dict()
    key_list = list(raw_data.keys()) #if mode == "opensource" else list(range(len(raw_data)))
    for key in tqdm.tqdm(key_list):
        if mode == ModelSource.OPENSOURCE:
            max_point = uncertain_data[key]['entropy']["max"][0]
            if len(uncertain_data[key]['diff_entropy'][0]) < 1:
                diff_entropy = max_point
            else:
                diff_entropy = uncertain_data[key]['diff_entropy'][0][0]
            min_point = uncertain_data[key]['entropy']["min"][0]
            original_text = tokenizer.decode(raw_data[key]['gen_sequences'].flatten())
            original_text = evaluator.post_process(original_text, tokenizer.eos_token)
            points = [diff_entropy, max_point, min_point]
        elif mode == ModelSource.OPENAI:
            original_text = raw_data[key]["raw_response"]['complete_text']
            points = list(uncertain_data[key].keys())
            max_point = "max_alter"
            min_point = "min_alter"
            diff_entropy = "max_diff_alter"
        if with_embed:
            origin_input = evaluator.compute_embedding([original_text]) # (1, embed_size)
        else:
            origin_input = [original_text]
        result = dict()
        for point in points:
            if mode == ModelSource.OPENSOURCE:
                #prefix = tokenizer.decode(raw_data[key]['gen_sequences'][:point].flatten())
                perturbed_text = [evaluator.post_process(tokenizer.decode(i['gen_sequences'].flatten()), 
                                                        tokenizer.eos_token) 
                                for i in uncertain_data[key][point]
                                ]
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
            vr = compute_VR([perturbed_input], dist_func)[0]
            vro = compute_VRO([perturbed_input], origin_input, dist_func)[0]
            result[point] = {"vr": vr, "vro": vro}
        uncertain_map[key] = {"max_entropy": result[max_point], "min_entropy": result[min_point], "max_diff": result[diff_entropy]}
    return uncertain_map

def get_sample_score(raw_data, uncertain_data, evaluator, tokenizer, dist_func, with_embed=True, mode=ModelSource.OPENSOURCE):
    """
    Return VR and VRO score for the given dataset
    """
    assert mode in ModelSource
    uncertain_map = dict()
    key_list = list(raw_data.keys()) #if mode == "opensource" else list(range(len(raw_data)))
    for key in tqdm.tqdm(key_list):
        if mode == ModelSource.OPENSOURCE:
            original_text = evaluator.post_process(raw_data[key]['decoded_text'][0], tokenizer.eos_token)
            perturbed_text = [evaluator.post_process(i['decoded_text'][0], tokenizer.eos_token) for i in uncertain_data[key]]
        elif mode == ModelSource.OPENAI:
            original_text = raw_data[key]["raw_response"]['complete_text'] # original input
            perturbed_text = [uncertain_data[key][idx]["raw_response"]["complete_text"] for idx in range(1, len(uncertain_data[key]))]
        if with_embed:
            original_input = evaluator.compute_embedding([original_text]) # (1, embed_size)
            perturbed_input = evaluator.compute_embedding(perturbed_text)
        else:
            original_input = [original_text]
            perturbed_input = perturbed_text
        vr = compute_VR([perturbed_input], dist_func)[0]
        vro = compute_VRO([perturbed_input], original_input, dist_func)[0]
        uncertain_map[key] = {"vr": vr, "vro":vro}
    return uncertain_map

def get_task_rouge1_func():
    evaluator = Evaluator(None, None, mode="summarization")
    def task_rouge1_func(output, gt):
        # Attention:  gt is the first parameter
        # Since output is usually much larger than gt.
        score = evaluator.compute_task_specific_score(gt, output)
        return score["rouge-1"]["r"]
    return task_rouge1_func

# python3 nlp_result_analysis.py --model llama2 --output_path /data/huangyuheng/QA/result/llama2_uncertainty_all.pickle  --log_file log/llama2_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model gpt2 --output_path /data/huangyuheng/QA/result/gpt2_uncertainty_all.pickle  --log_file log/gpt2_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model llama --output_path /data/huangyuheng/QA/result/llama_uncertainty_all.pickle  --log_file log/llama_uncertainty_analysis_all.log --model_path /home/huangyuheng/LLM/cache/7B_hf/
# python3 nlp_result_analysis.py --model davinci --output_path /data/huangyuheng/QA/result/davinci_uncertainty_all4.pickle  --log_file log/davinci_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model curie --output_path /data/huangyuheng/QA/result/curie_uncertainty_all4.pickle  --log_file log/curie_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model gpt4o --output_path /tmp/gpt4o.pickle  --log_file /tmp/gpt4o.log

# python3 nlp_result_analysis.py --model llama3 --output_path /data/huangyuheng/QA/result/llama3_uncertainty_all.pickle  --log_file log/llama3_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model phi3 --output_path /data/huangyuheng/QA/result/phi3_uncertainty_all.pickle  --log_file log/phi3_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model gemma2 --output_path /data/huangyuheng/QA/result/gemma2_uncertainty_all.pickle  --log_file log/gemma2_uncertainty_analysis_all.log

# python3 nlp_result_analysis.py --model gpt4o --output_path /data/huangyuheng/QA/result/gpt4o_uncertainty_all.pickle --log_file log/gpt4o_uncertainty_analysis_all.log
# python3 nlp_result_analysis.py --model gpt4omini --output_path /data/huangyuheng/QA/result/gpt4omini_uncertainty_all.pickle --log_file log/gpt4omini_uncertainty_analysis_all.log

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", choices=["gpt2", "llama", "llama2", "davinci", "gpt4o", "gpt4omini", 
                                                                       "curie", "llama2_notemplate", "llama3", "phi3", "gemma2"])
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--log_file", type=str, default="debug.log")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model. Only used for LLAMA")
    #parser.add_argument("--model_source", type=str, default="opensource", choices=["opensource", "openai"])
    #parser.add_argument("--evaluation_metric", type=str, default=None, choices=["rouge-1", "bleu", "f1"])
    args = parser.parse_args()
    
    # Setup logging
    file_filter = None
    logger, file_handler = utils.setupLogging(args.log_file, file_filter, multiprocess=True)
    logger.info("Running %s" % " ".join(sys.argv))
    
    model_source = ModelSource.OPENSOURCE
    model = args.model
    if args.model == "llama":
        data_paths = constant.LLAMA_RAW_DATA
    elif args.model == "llama2":
        data_paths = constant.LLAMA2_RAW_DATA
    elif args.model == "gpt2":
        data_paths = constant.GPT2_RAW_DATA
    elif args.model == "davinci":
        data_paths = constant.NLP_DAVINCI_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "curie":
        data_paths = constant.NLP_CURIE_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "gpt4o":
        data_paths = constant.NLP_GPT4O_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "gpt4omini":
        data_paths = constant.NLP_GPT4OMINI_RAW_DATA
        model_source = ModelSource.OPENAI
    elif args.model == "llama2_notemplate":
        data_paths = constant.LLAMA2_NO_TEMPLETE
        model = "llama2"
    elif args.model == "llama3":
        data_paths = constant.LLAMA3_RAW_DATA
    elif args.model == "phi3":
        data_paths = constant.PHI3_RAW_DATA
    elif args.model == "gemma2":
        data_paths = constant.GEMMA2_RAW_DATA
    
    if model_source == ModelSource.OPENSOURCE:
        tokenizer = utils.load_tokenizer(model, args.model_path)
    elif model_source == ModelSource.OPENAI:
        tokenizer = None
        
    logger.info(f"Using {args.model} model")
    
    from spacy.lang.en import English
    nlp = English()
    sentencizer = nlp.create_pipe('sentencizer')
    SEP_SET = set(sentencizer.default_punct_chars)
    from sentence_transformers import SentenceTransformer
    sen_path = "sentence-transformers/all-mpnet-base-v2"
    sentencemodel = SentenceTransformer(sen_path)
    # evaluator in the below is only used for post-processing and embedding computation
    # So it does not matter for the `task` paramter
    evaluator = Evaluator(None, sentencemodel, mode="summarization")

    result = dict()
    for dataset in tqdm.tqdm(data_paths):
        original_data_path = data_paths[dataset]
        if model_source == ModelSource.OPENSOURCE:
            shelve_raw_data = shelve.open(original_data_path)
            raw_data = dict(shelve_raw_data)
            shelve_raw_data.close()
            logger.info(f"Loading from {original_data_path}") 
        else:
            all_data = pd.read_pickle(original_data_path)
            all_data = all_data.to_dict()
            raw_data = all_data["naive_response"]
            sample_uncertain_data = all_data["sample_response"]
            alter_uncertain_data = all_data["pert_response"]
            logger.info(f"Loading all data from {original_data_path}") 
        # Evaluate
        if model_source == ModelSource.OPENSOURCE:
            alter_path = constant.RAW_DATA_PATHS[original_data_path][0][0]
            logger.info(f"Loading perturbation uncertainty data from {alter_path}")
            sample_path = constant.RAW_DATA_PATHS[original_data_path][1][0]
            logger.info(f"Loading sample uncertainty data from {sample_path}")
        perf_map, baseline_map = get_basic_info(raw_data, SEP_SET, evaluate_metric=get_task_rouge1_func(), mode=model_source)
        
        # This func is used to compute distance needed in uncertainty
        task_specific_func = get_task_dist_func(Evaluator(None, None, TASK_SPEC_METRIC_MAP[dataset]))
        
        if model_source == ModelSource.OPENSOURCE:
            shelve_sample_data = shelve.open(sample_path)
            sample_uncertain_data = dict(shelve_sample_data)
            shelve_sample_data.close()
        sample_map_cos = get_sample_score(raw_data, sample_uncertain_data, evaluator, tokenizer, cosine_dist_func, True, mode=model_source)
        sample_map_task = get_sample_score(raw_data, sample_uncertain_data, evaluator, tokenizer, task_specific_func, False, mode=model_source)
        
        if model_source == ModelSource.OPENSOURCE:
            shelve_alter_data = shelve.open(alter_path)
            alter_uncertain_data = dict(shelve_alter_data)
            shelve_alter_data.close()
        perturb_map_cos = get_perturbation_score(raw_data, alter_uncertain_data, evaluator, tokenizer, cosine_dist_func, True, mode=model_source)
        perturb_map_task = get_perturbation_score(raw_data, alter_uncertain_data, evaluator, tokenizer, task_specific_func, False, mode=model_source)
        
        result[original_data_path] = {"perf_map": perf_map, "baseline_map": baseline_map,
                                        "perturb_map_cos": perturb_map_cos, "perturb_map_task": perturb_map_task,
                                        "sample_map_cos": sample_map_cos, "sample_map_task": sample_map_task}
            
    with open(args.output_path, "wb") as ofile:
        pickle.dump(result, ofile)