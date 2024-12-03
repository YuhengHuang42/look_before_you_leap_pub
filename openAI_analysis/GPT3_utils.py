import numpy as np
import openai
import pandas as pd
import comp_helper
from tenacity import retry, stop_after_attempt, wait_random_exponential
import json
import spacy
from eval_utils import Evaluator
from sentence_transformers import SentenceTransformer
import os
import sys
sys.path.append("../")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


task_mode_table ={
    "wiki_qa_test_new": "qa",
    "eli5_category_test": "qa",
    "cnn_dailymail_3_test": "summarization",
    "wmt14_fr-en_test": "translation",
    "mbpp": "code",
    "humaneval": "code",
    }
num_table = {    
    "wiki_qa_test_new": 100,
    "eli5_category_test": 100,
    "cnn_dailymail_3_test": 100,
    "wmt14_fr-en_test": 100,
    "mbpp": 125,
    "humaneval": 40,
    }

class get_feature():
    """
    1. Get entropy and probability features from response tokens
    2. Locate the token with maximum/manimum/max_diff entropy and return crossponding top-k alternatives
    """
    def __init__(self, response):
        self.response = response

        top_k_prob = np.array(self.response['top_k_prob'])       # top-k prob of each token
        self.top_k_token = self.response['top_k_token']               # top-k alternative of each token

        self.entropy_list = np.zeros(len(top_k_prob))
        self.prob_list = np.zeros(len(top_k_prob))
        for i in range(len(top_k_prob)):
            self.entropy_list[i] = comp_helper.uncertainty(top_k_prob[i][top_k_prob[i]!=0])[1]    # hanlde 0 prob
            self.prob_list[i] = top_k_prob[i,0]

    def get_entropy(self):
        # mean/max/min/diff entropy over all tokens in response
        if len(self.entropy_list) > 0:
            entropy_mean = np.mean(self.entropy_list)
            entropy_max = np.max(self.entropy_list)
            entropy_min = np.min(self.entropy_list)
            if len(self.entropy_list) > 1:
                entropy_diff_mean = np.mean(np.abs(self.entropy_list[1:]-self.entropy_list[:-1]))
                entropy_diff_max = np.max(self.entropy_list[1:]-self.entropy_list[:-1])
            else:
                entropy_diff_mean = 0
                entropy_diff_max = 0
            return {"entropy_mean": entropy_mean,
                    "entropy_max": entropy_max,
                    "entropy_min": entropy_min,
                    "entropy_diff_mean": entropy_diff_mean,
                    "entropy_diff_max": entropy_diff_max,
                    }
        else: 
            return{"entropy_mean": 0,
                    "entropy_max": 0,
                    "entropy_min": 0,
                    "entropy_diff_mean": 0,
                    "entropy_diff_max": 0,
                    }

    def get_prob(self):
        # mean/min probabilities over all tokens
        prob_mean = np.mean(self.prob_list)
        prob_min = np.min(self.prob_list)
        prob_diff_mean = np.mean(np.abs(self.prob_list[1:]-self.prob_list[0:-1]))
        prob_diff_max = np.max(np.abs(self.prob_list[1:]-self.prob_list[0:-1]))
        return {"prob_mean": prob_mean,
               "prob_min": prob_min,
               "prob_diff_mean": prob_diff_mean,
               "prob_diff_max":prob_diff_max
               }

    def get_changing_point(self):
        # top-k alternatives of the perturbation token
        entropy_info = self.get_entropy()
        max_index = np.where(self.entropy_list==entropy_info["entropy_max"])[0][0]
        min_index = np.where(self.entropy_list==entropy_info["entropy_min"])[0][0]
        diff = self.entropy_list[1:]-self.entropy_list[0:-1]
        max_diff_index = np.where(diff==entropy_info["entropy_diff_max"])[0][0] + 1

        max_alter = {"index": max_index, "top_k_alter":self.top_k_token[max_index]}
        min_alter = {"index": min_index, "top_k_alter":self.top_k_token[min_index]}
        max_diff_alter = {"index": max_diff_index, "top_k_alter":self.top_k_token[max_diff_index]}
        return {"max_alter": max_alter,
                "min_alter": min_alter,
                "max_diff_alter": max_diff_alter}


# OpenAI API rate limit handler
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client, **kwargs):
        return client.chat.completions.create(**kwargs)

class Get_LLM_Response(object):
    """
    class to invoke and format GPT response from OpenAI API
    """
    def __init__(self, key):
            os.environ["OPENAI_API_KEY"] = key
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])   # key to access openAI API

    def get_response(self, sys_prompt:str, user_prompt:str, temperature:float, max_tokens=200, model_name="gpt-4o-mini"):
        """
        Send a prompt and get response
        """
        result =dict()

        # send the request to the API and get the response
        response = completion_with_backoff(
            client = self.client,
            model = model_name,
            messages=[{"role": "system", "content":  sys_prompt},
                    {"role": "user", "content": user_prompt}],
            logprobs = True,
            temperature = temperature,      # Between 0 and 2, Higher values make the output more random, while lower values will make it more focused and deterministic.
            top_logprobs = 5,
            presence_penalty = 0,           # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
            frequency_penalty = 0,          # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            max_tokens = max_tokens,        # The maximum number of tokens to generate in the completion.
        )
        
        # format response
        result = self.response_formatting(sys_prompt, user_prompt, response)

        # new_response = json.loads(str(response))["choices"][0]
        # top_logprobs = new_response["logprobs"]["top_logprobs"] # top-k prob for each token
        # num_token_response = len(top_logprobs)                  # num of tokens in response
        # top_k_prob = np.zeros([num_token_response, 5])          # array to hold all top-k prob [num_token, top_k_prob]
        # top_k_scores = np.zeros([num_token_response, 5])        # top-k score for each token
        # usgae = json.loads(str(response))["usage"]

        # # tokens in response
        # top_k_tokens = [list(content.keys()) for content in top_logprobs]
        # top_k_tokens = np.array(top_k_tokens)

        # # sort the order of top-k toekns, prob, socre follwing prob with decending order
        # for index, content in enumerate(top_logprobs):
        #     top_k_prob[index, :] = np.sort(np.exp(np.array(list(top_logprobs[index].values()))))[::-1]
        #     top_k_scores[index, :] = np.sort(np.array(list(top_logprobs[index].values())))[::-1]
        #     new_index = np.argsort(np.exp(np.array(list(top_logprobs[index].values()))))[::-1]
        #     top_k_tokens[index, :] = top_k_tokens[index, new_index]

        # # add LLM outputs to dataframe
        # result['prompt'] = sys_prompt + user_prompt  # input prompt
        # # result['response'] = response             # original response
        # result['text'] = new_response["text"]     # text answer
        # result['token'] = new_response["logprobs"]["tokens"]      # token answer
        # result['top_k_token'] = top_k_tokens      # top-k tokens
        # result['top_k_prob'] = top_k_prob         # top-k prob
        # result['top_logprobs'] = top_logprobs     # top-k token-prob pair
        # result["complete_text"] = new_response["text"]
        return result


    def response_formatting(self, sys_prompt, user_prompt, response):
        """format the structure of the response"""
        answer_text = response.choices[0].message.content
        logprobs_list = response.choices[0].logprobs.content
        logprobs = list()
        tokens = list()
        top_k_tokens = list()
        top_k_prob = list()

        for toekn_probs in logprobs_list:
            token_prob = {}
            for element in toekn_probs.top_logprobs:
                token_prob[element.token] = np.exp(element.logprob)
            tokens.append(toekn_probs.top_logprobs[0].token)
            logprobs.append(token_prob)
            top_k_tokens.append([i.token for i in toekn_probs.top_logprobs])
            top_k_prob.append([np.exp(i.logprob )for i in toekn_probs.top_logprobs])

        formatted_response = {
            "sys_prompt": sys_prompt,
            "user_prompt": user_prompt,
            "text": answer_text,
            "token": tokens,
            "top_k_token": np.array(top_k_tokens),      # top-k tokens
            "top_k_prob": np.array(top_k_prob),         # top-k prob
            "complete_text": answer_text,
            "top_logprobs": logprobs
        }
        return formatted_response


    def get_multi_sample_response(self, sys_prompt:str, user_prompt:str, temperature:float, sample_num=5, max_tokens=200, model_name="gpt-4o-mini"):
        """
        For sample-based uncertainty measurement. Get multiple samples ragarding one prompt 
        """
        result = []
        for i in range(sample_num):
            result.append(self.get_response(sys_prompt, user_prompt, temperature, max_tokens, model_name))
        return result


    def get_pertu_response(self, origin_response, temperature:float,max_tokens=200, model_name="gpt-4o-mini"):
        """
        Find changing point and get perturbated response
        """
        origin_tokens = origin_response["token"]
        sys_prompt = origin_response["sys_prompt"]
        origin_user_prompt = origin_response["user_prompt"]
        feature_extractor = get_feature(origin_response)
        change_point = feature_extractor.get_changing_point()
        result = dict()
        if change_point["max_alter"]["index"] != change_point["max_diff_alter"]["index"]:
            for key, value in change_point.items():
                sub_list = [origin_response]
                alter_tokens = value["top_k_alter"][1:]
                for i in range(0, len(alter_tokens)):
                    new_prompt = origin_user_prompt + "".join(origin_tokens[:value["index"]]) + alter_tokens[i]
                    response = self.get_response(sys_prompt, new_prompt, temperature, max_tokens, model_name)
                    response["complete_text"] = "".join(origin_tokens[:value["index"]]) + alter_tokens[i] + response["text"]
                    sub_list.append(response)
                result[key] = sub_list
        else:
            for key in ["max_alter", "min_alter"]:
                sub_list = [origin_response]
                value = change_point[key]
                alter_tokens = value["top_k_alter"][1:]
                for i in range(0, len(alter_tokens)):
                    new_prompt = origin_user_prompt + "".join(origin_tokens[:value["index"]]) + alter_tokens[i]
                    response = self.get_response(sys_prompt, new_prompt, temperature, max_tokens, model_name)
                    response["complete_text"] = "".join(origin_tokens[:value["index"]]) + alter_tokens[i] + response["text"]
                    sub_list.append(response)
                result[key] = sub_list
            result["max_diff_alter"] = result["max_alter"]
        return result


    def response_evaluation(self, evaluator, response, answer, mode, response_type ="regular", task=None, reference=None):
        """
        Calculate score for input response
        """
        result = dict()
        if response_type == "perturbation":
            response_text = response["complete_text"]
        else:
            response_text = response["text"]   
        if mode in ["qa", "summarization", "translation"]:
            # handle special text in perturbation-based response 
            result["evaluation"] = evaluator.compute_passage_score(response_text, answer)
            result["evaluation"]["task_metric"] = evaluator.compute_task_specific_score(response_text, answer)
            result["raw_response"] = response
            feature_extractor = get_feature(response)
            result["entropy"] = feature_extractor.get_entropy()
            result['evaluation']['input'] = response["user_prompt"]
            result['evaluation']['gt'] = answer
        else:
            result["evaluation"] = task.process_results([[response_text]], [reference])
            result["raw_response"] = response
            feature_extractor = get_feature(response)
            result["entropy"] = feature_extractor.get_entropy()
            result['evaluation']['input'] = response["user_prompt"]
            result['evaluation']['gt'] = reference
        return result

    def get_instance(self, dataset_path, output_path, num_instance, spacy_model, sentence_model, mode, print_freq=10, llm_model="gpt-4o-mini"):
        """ 
        Collect respones regarding different uncertainty measurement methods
        """
        nlp = spacy.load(spacy_model)
        sentencemodel = SentenceTransformer(sentence_model)
        
        print(f"Loading dataset: {dataset_path} \n" + 
                f"Get instance number: {num_instance} \n" + 
                f"Using spacy model: {spacy_model} \n" + 
                f"Using embedding model: {sentence_model} \n" + 
                f"Using LLM model: {llm_model} \n" + 
                f"Taks mode: {mode}")

        # setup task-specific parameters
        if mode in ["qa", "summarization", "translation"]:
            evaluator = Evaluator(nlp, sentencemodel, mode=mode)
            df = pd.read_csv(dataset_path)
            df = df.loc[0:num_instance-1]
            dataset_len = len(df)

        elif mode == "code":
            if dataset_path == "mbpp":
                from code_analysis import mbpp
                task = mbpp.MBPP()
            elif dataset_path == "humaneval":
                from code_analysis import human_eval
                task = human_eval.HumanEval()

            dataset = task.get_dataset()
            dataset_len = len(dataset)
            references = [task.get_reference(dataset[i]) for i in range(dataset_len)]

        else:
            print("Task not found")

        mode_specific_parameter = {
            "prompt_style":     {"qa": "Answer the following question: ", 
                                "summarization": "Summarize the following sentences: ",
                                "code": "You are a helpful assistant. Generate the code with necessary libiray dependencies without python markdown or explaination.",
                                # "code": "You are a helpful assistant. Generate the function with necessary libiray dependencies without python markdown.",
                                "translation": "Translate French to English: "},
            "max_token":    {"qa": 100,
                            "summarization": 200,
                            "translation": 700,
                            "code": 650}}

        # collect responses with different uncertainty measurement
        # 1(naive) + 5(sample) + 3*5(perturbation) = 21 responses for each question
        result = pd.DataFrame(columns=['question', 'answer', 'naive_response', 'sample_response', 'pert_response'], index=range(0, num_instance))
        if mode in ["qa", "summarization", "translation"]:
            for idx, row in df.iterrows():
                if idx%print_freq == 0: print(f"Iterating index: {idx}")
                try:
                    sys_prompt = mode_specific_parameter["prompt_style"][mode]
                    user_prompt = row["question"]
                    naive_response = self.get_response(
                                            sys_prompt, 
                                            user_prompt,
                                            temperature=0.0, 
                                            max_tokens=mode_specific_parameter["max_token"][mode],
                                            model_name=llm_model)

                    sample_response = self.get_multi_sample_response(
                                            sys_prompt, 
                                            user_prompt,
                                            temperature=0.7, 
                                            sample_num = 5,
                                            max_tokens=mode_specific_parameter["max_token"][mode],
                                            model_name=llm_model)
                    
                    pert_response = self.get_pertu_response(
                                            naive_response,
                                            temperature=0.0, 
                                            max_tokens=mode_specific_parameter["max_token"][mode],
                                            model_name=llm_model)
                    # evalute response
                    result.at[idx, "question"] = row["question"]
                    result.at[idx, "prompt"] = user_prompt
                    result.at[idx, "answer"] = row["answer"]
                    result.at[idx, "naive_response"] = self.response_evaluation(evaluator, naive_response, row["answer"], mode)
                    result.at[idx, "sample_response"] = [self.response_evaluation(evaluator, response, row["answer"], mode) for response in sample_response]

                    for key, value in pert_response.items():
                        sub_list = []
                        for i, response in enumerate(value):
                            sub_list.append(self.response_evaluation(evaluator, response, row["answer"], mode, response_type="perturbation"))
                        pert_response[key] = sub_list
                        result.at[idx, "pert_response"] = pert_response
                except:
                        print(f"\nError encounterd at {idx} -----------------------------\n")
                        continue

                
        elif mode == "code":
            evaluator = Evaluator(nlp, sentencemodel, mode="qa")    # dummy variable
            for idx in range(num_instance):
                if idx%print_freq == 0: print(f"Iterating index: {idx}")
                try:
                    user_prompt = task.get_prompt(dataset[idx])
                    sys_prompt = mode_specific_parameter["prompt_style"][mode]
                    reference = references[idx]
                    naive_response = self.get_response(
                                            sys_prompt = sys_prompt, 
                                            user_prompt = user_prompt,
                                            temperature=0.0, 
                                            max_tokens=mode_specific_parameter["max_token"][mode],
                                            model_name=llm_model)

                    sample_response = self.get_multi_sample_response(
                                            sys_prompt = sys_prompt, 
                                            user_prompt = user_prompt, 
                                            temperature=0.7, 
                                            sample_num = 5,
                                            max_tokens=mode_specific_parameter["max_token"][mode],
                                            model_name=llm_model)
                    
                    pert_response = self.get_pertu_response(
                                            naive_response,
                                            temperature=0.0, 
                                            max_tokens=mode_specific_parameter["max_token"][mode],
                                            model_name=llm_model)
                    # evalute response
                    result.at[idx, "question"] = idx
                    result.at[idx, "prompt"] = user_prompt
                    result.at[idx, "answer"] = reference
                    result.at[idx, "naive_response"] = self.response_evaluation(evaluator, naive_response, reference, mode, task=task, reference=reference)
                    result.at[idx, "sample_response"] = [self.response_evaluation(evaluator, response, reference, mode, task=task, reference=reference) for response in sample_response]
                    for key, value in pert_response.items():
                        sub_list = []
                        for i, response in enumerate(value):
                            sub_list.append(self.response_evaluation(evaluator, response, mode, reference, response_type="perturbation", task=task, reference=reference))
                        pert_response[key] = sub_list
                    result.at[idx, "pert_response"] = pert_response
                except:
                        print(f"\nError encounterd at {idx} -----------------------------\n")
                        # continue
                #         return [naive_response, sample_response, pert_response, result]
                #         raise Exception
        else:
            print("No instance proceed")

        # save results
        # result.to_csv(output_path, index=False)
        result.to_pickle(output_path)
        print(f"Complete. Results saved to: {output_path} \n")
        return result