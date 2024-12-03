import numpy as np
import pandas as pd
import pickle
import spacy
from sentence_transformers import SentenceTransformer
from copy import deepcopy
import sys
import shelve
sys.path.append("../")
import abstract_state
import tqdm
from itertools import chain
from scipy.stats import ks_2samp
from GPT3_utils import get_feature, Get_LLM_Response, task_mode_table, num_table
from eval_utils import Evaluator
import nltk
from nltk.tokenize import word_tokenize
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# nltk.download('punkt')
# from code_analysis import mbpp, human_eval

"""
Datasets: wiki_qa_test_new, eli5_category_test, cnn_dailymail_3_test, wmt14_fr-en_test, mbpp, humaneval
GPT model: gpt-4o-mini,  gpt-4o
Task type: qa, summarization, translation, code
"""

model_name = "gpt-4o"
data_set_name = "humaneval"

spacy_model = "en_core_web_trf"
sentence_model = "sentence-transformers/all-mpnet-base-v2"
print_freq = 10



if __name__ == "__main__":
    GPT_invoker = Get_LLM_Response(api_key)

    for data_set_name in ["wiki_qa_test_new", "eli5_category_test", "cnn_dailymail_3_test", "wmt14_fr-en_test", "mbpp", "humaneval"]:
        mode = task_mode_table[data_set_name]
        num_instance = num_table[data_set_name]
        output_path = "../data/experiment/result/new_results/" + data_set_name + "_" + str(num_instance) + "_" + model_name + ".pkl"

        if mode == "code":
            dataset_path = data_set_name
        else:
            dataset_path = "../data/experiment/" + data_set_name + ".csv"

        result = GPT_invoker.get_instance(dataset_path, output_path, num_instance, spacy_model, sentence_model, mode, print_freq, model_name)