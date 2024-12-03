# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility function for nq evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
from gzip import GzipFile
import json
import multiprocessing
from datasets import load_dataset
from numpy.linalg import norm
import numpy as np
import math
from scipy.spatial.distance import cityblock, minkowski, jaccard
#from absl import flags
#from absl import logging

"""
flags.DEFINE_integer(
    'long_non_null_threshold', 2,
    'Require this many non-null long answer annotations '
    'to count gold as containing a long answer.')
flags.DEFINE_integer(
    'short_non_null_threshold', 2,
    'Require this many non-null short answer annotations '
    'to count gold as containing a short answer.')

FLAGS = flags.FLAGS
"""

# A data structure for storing prediction and annotation.
# When a example has multiple annotations, multiple NQLabel will be used.
NQLabel = collections.namedtuple(
    'NQLabel',
    [
        'example_id',  # the unique id for each NQ example.
        'long_answer_span',  # A Span object for long answer.
        'short_answer_span_list',  # A list of Spans for short answer.
        #   Note that In NQ, the short answers
        #   do not need to be in a single span.
        'yes_no_answer',  # Indicate if the short answer is an yes/no answer
        #   The possible values are "yes", "no", "none".
        #   (case insensitive)
        #   If the field is "yes", short_answer_span_list
        #   should be empty or only contain null spans.
        'long_score',  # The prediction score for the long answer prediction.
        'short_score'  # The prediction score for the short answer prediction.
    ])



def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
      result = int(pred_tokens == truth_tokens)
      return {"f1": result, "precision": result, "recall": result}
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    return {"precision": prec, "recall": rec, "f1":f1}
  
  
# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
  
class Span(object):
  """A class for handling token and byte spans.
    The logic is:
    1) if both start_byte !=  -1 and end_byte != -1 then the span is defined
       by byte offsets
    2) else, if start_token != -1 and end_token != -1 then the span is define
       by token offsets
    3) else, this is a null span.
    Null spans means that there is no (long or short) answers.
    If your systems only care about token spans rather than byte spans, set all
    byte spans to -1.
  """

  def __init__(self, start_byte, end_byte, start_token_idx, end_token_idx):

    if ((start_byte < 0 and end_byte >= 0) or
        (start_byte >= 0 and end_byte < 0)):
      raise ValueError('Inconsistent Null Spans (Byte).')

    if ((start_token_idx < 0 and end_token_idx >= 0) or
        (start_token_idx >= 0 and end_token_idx < 0)):
      raise ValueError('Inconsistent Null Spans (Token).')

    if start_byte >= 0 and end_byte >= 0 and start_byte >= end_byte:
      raise ValueError('Invalid byte spans (start_byte >= end_byte).')

    if ((start_token_idx >= 0 and end_token_idx >= 0) and
        (start_token_idx >= end_token_idx)):
      raise ValueError('Invalid token spans (start_token_idx >= end_token_idx)')

    self.start_byte = start_byte
    self.end_byte = end_byte
    self.start_token_idx = start_token_idx
    self.end_token_idx = end_token_idx

  def is_null_span(self):
    """A span is a null span if the start and end are both -1."""

    if (self.start_byte < 0 and self.end_byte < 0 and
        self.start_token_idx < 0 and self.end_token_idx < 0):
      return True
    return False

  def __str__(self):
    byte_str = 'byte: [' + str(self.start_byte) + ',' + str(self.end_byte) + ')'
    tok_str = ('tok: [' + str(self.start_token_idx) + ',' +
               str(self.end_token_idx) + ')')

    return byte_str + ' ' + tok_str

  def __repr__(self):
    return self.__str__()


def is_null_span_list(span_list):
  """Returns true iff all spans in span_list are null or span_list is empty."""
  if not span_list or all([span.is_null_span() for span in span_list]):
    return True
  return False


def nonnull_span_equal(span_a, span_b):
  """Given two spans, return if they are equal.
  Args:
    span_a: a Span object.
    span_b: a Span object.  Only compare non-null spans. First, if the bytes are
      not negative, compare byte offsets, Otherwise, compare token offsets.
  Returns:
    True or False
  """
  assert isinstance(span_a, Span)
  assert isinstance(span_b, Span)
  assert not span_a.is_null_span()
  assert not span_b.is_null_span()

  # if byte offsets are not negative, compare byte offsets
  if ((span_a.start_byte >= 0 and span_a.end_byte >= 0) and
      (span_b.start_byte >= 0 and span_b.end_byte >= 0)):

    if ((span_a.start_byte == span_b.start_byte) and
        (span_a.end_byte == span_b.end_byte)):
      return True

  # if token offsets are not negative, compare token offsets
  if ((span_a.start_token_idx >= 0 and span_a.end_token_idx >= 0) and
      (span_b.start_token_idx >= 0 and span_b.end_token_idx >= 0)):

    if ((span_a.start_token_idx == span_b.start_token_idx) and
        (span_a.end_token_idx == span_b.end_token_idx)):
      return True

  return False


def span_set_equal(gold_span_list, pred_span_list):
  """Make the spans are completely equal besides null spans."""

  gold_span_list = [span for span in gold_span_list if not span.is_null_span()]
  pred_span_list = [span for span in pred_span_list if not span.is_null_span()]

  for pspan in pred_span_list:
    # not finding pspan equal to any spans in gold_span_list
    if not any([nonnull_span_equal(pspan, gspan) for gspan in gold_span_list]):
      return False

  for gspan in gold_span_list:
    # not finding gspan equal to any spans in pred_span_list
    if not any([nonnull_span_equal(pspan, gspan) for pspan in pred_span_list]):
      return False

  return True


def gold_has_short_answer(gold_label_list):
  """Gets vote from multi-annotators for judging if there is a short answer."""

  #  We consider if there is a short answer if there is an short answer span or
  #  the yes/no answer is not none.
  gold_has_answer = gold_label_list and sum([
      ((not is_null_span_list(label.short_answer_span_list)) or
       (label.yes_no_answer != 'none')) for label in gold_label_list
  ]) >= FLAGS.short_non_null_threshold

  return gold_has_answer


def gold_has_long_answer(gold_label_list):
  """Gets vote from multi-annotators for judging if there is a long answer."""

  gold_has_answer = gold_label_list and (sum([
      not label.long_answer_span.is_null_span()  # long answer not null
      for label in gold_label_list  # for each annotator
  ]) >= FLAGS.long_non_null_threshold)

  return gold_has_answer


def read_prediction_json(predictions_path):
  """Read the prediction json with scores.
  Args:
    predictions_path: the path for the prediction json.
  Returns:
    A dictionary with key = example_id, value = NQInstancePrediction.
  """
  print('Reading predictions from file: %s', format(predictions_path))
  #logging.info('Reading predictions from file: %s', format(predictions_path))
  with open(predictions_path, 'r') as f:
    predictions = json.loads(f.read())

  nq_pred_dict = {}
  for single_prediction in predictions['predictions']:

    if 'long_answer' in single_prediction:
      long_span = Span(single_prediction['long_answer']['start_byte'],
                       single_prediction['long_answer']['end_byte'],
                       single_prediction['long_answer']['start_token'],
                       single_prediction['long_answer']['end_token'])
    else:
      long_span = Span(-1, -1, -1, -1)  # Span is null if not presented.

    short_span_list = []
    if 'short_answers' in single_prediction:
      for short_item in single_prediction['short_answers']:
        short_span_list.append(
            Span(short_item['start_byte'], short_item['end_byte'],
                 short_item['start_token'], short_item['end_token']))

    yes_no_answer = 'none'
    if 'yes_no_answer' in single_prediction:
      yes_no_answer = single_prediction['yes_no_answer'].lower()
      if yes_no_answer not in ['yes', 'no', 'none']:
        raise ValueError('Invalid yes_no_answer value in prediction')

      if yes_no_answer != 'none' and not is_null_span_list(short_span_list):
        raise ValueError('yes/no prediction and short answers cannot coexist.')

    pred_item = NQLabel(
        example_id=single_prediction['example_id'],
        long_answer_span=long_span,
        short_answer_span_list=short_span_list,
        yes_no_answer=yes_no_answer,
        long_score=single_prediction['long_answer_score'],
        short_score=single_prediction['short_answers_score'])

    nq_pred_dict[single_prediction['example_id']] = pred_item

  return nq_pred_dict

def read_annotation_from_one_split(gzipped_input_file):
  """Read annotation from one split of file."""
  if isinstance(gzipped_input_file, str):
    gzipped_input_file = open(gzipped_input_file, 'rb') 
  #logging.info('parsing %s ..... ', gzipped_input_file.name)
  print('parsing %s ..... ', gzipped_input_file.name)
  annotation_dict = {}
  with GzipFile(fileobj=gzipped_input_file) as input_file:
    for line in input_file:
      json_example = json.loads(line)
      example_id = json_example['example_id']

      # There are multiple annotations for one nq example.
      annotation_list = []
      for annotation in json_example['annotations']:
        long_span_rec = annotation['long_answer']
        long_span = Span(long_span_rec['start_byte'], long_span_rec['end_byte'],
                         long_span_rec['start_token'],
                         long_span_rec['end_token'])

        short_span_list = []
        for short_span_rec in annotation['short_answers']:
          short_span = Span(short_span_rec['start_byte'],
                            short_span_rec['end_byte'],
                            short_span_rec['start_token'],
                            short_span_rec['end_token'])
          short_span_list.append(short_span)

        gold_label = NQLabel(
            example_id=example_id,
            long_answer_span=long_span,
            short_answer_span_list=short_span_list,
            long_score=0,
            short_score=0,
            yes_no_answer=annotation['yes_no_answer'].lower())

        annotation_list.append(gold_label)
      annotation_dict[example_id] = annotation_list

  return annotation_dict

def read_data_instance_from_one_split(gzipped_input_file):
  """Read dataset and return instance from one split of file."""
  if isinstance(gzipped_input_file, str):
    gzipped_input_file = open(gzipped_input_file, 'rb') 
  #logging.info('parsing %s ..... ', gzipped_input_file.name)
  print('parsing %s ..... ', gzipped_input_file.name)
  annotation_dict = {}
  with GzipFile(fileobj=gzipped_input_file) as input_file:
    for line in input_file:
      json_example = json.loads(line)
      example_id = json_example['example_id']

      # There are multiple annotations for one nq example.
      annotation_list = []

      for annotation in json_example['annotations']:
        long_span_rec = annotation['long_answer']
        long_span = Span(long_span_rec['start_byte'], long_span_rec['end_byte'],
                         long_span_rec['start_token'],
                         long_span_rec['end_token'])

        short_span_list = []
        for short_span_rec in annotation['short_answers']:
          short_span = Span(short_span_rec['start_byte'],
                            short_span_rec['end_byte'],
                            short_span_rec['start_token'],
                            short_span_rec['end_token'])
          short_span_list.append(short_span)

        gold_label = NQLabel(
            example_id=example_id,
            long_answer_span=long_span,
            short_answer_span_list=short_span_list,
            long_score=0,
            short_score=0,
            yes_no_answer=annotation['yes_no_answer'].lower())

        annotation_list.append(gold_label)
      json_example = simplify_nq_example(json_example)
      annotation_dict[example_id] = {"original_example": json_example, "anno": annotation_list}

  return annotation_dict

def read_annotation(path_name, n_threads=10, train_eval_switch=False):
  """Read annotations with real multiple processes."""
  input_paths = glob.glob(path_name)
  pool = multiprocessing.Pool(n_threads)
  try:
    if train_eval_switch is False:
        dict_list = pool.map(read_annotation_from_one_split, input_paths)
    else:
        dict_list = pool.map(read_data_instance_from_one_split, input_paths)
  finally:
    pool.close()
    pool.join()

  final_dict = {}
  for single_dict in dict_list:
    final_dict.update(single_dict)

  return final_dict

import re


def get_nq_tokens(simplified_nq_example):
  """Returns list of blank separated tokens."""

  if "document_text" not in simplified_nq_example:
    raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                     "example that contains the `document_text` field.")

  return simplified_nq_example["document_text"].split(" ")


def simplify_nq_example(nq_example):
  r"""Returns dictionary with blank separated tokens in `document_text` field.
  Removes byte offsets from annotations, and removes `document_html` and
  `document_tokens` fields. All annotations in the ouput are represented as
  [start_token, end_token) offsets into the blank separated tokens in the
  `document_text` field.
  WARNING: Tokens are separated by a single blank character. Do not split on
    arbitrary whitespace since different implementations have different
    treatments of some unicode characters such as \u180e.
  Args:
    nq_example: Dictionary containing original NQ example fields.
  Returns:
    Dictionary containing `document_text` field, not containing
    `document_tokens` or `document_html`, and with all annotations represented
    as [`start_token`, `end_token`) offsets into the space separated sequence.
  """

  def _clean_token(token):
    """Returns token in which blanks are replaced with underscores.
    HTML table cell openers may contain blanks if they span multiple columns.
    There are also a very few unicode characters that are prepended with blanks.
    Args:
      token: Dictionary representation of token in original NQ format.
    Returns:
      String token.
    """
    return re.sub(u" ", "_", token["token"])

  text = " ".join([_clean_token(t) for t in nq_example["document_tokens"]])

  def _remove_html_byte_offsets(span):
    if "start_byte" in span:
      del span["start_byte"]

    if "end_byte" in span:
      del span["end_byte"]

    return span

  def _clean_annotation(annotation):
    annotation["long_answer"] = _remove_html_byte_offsets(
        annotation["long_answer"])
    annotation["short_answers"] = [
        _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
    ]
    return annotation

  simplified_nq_example = {
      "question_text": nq_example["question_text"],
      "example_id": nq_example["example_id"],
      "document_url": nq_example["document_url"],
      "document_text": text,
      "long_answer_candidates": [
          _remove_html_byte_offsets(c)
          for c in nq_example["long_answer_candidates"]
      ],
      "annotations": [_clean_annotation(a) for a in nq_example["annotations"]]
  }

  if len(get_nq_tokens(simplified_nq_example)) != len(
      nq_example["document_tokens"]):
    raise ValueError("Incorrect number of tokens.")

  return simplified_nq_example


def gather_text_only_gt(original, anno):
    """
    Return long and short answrs as text only format.
    Args:
        original: nq_gold_dict[key]['original_example']
        anno: nq_gold_dict[key]['anno']
    Please refer to read_data_instance_from_one_split for
    more details regarding the input format
    """
    text = original['document_text'].split()
    long_span_set = set()
    short_span_set = set()
    for item in anno:
        long_span = (item.long_answer_span.start_token_idx, item.long_answer_span.end_token_idx)
        long_span_set.add(long_span)
        for single in item.short_answer_span_list:
            short_span = (single.start_token_idx, single.end_token_idx)
            short_span_set.add(short_span)
    result = {
        "long": [" ".join(text[i[0]: i[1]]) for i in long_span_set],
        "short": [" ".join(text[i[0]: i[1]]) for i in short_span_set]
    }
    return result


def SaveToCSV(source, dataset_name, target_key, target_column, answer_key, save_path, is_label=False, is_context=False):
    """
    source: name of dataset to load from API
    dataset_name:   name of the datset to save
    target_key:     name of the target splits in datasets
    target_column:  name of the columns to keep in each df
    answer_key:     if answer is dict, only keep the selected content
    is_label:       if the dataset is mixed with wrong answers
    is_context:     if this QA has given context
    """

    # Load datasets from API
    dataset = load_dataset(source)
    print(f"dataset: {source}, keys: {list(dataset.keys())} \n")

    output_dict = {}

    # extract data from target categories
    for key in target_key:
        # convert to pandas df
        df = dataset[key].to_pandas()

        # only take rows with true lable
        if is_label:
            df = df[df['label'] == 1].reset_index()

        # drop redundant deatures/columns
        keys_to_drop  = [ele for ele in list(df.keys()) if ele not in target_column]
        df = df.drop(keys_to_drop, axis=1)

        # rename column
        if is_context:
            df = df.rename(columns={target_column[0]: "question", target_column[1]: "answer", target_column[2]: "context"})
        else:
            df = df.rename(columns={target_column[0]: "question", target_column[1]: "answer"})

        # map answer as text format
        if len(answer_key) != 0:
            df["answer"] = df["answer"].apply(lambda x: x[answer_key[0]][0])
        
        # log info
        print(f"key: {key}, num: {len(df)} \n")
        output_dict[key] = df

    # save as csv
    for key in output_dict.keys():
        output_dict[key].to_csv(save_path + dataset_name + "_" + key + ".csv", index=False)
  
  
class Evaluator():
    def __init__(self, nlp, model, mode="qa"):
        self.nlp = nlp
        self.model = model
        self.mode = mode
        assert mode in ["qa", "summarization", "translation"]
        if mode == "summarization":
          from rouge_metric import PyRouge
          rouge = PyRouge(rouge_n=(1))
          
          #from rouge import Rouge # Python package
          #rouge = Rouge()
          self.computer = rouge
        elif mode == "translation":
          import nltk
          from nltk.tokenize import word_tokenize
          def get_score(reference, candidates):
            reference = [word_tokenize(reference)]
            candidates = word_tokenize(candidates)
            BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, candidates)
            return BLEUscore
          self.computer = get_score

    def _compute_embedding_score(self, target_1:str, target_2:str):
        """
        Compute cosine, euclidean, dot, etc similarity/distance based on embedding vectors
        """
        vec_1, vec_2 = self.model.encode([target_1, target_2])   # get vectors
        cosine = np.dot(vec_1, vec_2)/(norm(vec_1)*norm(vec_2))      # compute cosine similarity
        euclidean = math.dist(vec_1, vec_2)             # compute Euclidean distance
        dot_product = np.dot(vec_1, vec_2)              # compute dot product
        manhattan = cityblock(vec_1, vec_2)             #calculate Manhattan distance between vectors
        minkowski_distance = minkowski(vec_1, vec_2)    # Minkowski distance
        jaccard_sim = jaccard(vec_1, vec_2)             # Jaccard similarity

        result = {"cosine": cosine,
                  "euclidean": euclidean,
                  "dot_product": dot_product,
                  "manhattan": manhattan,
                  "minkowski": minkowski_distance,
                  "jaccard": jaccard_sim}
        return result

    def compute_passage_score(self, response:str, answer:str):
        """
        Compute passage-level similarity scores 
        """
        result = self._compute_embedding_score(response, answer)
        return result

    def compute_sentence_score(self, response: list, answer: list):
        """
        Compute sentence-level similarity scores 
        """
        result = list() # list(dict) one dict for each sentence
        start_index = 0
        assert len(response) == len(answer)
        # loop each sentence
        for idx, re in enumerate(response):
            an = answer[idx]
            result.append(self._compute_embedding_score(re, an))
        return result
    
    def compute_task_specific_score(self, response: str, answer: str):
      if self.mode == "summarization":
        #if len(response) == 0 or len(answer) == 0:
        #      result = int(response == answer)
        #      return {"rouge-1": result, "rouge-2": result, "rouge-l": result}
        #res_characters = set(response)
        #ans_characters = set(answer)
        # Refer to https://github.com/pltrdy/rouge/blob/c1b90a78c303b5d5956b1c1f2dac19cdd553d9b9/rouge/rouge.py#LL115C60-L115C61
        # It will takes "." as the end of a sentence.
        # Input with only "." will cause error.
        #if len(res_characters) == 1 and list(res_characters)[0] == ".":
        #  response = response.replace(".", "*")
        #if len(ans_characters) == 1 and list(ans_characters)[0] == ".":
        #  answer = answer.replace(".", "*")
        #scores = self.computer.get_scores(response, answer)
        scores = self.computer.evaluate([response], [[answer]])
        return scores
      elif self.mode == "qa":
        scores = compute_f1(response, answer)
        return scores
      elif self.mode == "translation":
        return self.computer(answer, response)
        
    
    def post_process(self, response: str, special_token:str):
      """
      Post-process for generated text
      """
      response = response.replace(special_token, "")
      response = response.strip()
      return response
    
    def compute_embedding(self, targets: list):
      return self.model.encode(targets)
  
def text_filter(text:str, nlp, target_tag:set, is_lemma=False):
  """
  Filter tokens with selected POS tag
  """
  doc = nlp(text)
  select_tokens = list()
  for token in doc:
      if token.pos_ in target_tag:
          if is_lemma:
              select_tokens.append(token.lemma_)
          else:
              select_tokens.append(token.text)
  return " ".join(select_tokens)