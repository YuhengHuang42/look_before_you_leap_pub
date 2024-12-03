import numpy as np
import pandas as pd
import json
import dataclasses
from dataclasses import dataclass, fields, _MISSING_TYPE
from heapq import heappush, heappop, heappushpop
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
from entropy_calculation import uncertainty



SPACY_INFO_DICT = {"text":[], "lemma_":[], "pos_":[], 
                  "tag_":[], "dep_":[], "shape_":[], "is_alpha":[],
                 "is_stop":[], "i":[]}

# Universal POS tags: https://universaldependencies.org/u/pos/
POS_NOUN = set(["NOUN", "PRON", "PROPN", "NN", "NNP", "NNPS", "NNS", "NE", "NNE"])
POS_VERB = set(["VERB", "AUX"])
NOUN_CHILDREN = set(["NOUN", "PRON", "PROPN", "ADJ"])
VERB_CHILDREN = set(["NOUN", "PRON", "PROPN", "ADV"])

# @dataclass(unsafe_hash=True)
class StateStatics:
    """
    Class to collect statistics at state level
    1. moving statistics 
    2. weighted statistics
    """
    def __init__(self):
        self.moving_doc_median = dict()
        self.moving_rmse = dict()
        self.moving_mean = dict()
        self.weighted_moving_doc_median = dict()   # weighted moving statistics
        self.weighted_moving_rmse = dict()
        self.weighted_moving_mean = dict()
        self.alter_token = dict()       # alternative token, word similarity, pos_tag: {"alter_token_1": [similarity, pos_tag], ...}
        self.weighted_prob = 0.0        # weighted prob w.r.t syntactic children
        self.weighted_entropy = 0.0     # weighted entropy w.r.t syntactic children

@dataclass(eq=False)
class PropTag:
    """
    Class to represent the property tag of each token
    Convert POS tags to equivalent property tags
    """
    # pos_tags: list = None          # Can there exist multiple pos_tages for one token?
    pos_tag: str            # Coarse-grained part-of-speech
    propstr: str = None     # map an equivalent property to the same hashable value

    def __post_init__(self):
        self.propstr = self._hash_map_func(self.pos_tag)
    
    # equivalent property tag mapping
    def _hash_map_func(self, prop):
        if prop in POS_NOUN:
            return "NOUN"
        elif prop in POS_VERB:
            return "VERB"
        else:
            return prop

@dataclass(unsafe_hash=True)
class State:
    """ 
    Construct the basic state following the aligned outputs from llm and spacy
    Additional attributes will be added in later processes 
    """
    idx: int            # token idx w.r.t entire passage
    token: str          # text based on spacy
    entropy: float      # Shannon entropy 
    probability: float  # highest prob
    llm_info: list      # one word in spact may represent serval tokens in llm
    spacy_info: dict    # nlp_result
    summary: dict       # all token info for backup
    weights: float= None            # weights assigned for syntactic children
    prop_tag: PropTag = None
    statistics: StateStatics = None
    children: dict = None           # syntactic children
        
class StateOverview:
    """
    Copmute some statistical info w.r.t both state and passage levels:
    1. split states by sentences
    2. compute moving statisics (sliding window)
    3. compute weighted statisics based on syntactic children
    """
    def __init__(self, states:list, window_len:int, target_tags:set, children_weights:float, mode:str, nlp_model):
        self.states = states            # constructed state
        self.window_len = window_len    # sliding window for interval info
        self.target_tags = target_tags    # copmute wighted statistics for tokens with target tags 
        self.children_weights = children_weights  # weights for dependent child states
        self.sentence_state, self.end_state_index = self.split_states_by_sentence(states)
        self.moving_mdoe = mode         # passage level or sentence level
        self.nlp_model = nlp_model      # spacy model to process alternative tokens
        # self.prop_link_map = self.generate_property_link(states)
        
        self.compute_weighted_statistics()
        self.compute_moving_statistics()
        self.compute_alter_token_statistics()

    def compute_alter_token_statistics(self):
        """
        Get alternative tokens for each state, get word similarity score, pos_tag, probability
        """
        for state in self.states:
            alter_tokens = state.llm_info[0]["top_k_token"][1:]    # if state contains multiple llm toknes, take the first one
            
            alter_token_score = list()          # get semantic similarity score (cosine over vectors)
            for token in alter_tokens:      
                if not self.nlp_model(token).has_vector:    # if empty vectors enountered
                    alter_token_score.append(0)
                else:
                    alter_token_score.append(self.nlp_model(state.token).similarity(self.nlp_model(token)))
            # alter_token_score = [self.nlp_model(state.token).similarity(self.nlp_model(token)) for token in alter_tokens]

            alter_toekn_pos_tag = [self.nlp_model(token)[0].pos_ for token in alter_tokens] # if multiple token available, take the first one
            alter_token_prob = state.llm_info[0]["top_k_prob"]     # alter token probabilities
            for i, token in enumerate(alter_tokens):
                state.statistics.alter_token[token] = {"similarity": alter_token_score[i],
                                                        "pos_tag": alter_toekn_pos_tag[i],
                                                        "probability": alter_token_prob[i]}


    def split_states_by_sentence(self, states):
        """
        Splite states by sentences
        Return: segmented states(by sentence), index of end states
        """
        result = list()     # states split by sentences [[sentence 1], [sentence 2], ...]
        end_state_index = list()    # dix of states which are the ends of sentences
        new_start = 0
        for i, state in enumerate(states):
            if state.spacy_info["new_sentence"]:
                try: result.append(states[new_start:i+1])  # if the passage not end with "."
                except: result.append(states[new_start:i])
                end_state_index.append(i+1)
                new_start = i+1
        return result, end_state_index
    
    def compute_weighted_statistics(self):
        """
        Assign weighted probability, entropy to state with target POS tags
        """
        for i, state in enumerate(self.states[1:]):

            # if state with target tag also has children
            if (state.prop_tag.propstr in self.target_tags) and len(state.children)>0:
                children_states = [self.states[i] for i in list(state.children.values())]

                # wieghted prob = (1-weight)*state_prob + weight*avg(children_probs)
                state.statistics.weighted_prob = (1.0-self.children_weights)*state.probability + \
                        self.children_weights*np.average([child.probability for child in children_states])
                
                # wieghted entropy: same as above
                state.statistics.weighted_entropy = (1.0-self.children_weights)*state.entropy + \
                        self.children_weights*np.average([child.entropy for child in children_states])
            else:
                state.statistics.weighted_entropy = state.entropy
                state.statistics.weighted_prob = state.probability


    def _add_num_for_stat(self, num, list_of_stat):
        for i in list_of_stat:
            i.add_num(num)

    def compute_moving_statistics(self):
        """
        Compute interval/moving statistics for state probability and entropy
        """
        
        statistic_manager = dict()
        # passage level prob/entropy median
        doc_prob_median, doc_entropy_median = MedianFinder(), MedianFinder()

        for method in ["normal", "weighted"]:
            # moving statistics in sentence level
            if self.moving_mdoe == "sentence": 
                for sentence in self.sentence_state:
                    sentence_prob, sentence_entropy = MovingAverage(self.window_len),  MovingAverage(self.window_len)
                    for state in sentence:
                        # compute prob, entropy w.r.t prop_tag, either pos_tag or propstr
                        prop_tag = state.prop_tag.propstr
                        if prop_tag not in statistic_manager.keys():
                            statistic_manager[prop_tag] = {"probability": MovingAverage(self.window_len),
                                                        "entropy": MovingAverage(self.window_len)}
                        self._add_num_for_stat(state.entropy if method=="normal"else state.statistics.weighted_entropy, 
                                                [statistic_manager[prop_tag]["entropy"], 
                                                doc_entropy_median, sentence_entropy])
                        self._add_num_for_stat(state.probability if method=="normal"else state.statistics.weighted_prob,
                                            [statistic_manager[prop_tag]["probability"],
                                            doc_prob_median, sentence_prob])
                        # doc median
                        moving_doc_median = {"entropy":doc_entropy_median.findMedian(), "probability":doc_prob_median.findMedian()}
                        # sentence avg, rmse
                        sentence_entropy_avg, sentence_entropy_rmse = sentence_entropy.get_cur_state()
                        sentence_prob_avg, sentence_prob_rmse = sentence_prob.get_cur_state()
                        # tag avg, rmse
                        tag_entropy_avg, tag_entropy_rmse = statistic_manager[prop_tag]["entropy"].get_cur_state()
                        tag_prob_avg, tag_prob_rmse = statistic_manager[prop_tag]["probability"].get_cur_state()
                        
                        if method == "normal":
                            state.statistics.moving_doc_median = moving_doc_median
                            state.statistics.moving_rmse = {"sentence_entropy": sentence_entropy_rmse, "sentence_prob": sentence_prob_rmse,
                                        "tag_entropy": tag_entropy_rmse, "tag_prob": tag_prob_rmse}
                            state.statistics.moving_mean = {"sentence_entropy": sentence_entropy_avg, "sentence_prob": sentence_prob_avg,
                                        "tag_entropy": tag_entropy_avg, "tag_prob": tag_prob_avg}
                        else:                     
                            # weighted statistics
                            state.statistics.weighted_moving_doc_median = moving_doc_median
                            state.statistics.weighted_moving_rmse = {"sentence_entropy": sentence_entropy_rmse, "sentence_prob": sentence_prob_rmse,
                                        "tag_entropy": tag_entropy_rmse, "tag_prob": tag_prob_rmse}
                            state.statistics.weighted_moving_mean = {"sentence_entropy": sentence_entropy_avg, "sentence_prob": sentence_prob_avg,
                                        "tag_entropy": tag_entropy_avg, "tag_prob": tag_prob_avg}
            
            # moving statistics in passage level
            elif self.moving_mdoe == "passage":
                    doc_prob, doc_entropy = MovingAverage(self.window_len),  MovingAverage(self.window_len)
                    for state in self.states:
                        # compute prob, entropy w.r.t prop_tag, either pos_tag or propstr
                        prop_tag = state.prop_tag.propstr
                        if prop_tag not in statistic_manager.keys():
                            statistic_manager[prop_tag] = {"probability": MovingAverage(self.window_len),
                                                        "entropy": MovingAverage(self.window_len)}
                        self._add_num_for_stat(state.entropy if method=="normal"else state.statistics.weighted_entropy, 
                                                [statistic_manager[prop_tag]["entropy"], 
                                                doc_entropy_median, doc_entropy])
                        self._add_num_for_stat(state.probability if method=="normal"else state.statistics.weighted_prob,
                                            [statistic_manager[prop_tag]["probability"],
                                            doc_prob_median, doc_prob])
                        # doc median
                        moving_doc_median = {"entropy":doc_entropy_median.findMedian(), "probability":doc_prob_median.findMedian()}
                        # doc avg, rmse
                        doc_entropy_avg, doc_entropy_rmse = doc_entropy.get_cur_state()
                        doc_prob_avg, doc_prob_rmse = doc_prob.get_cur_state()
                        # tag avg, rmse
                        tag_entropy_avg, tag_entropy_rmse = statistic_manager[prop_tag]["entropy"].get_cur_state()
                        tag_prob_avg, tag_prob_rmse = statistic_manager[prop_tag]["probability"].get_cur_state()
                        
                        if method == "normal":
                            state.statistics.moving_doc_median = moving_doc_median
                            state.statistics.moving_rmse = {"doc_entropy": doc_entropy_rmse, "doc_prob": doc_prob_rmse,
                                        "tag_entropy": tag_entropy_rmse, "tag_prob": tag_prob_rmse}
                            state.statistics.moving_mean = {"doc_entropy": doc_entropy_avg, "doc_prob": doc_prob_avg,
                                        "tag_entropy": tag_entropy_avg, "tag_prob": tag_prob_avg}
                        else:                     
                            # weighted statistics
                            state.statistics.weighted_moving_doc_median = moving_doc_median
                            state.statistics.weighted_moving_rmse = {"doc_entropy": doc_entropy_rmse, "doc_prob": sentence_prob_rmse,
                                        "tag_entropy": tag_entropy_rmse, "tag_prob": tag_prob_rmse}
                            state.statistics.weighted_moving_mean = {"doc_entropy": doc_entropy_avg, "doc_prob": sentence_prob_avg,
                                        "tag_entropy": tag_entropy_avg, "tag_prob": tag_prob_avg}

            else:
                print("Mode not found! Either sentence or passage. \n")

class MedianFinder:
    # Refer to: https://github.com/criszhou/LeetCode-Python/blob/master/295.%20Find%20Median%20from%20Data%20Stream.py
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.upperHeap = [float('inf')]
        self.lowerHeap = [float('inf')]
        # lowerHeap's numbers are minus original numbers, because in Python heap is min-heap

        # always maintain that their lens are equal, or upper has 1 more than lower

    def add_num(self, num):
        """
        Adds a num into the data structure.
        :type num: int
        :rtype: void
        """
        upperMin = + self.upperHeap[0]
        lowerMax = - self.lowerHeap[0]

        if num > upperMin or (lowerMax<=num<=upperMin and len(self.upperHeap)==len(self.lowerHeap)):
            heappush(self.upperHeap, num)
        else:
            heappush(self.lowerHeap, -num)

        # maintain the invariant that their lens are equal, or upper has 1 more than lower
        if len(self.upperHeap)-len(self.lowerHeap) > 1:
            heappush( self.lowerHeap, -heappop( self.upperHeap ) )
        elif len(self.lowerHeap) > len(self.upperHeap):
            heappush( self.upperHeap, -heappop( self.lowerHeap ) )


    def findMedian(self):
        """
        Returns the median of current data stream
        :rtype: float
        """
        if len(self.upperHeap) == len(self.lowerHeap):
            upperMin = + self.upperHeap[0]
            lowerMax = - self.lowerHeap[0]
            return ( float(upperMin) + float(lowerMax) ) / 2.0
        else:
            assert len(self.upperHeap) == len(self.lowerHeap) + 1
            return float(self.upperHeap[0])
    
class MovingAverage:
    def __init__(self, window_size):
        """
        Compute moving average for given data array and window_size.
            When len(data) < window_size, compute moving average without window_size constraint
            When len(data) > window_size, the fisrt window_size - 1 components is computed by
                moving average without constraint also.
        """
        self.window_size = window_size
        self.values = np.zeros(window_size)
        self.sum = 0
        self.index = 0
        self.count = 0
        self.recorder = list()

    def add_num(self, num):
        if self.count < self.window_size:
            self.count += 1
        else:
            self.sum = self.sum - self.values[self.index]
        self.sum += num
        self.values[self.index] = num
        self.index = (self.index + 1) % self.window_size
        self.recorder.append(num)

    def get_average(self):
        if self.count == 0:
            return None
        return self.sum / self.count
    
    def get_cur_state(self):
        avg =  self.get_average()
        errors = np.array(self.recorder[-self.window_size:]) - avg
        squared_errors = errors ** 2
        mse = np.mean(squared_errors)
        return avg, np.sqrt(mse)
    
    def get_rmse(self):
        avg, rmse = self.get_cur_state()
        return rmse


def get_spacy_info(doc:spacy.tokens.doc.Doc):
    """
    Get information from spacy pipeline for every token
    """
    result = OrderedDict()
    left_idx = 0

    for i, token in enumerate(doc):
        right_idx = left_idx + len(token)
        result[(left_idx, right_idx)] = dict()
        # assign token attributes
        for key in SPACY_INFO_DICT:
            result[(left_idx, right_idx)][key] = getattr(token, key)
        # token’s immediate syntactic children
        result[(left_idx, right_idx)]["children"] = [child.i for child in token.children]
        # if is new sentnce
        if i > 0 and token.tag_ == ".":
            result[(left_idx, right_idx)]["new_sentence"] = True
        else:
            result[(left_idx, right_idx)]["new_sentence"] = False
        left_idx = right_idx
    return result

def get_llm_info(text:str, llm_token:list, top_k_prob:np.ndarray, top_k_token:np.ndarray):
    """
    Get information from LLM: token, top-k prob, top-k alternative token, entropy
    """
    result = OrderedDict()
    token_left_idx = 0
    text_left_inx = 0

    for i, token in enumerate(llm_token):
        token_right_idx = token_left_idx + len(token)
        word = text[token_left_idx: token_right_idx]
        
        # check token-word alignment
        if word != token:
            print(f"word-token mismatch: {word} - {token}")
            break
        # remove beginning space
        if token.startswith(" "):
            word = word[1:]
        # compute Shannon entropy
        token_entropy = uncertainty(top_k_prob[i,:])[1]
        # assign attributes to token 
        text_right_idx = text_left_inx + len(word)
        result[(text_left_inx, text_right_idx)] = {'text': word, 
                                                   "llm_token_idx": i,
                                                   "top_k_prob": top_k_prob[i,:].tolist(),
                                                   "top_k_token": top_k_token[i,:].tolist(),
                                                   "entropy": token_entropy}
        text_left_inx, token_left_idx = text_right_idx, token_right_idx
    return result

def align_llm_spacy_output(nlp_result:OrderedDict, llm_result:OrderedDict, verbose = False):
    """
    Aligh the result return by `get_spacy_info` and `get_llm_info`
    Return: merged token info -> semantics info + statistical info
    """
    result = OrderedDict()
    nlp_idx, llm_idx = np.array(list(nlp_result.keys())), np.array(list(llm_result.keys()))
    llm_start_idx, llm_end_idx = llm_idx[:,0], llm_idx[:,1]
    # word not found in llm, word not match in llm
    num_word_not_found, num_word_not_match = 0, 0

    for i, idx_range in enumerate(nlp_result):
        left_idx, right_idx = idx_range

        # check idx alignmnet
        # neither starting/ending idx exist in llm_idx list
        if (left_idx not in llm_idx) or (right_idx not in llm_idx):
            # if verbose: print(f"word: {nlp_result(idx_range)} not found in llm result")
            # void llm results to handle token-word mismatch
            # void_llm_result = [{"text": "N/A", "entropy": 0, "top_k_prob": [0,0,0,0,0], "top_k_token":['N/A', 'N/A', 'N/A', 'N/A']}]
            result[idx_range] = {"text": nlp_result[idx_range]["text"],
                                "nlp_result": nlp_result[idx_range],
                                "llm_result": [list(llm_result.items())[i][1]],
                                 "is_found": False, # not found at all
                                 "is_match": False}  # found but idx mismatch
            num_word_not_found += 1
            continue
        # idx mismatch: word in nlp contains multiple token in llm results
        elif idx_range not in llm_result:
            cross_llm_idx = [list(llm_start_idx).index(left_idx), list(llm_end_idx).index(right_idx)]
            
            result[idx_range] = {"text": nlp_result[idx_range]["text"],
                                "nlp_result": nlp_result[idx_range],
                                "llm_result": [llm_result[tuple(llm_idx[i])] for i in cross_llm_idx],
                                "is_found": True,
                                "is_match": False}
            num_word_not_match += 1
        # exact match
        else:
            result[idx_range] = {"text": nlp_result[idx_range]["text"],
                                "nlp_result": nlp_result[idx_range],
                                "llm_result": [llm_result[idx_range]],
                                "is_found": True,
                                "is_match": True}
    if verbose: print(f"num of word not found: {num_word_not_found} \n" + 
                      f"num of word not matched: {num_word_not_match} \n")
    return result

def get_model_state(align_output:OrderedDict): 
    """
    Parse aligned spacy/llm output and convert to state format
    Output: a list of `State` classes
    """
    result = []
    for i, token in enumerate(list(align_output.items())):
        token_info = token[1]
        # if multiple llm result available, take the first one for state entropy and prob 
        word_state = State(i,
                           token_info["text"], 
                           token_info["llm_result"][0]["entropy"],  
                           token_info["llm_result"][0]["top_k_prob"][0],
                           token_info["llm_result"],
                           token_info["nlp_result"],
                           token_info,
                           statistics=StateStatics())
        # try:
        word_state.prop_tag = PropTag(word_state.spacy_info["pos_"])
        # except:
        #     print(word_state.spacy_info["pos_"])
        #     break
        result.append(word_state)
    result = get_children(result, NOUN_CHILDREN, VERB_CHILDREN)
    return result


def get_children(states:State, NOUN_CHILDREN:set, VERB_CHILDREN:set):
    """
    Assign children token/state info to each state
    """
    for state in states:
        result = dict()
        children = state.spacy_info["children"]
        # attach children for NOUN state
        if len(children) > 0 and state.prop_tag.propstr=="NOUN":
            for child_idx in children:
                child = states[child_idx] # locate child crossponding state
                if child.prop_tag.pos_tag in NOUN_CHILDREN:
                    result[child.token] = child.idx     #[child_text: child_idx_in_states]
            state.children = result
            continue
        # attach children for VERB state
        elif len(children) > 0 and state.prop_tag.propstr=="VERB":
            for child_idx in children:
                child = states[child_idx] # locate child crossponding state
                if child.prop_tag.pos_tag in VERB_CHILDREN:
                    result[child.token] = child.idx     #[child_text: child_idx_in_states]
            state.children = result
        else:
            state.children = result
    return states

