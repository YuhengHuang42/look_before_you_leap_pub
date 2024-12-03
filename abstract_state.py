import transformers
import spacy
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from heapq import heappush, heappop, heappushpop

from utils import compute_entropy
import comp_helper

SPACY_INFO_DICT = {"text":[], "lemma_":[], "pos_":[], 
                  "tag_":[], "dep_":[], "shape_":[], "is_alpha":[],
                 "is_stop":[]}

# Universal POS tags: https://universaldependencies.org/u/pos/
POS_NOUN = set(["NOUN", "PRON", "PROPN", "NN", "NNP", "NNPS", "NNS", "NE", "NNE"])
POS_VERB = set(["VERB", "AUX"])
NOUN_CHILDREN = set(["NOUN", "PRON", "PROPN", "ADJ"])
VERB_CHILDREN = set(["NOUN", "PRON", "PROPN", "ADV"])
ENDING_TAG = set(["."])

@dataclass(eq=False)
class PropTag:
    # Class to represent the property tag of token
    #pos_tags: list[str]
    pos_tag: str
    propstr: str
    
    def __init__(self, pos_tag):
        self.pos_tag = pos_tag
        #if token_info is None or len(token_info) == 0:
            #self.pos_tags = None
            #self.pos_tag = None
        #else:
            #pos_tags = [(i[0]['pos_'], i[1]) for i in token_info] # [(tag, length), ..]
            #self.pos_tags = pos_tags
            #if len(pos_tags) == 1:
            #    self.pos_tag = pos_tags[0][0]
            #else:
                # Multiple POS tags available, choose the maxium range one.
            #    self.pos_tag = max(pos_tags, key=lambda x: x[1])[0][0]
        self.propstr = self._get_str()
            
        
    def _hash_map_func(self, prop):
        # A helper function to map an equivalent property to the same hashable value
        if prop in POS_NOUN:
            return "NOUN"
        elif prop in POS_VERB:
            return "VERB"
        else:
            return prop
    
    def _get_str(self):
        if self.pos_tag is None:
            return "NONE"
        else:
            return self._hash_map_func(self.pos_tag)
        
    def __eq__(self, other):
        if self.pos_tag in POS_NOUN and other.pos_tag in POS_NOUN:
            return True
        elif self.pos_tag in POS_VERB and other.pos_tag in POS_VERB:
            return True
        elif self.pos_tag == other.pos_tag:
            return True
        else:
            return False
    
    def __hash__(self):
        return hash(self._hash_map_func(self.pos_tag))
    
    def __str__(self):
        return self.propstr

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
        
        self.available_member = ["moving_doc_median", "moving_rmse", "moving_mean"]
                                 #"weighted_moving_doc_median", "weighted_moving_rmse",
                                 #"weighted_moving_mean", #"alter_token", 
                                 #"weighted_prob", "weighted_entropy"]
    
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
    llm_info: list = None         # info from llm
    spacy_info: dict = None    # nlp_result
    token_info: list = None 
    weights: float= None            # weights assigned for syntactic children
    prop_tag: PropTag = None
    statistics: StateStatics = None
    children: dict = None           # syntactic children

necessary_member_list_api = ["text", "token", "top_k_prob", "top_k_token"]


def get_model_state(spacy_nlp, mode, mode_specific_info: dict = None):
    assert mode in set(["api", "opensource"])
    if mode == "api":
        assert "response" in mode_specific_info
        response = mode_specific_info["response"]
        for member in necessary_member_list_api:
            try:
                assert member in response
            except:
                print("Require member: ", member)
                raise Exception
        
        doc = spacy_nlp(response["text"])
        nlp_result = get_spacy_info_api(doc)
        llm_result = get_llm_info(response["text"], response['token'], np.array(response['top_k_prob']), np.array(response['top_k_token']))
        align_result = align_llm_spacy_output_api(nlp_result, llm_result, verbose=False)
        states = get_model_state_api(align_result)
    elif mode =="opensource":
        model_output = mode_specific_info["model_output"]
        assert "tokenizer" in mode_specific_info
        assert "model_output" in mode_specific_info
        assert "scores" in model_output or "llm_info" in model_output
        assert "decoded_word" in model_output
        
        sentences = "".join(model_output['decoded_word'][0])
        tokenizer = mode_specific_info["tokenizer"]
        sentences_offest = get_sentence_offset(model_output['decoded_word'][0])
        doc = spacy_nlp(sentences)
        spacy_info = get_spacy_info_opensource(doc)
        align_dict = align_llm_spacy_output_opensource(sentences_offest, spacy_info)
        if "llm_info" not in model_output:
            model_output = get_topk_token_output(model_output, tokenizer)
        states = get_model_state_opensoure(model_output, align_dict)
        
    return states
    

    
def get_sentence_offset(text: str):
    """
    Return [left_token_idx, right_token_idx] --> string mapping using transformers.PreTrainedTokenizerFast
    Delete the first space if there is any.
    """
    result = OrderedDict()
    #eof_special_case = False
    #if text.endswith("\n\n"):
        # \n\n as ends for tokenizer will yield [628]
        # \n\n[EOF] will yield [198, 198, EOF_IDX]
        # We treat \n\n as two separate tokens, so here we add EOF.
    #    text += tokenizer.eos_token
    #    eof_special_case = True
    pointer = 0
    for word in text:
        result[(pointer, len(word) + pointer)] = word 
        pointer += len(word)
    #input_info = tokenizer(text, return_offsets_mapping=True)
    #offset_mapping = input_info["offset_mapping"][:-1] if eof_special_case else input_info["offset_mapping"]
    
    #for mapping in input_info["offset_mapping"]:
    #    left, right = mapping
    #    word = text[left: right]
    #    if word.startswith(" "):
    #        left += 1
    #        word = word[1:]
    #    if left >= right:
    #        # Only space left
    #        continue
    #    result[(left, right)] = word
    return result

    
def get_spacy_info_opensource(doc:spacy.tokens.doc.Doc):
    """
    Get information from spacy pipeline for every token
    """
    result = OrderedDict()
    for i, token in enumerate(doc):
        if token.pos_ == "SPACE":
            continue
        left = token.idx
        right = left + len(token)
        result[(left, right)] = dict()
        for key in SPACY_INFO_DICT:
            result[(left, right)][key] = getattr(token, key)
        # token’s immediate syntactic children
        result[(left, right)]["children"] = [child.i for child in token.children]
        #print((left, right))
        #print(result[(left, right)]["children"])
        if i > 0 and token.tag_ in ENDING_TAG:
            result[(left, right)]["new_sentence"] = True
        else:
            result[(left, right)]["new_sentence"] = False
    return result

def align_llm_spacy_output_opensource(sentences_offest, spacy_info):
    """
    Aligh the result return by `get_sentence_offset` and `get_spacy_info`
    Return: sentence range key --> spacy token info
    """
    def compute_intersection_range(range1, range2):
        start = max(range1[0], range2[0])
        end = min(range1[1], range2[1])
        length = max(0, end - start)
        return length
    
    result = OrderedDict()
    for output_tok_range in sentences_offest:
        result[output_tok_range] = [[], []] # [token_info, spacy_info]
        for spacy_tok_range in spacy_info:
            if spacy_tok_range[0] >= output_tok_range[1]: # End of the search.
                break
            elif spacy_tok_range[1] <= output_tok_range[0]: # Skip the following statements
                continue
            elif spacy_tok_range[0] <= output_tok_range[0] and spacy_tok_range[1] >= output_tok_range[1]: # Contain
                range_len = compute_intersection_range(spacy_tok_range, output_tok_range)
                result[output_tok_range][0].append((spacy_info[spacy_tok_range], range_len))
                break
            elif spacy_tok_range[1] > output_tok_range[0] and spacy_tok_range[1] <= output_tok_range[1]: # Intersection
                range_len = compute_intersection_range(spacy_tok_range, output_tok_range)
                result[output_tok_range][0].append((spacy_info[spacy_tok_range], range_len))
                continue
            elif spacy_tok_range[0] >= output_tok_range[0] and spacy_tok_range[0] < output_tok_range[1]: # Intersection
                range_len = compute_intersection_range(spacy_tok_range, output_tok_range)
                result[output_tok_range][0].append((spacy_info[spacy_tok_range], range_len))
                continue
            else:
                print("output range: ", output_tok_range)
                print("spacy range: ", spacy_tok_range)
                raise Exception("Search Error")
        # spacy_info
        if len(result[output_tok_range][0]) == 0:
            # Not found
            result[output_tok_range][1] = None
        else:
            result[output_tok_range][1] = max(result[output_tok_range][0], key=lambda x: x[1])[0]
    return result

def get_children(states:State, NOUN_CHILDREN:set, VERB_CHILDREN:set):
    """
    Assign children token/state info to each state
    """
    for state in states:
        result = dict()
        if state.spacy_info is None:
            state.children = []
            continue
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

def get_topk_token_output(model_output, tokenizer, topk=5):
    """
    Get LLM info from open-source model.
    Args:
        model_output: should be torch.tensor in the format of (Token_num, 1, Embedding_length)
        tokenizer: the tokenizer model uses
        topk: save the topk token info
    """
    # squeeze without dim will yield error when
    # the input only contains one token (EOS)
    values, index = model_output["scores"].float().softmax(-1).topk(topk, dim=-1)
    # (Token_num, 1, Embedding_length)
    token_result = list()
    token_n = index.shape[0]
    for token_idx in range(token_n):
        token_result.append(tokenizer.batch_decode(index[token_idx].flatten()))
    values = values.tolist()
    model_output["llm_info"] = [{"top_k_token": token_result[i], 
                                 "top_k_prob": values[i]} for i in range(token_n)]
    return model_output

def get_model_state_opensoure(model_output, output_spacy_align: OrderedDict):
    """
    Parse the model output and return
    a list of `State` classes as output
    """
    result = list()
    words = np.array(model_output['decoded_word']).flatten()
    if "entropy" not in model_output:
        scores = model_output['scores']
        entropy = compute_entropy(scores)
    else:
        entropy = model_output['entropy']
    gen_probes = model_output['gen_probs'].flatten()
    llm_info = model_output["llm_info"]
    ordered_key_list = list(output_spacy_align) # Assume the dict is OrderedDict
    assert len(ordered_key_list) == len(words)
    for token_idx, word in enumerate(words):
        word_state = State(token_idx, 
                           word, 
                           float(entropy[token_idx]), 
                           float(gen_probes[token_idx]),
                           llm_info=[llm_info[token_idx]],
                           spacy_info = output_spacy_align[ordered_key_list[token_idx]][1],
                           token_info = output_spacy_align[ordered_key_list[token_idx]][0],
                           statistics=StateStatics()
                           )
        pos = word_state.spacy_info["pos_"] if word_state.spacy_info is not None else None
        word_state.prop_tag = PropTag(pos)
        result.append(word_state)
    # FIXME: align state index and spacy index
    #result = get_children(result, NOUN_CHILDREN, VERB_CHILDREN)
    return result



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

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.average(data)
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, mode='valid')

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
    



''' This function is slower then MovingAverage class when there is a
small number of data but the computation would be repeated many times.

def moving_average(data, window_size):
    """
    Compute moving average for given data array and window_size.
        When len(data) < window_size, compute moving average without window_size constraint
        When len(data) > window_size, the fisrt window_size - 1 components is computed by
            moving average without constraint also.
    """
    if len(data) < window_size:
        return [np.average(data[:i+1]) for i in range(len(data))]
    # Calculate the moving averages for the shorter windows
    shorter_moving_averages = [np.average(data[:i+1]) for i in range(window_size-1)]
    weights = np.repeat(1.0, window_size) / window_size
    full_window_moving_average = np.convolve(data, weights, mode='valid')
    # Concatenate the results
    return np.concatenate((shorter_moving_averages, full_window_moving_average))
'''

class StateOverview:
    def __init__(self, states:list, window_len:int, target_tags:set, 
                 children_weights:float, mode:str, nlp_model,
                 disable_alter_comp=False, disable_child_comp=False):
        self.states = states
        self.window_len = window_len
        self.target_tags = target_tags    # copmute wighted statistics for tokens with target tags 
        self.children_weights = children_weights  # weights for dependent child states
        self.sentence_state, self.end_state_index = self.split_states_by_sentence(states)
        self.moving_mdoe = mode         # passage level or sentence level
        self.nlp_model = nlp_model      # spacy model to process alternative tokens
        self.prop_link_map = self.generate_property_link(states)
        
        self.compute_moving_statistics()
        if not disable_child_comp:
            self.compute_weighted_statistics()
        if not disable_alter_comp:
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
                
    def generate_property_link(self, states):
        result = dict()
        for state in states:
            if state.prop_tag not in result:
                result[state.prop_tag] = list()
            result[state.prop_tag].append(state.idx)
        return result
    
    def split_states_by_sentence(self, states):
        result = list()
        temp = list()
        seps = list()
        for state in states:
            if state.spacy_info is not None and state.spacy_info["new_sentence"]: # sentence closing
                result.append(temp)
                seps.append(state)
                temp = list()
            else:
                temp.append(state)
        if len(temp) > 0:
            result.append(temp)
        return result, seps
    
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

    def compute_weighted_statistics(self):
        """
        Assign weighted probability, entropy to state with target POS tags
        """
        for state in self.states:
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



# ====== Utility for processing API ======

def get_spacy_info_api(doc:spacy.tokens.doc.Doc):
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
    Used in processing API calling
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
        token_entropy = comp_helper.uncertainty(top_k_prob[i,:])[1]
        # assign attributes to token 
        text_right_idx = text_left_inx + len(word)
        result[(text_left_inx, text_right_idx)] = {'text': word, 
                                                   "llm_token_idx": i,
                                                   "top_k_prob": top_k_prob[i,:].tolist(),
                                                   "top_k_token": top_k_token[i,:].tolist(),
                                                   "entropy": token_entropy}
        text_left_inx, token_left_idx = text_right_idx, token_right_idx
    return result

def align_llm_spacy_output_api(nlp_result:OrderedDict, llm_result:OrderedDict, verbose = False):
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

def get_model_state_api(align_output:OrderedDict): 
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
                           llm_info=token_info["llm_result"],
                           spacy_info=token_info["nlp_result"],
                           token_info=token_info,
                           statistics=StateStatics())
        # try:
        word_state.prop_tag = PropTag(word_state.spacy_info["pos_"])
        # except:
        #     print(word_state.spacy_info["pos_"])
        #     break
        result.append(word_state)
    result = get_children(result, NOUN_CHILDREN, VERB_CHILDREN)
    return result