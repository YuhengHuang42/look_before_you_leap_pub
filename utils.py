import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from numpy import log2, prod, mean
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_TOKENIZER_INPUT_LEN = 800 # 1024 is the maximum GPT-2 can handle
MAX_DEFAULT_TOKEN_LEN = 100
MANDATORY_NEW_LEN = 50 # If the input already has MAX_len tokens, we will allow the model to generate at most 50 new tokens

LLAMA2_CHAT_SYSTEM = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_PROMPT_LENGTH = 137

from enum import Enum
class ModelSource(Enum):
    OPENAI = "openai"
    OPENSOURCE = "opensource"
    
# supplymnet functions for entropy calculation
def shannon_entropy(problist):
    """Shannon Entropy: negative sum over all probabilities*log2_probabilities
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    return -1 * sum([prob * log2(prob) for prob in problist])

def compute_entropy(scores):
    """
    Compute entropy by given scores over the entire vocabularies.
    Args:
        scores: the scores for list of tokens returned from the model.
    Return:
        The entropy
    """
    # Reference: https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/generation/logits_process.py#L330
    
    # We do the following computation in 
    # for-loop because in some cases,
    # it may contain a list of 
    # tensors whose sizes are not the same
    ent_list = list()
    for i in scores:
        normalized = torch.nn.functional.log_softmax(i, dim=-1)
        p = torch.exp(normalized)
        ent_list.append(-(normalized * p).nansum(-1, keepdim=True))
    ent = torch.stack(ent_list).reshape(-1)
    return ent

def split_by_highest_entropy(y):
    """
    This function split a list of entropy into sub-lists.
    For two adjacent list, the first element of the latter
    mush be higher than any elements in the previous list.
    Args:
        y: 1-D list for Entropy
    Return:
        list of sub lists
    """
    sep_list = list()
    last_point = y[0]
    temp = [0]
    for idx, item in enumerate(y[1:]):
        if item < last_point:
            temp.append(idx + 1)
        else:
            last_point = item
            sep_list.append(temp)
            temp = [idx + 1]
    sep_list.append(temp)
    return sep_list

def process_sampleonly_output(input_len, sampleonly, tokenizer):
    gen_sequences = sampleonly.sequences[:, input_len:].cpu()
    result = list()
    for idx, seq in enumerate(sampleonly.scores):
        t = seq.cpu().reshape(-1)
        indices = torch.arange(len(t))[torch.isfinite(t)]
        values = t[indices].float()
        output_token = gen_sequences[0][idx]
        output_token_idx = int((indices == output_token).nonzero())
        token_prob_dist = values.softmax(-1)
        result.append([indices, values, token_prob_dist[output_token_idx]])
    return {"scores": {"indices": [i[0] for i in result], 
                       "values": [i[1] for i in result]}, 
            "gen_probs": torch.stack([i[2] for i in result]).float(),
            "gen_sequences": gen_sequences,
            "decoded_text":[tokenizer.decode(i) for i in gen_sequences],
            "decoded_word":[[tokenizer.decode(word) for word in i] for i in gen_sequences]}

def process_naive_output(input_len, outputs, tokenizer):
    gen_sequences = outputs.sequences[:, input_len:].cpu()
    scores = outputs.scores
    #if gen_sequences.flatten()[-1] == tokenizer.eos_token_id:
    #    gen_sequences = gen_sequences[:, :-1]
    #    scores = scores[:-1]
    probs = torch.stack(scores, dim=1).float().softmax(-1).cpu()
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    return {"scores": torch.stack(scores).float().cpu(), "gen_probs":gen_probs, 
            "gen_sequences":gen_sequences,
            "decoded_text":[tokenizer.decode(i) for i in gen_sequences],
            "decoded_word":[[tokenizer.decode(word) for word in i] for i in gen_sequences]}

def parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)
    
def generate_gen_kwargs(do_sample, temperature, top_p, top_k, max_length, task, tokenizer):
    """
    Used in code generation
        top_k: Optional[int] = field(
                default=0, metadata={"help": "Top-k parameter used for generation."}
            )
        top_p: Optional[float] = field(
            default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
        )
    """
    gen_kwargs = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_length": max_length,
    }
    if task.stop_words:
        if tokenizer.eos_token:
            task.stop_words.append(tokenizer.eos_token)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
        )
    return gen_kwargs

class TemplatePrcessor():
    def __init__(self, model_type, disable_template=False, tokenizer=None):
        self.model_type = model_type
        self.disable_template = disable_template
        self.tokenizer = tokenizer

    def wrap_in_template(self, input_text):
        """
        meta-llama/Llama-2-7b-chat-hf is a fine-tuned model
        based on template. It is thus necessary to add template
        for the input.
        Reference: https://github.com/huggingface/blog/blob/main/llama2.md
        """
        if self.disable_template:
            return input_text
        # special tokens used by llama 2 chat
        if self.model_type ==  "llama2": #"meta-llama/Llama-2-7b-chat-hf":
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            return_message = B_INST + " " + B_SYS + LLAMA2_CHAT_SYSTEM + E_SYS + input_text.strip() + " " + E_INST
            return return_message
        elif self.model_type == "phi3":
            assert self.tokenizer is not None
            chat = [
                {"role": "user", "content": input_text.strip()},
            ]
            # https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
            input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        elif self.model_type == "llama3":
            assert self.tokenizer is not None
            chat = [
                {"role": "user", "content": input_text.strip()},
            ]
            # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
            input_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return input_text
        
class LLMInference():
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
    
    def forward(self, text, mode="nlp", mode_specific_args=None):
        if mode == "nlp":
            mode_specific_args = copy.deepcopy(mode_specific_args)
            sampling = "naive"
            max_length = MAX_DEFAULT_TOKEN_LEN
            # Default: max_length is set
            if mode_specific_args is None:
                mode_specific_args = {"max_length": max_length}
            else:
                # If mode_specific_args is not None.
                if "max_length" not in mode_specific_args and "max_new_tokens" not in mode_specific_args:
                    mode_specific_args["max_length"] = max_length
                    if sampling in mode_specific_args:
                        sampling = mode_specific_args["sampling"]
                if "max_new_tokens" in mode_specific_args:
                    if "max_length" in mode_specific_args:
                        # When max_new_tokens are set, max_length
                        # is only used to truncate the input.
                        max_length = mode_specific_args["max_length"]
                        mode_specific_args.pop("max_length")
                    else:
                        max_length = MAX_TOKENIZER_INPUT_LEN
                elif "max_length" in mode_specific_args:
                    # When there is not max_new_tokens
                    max_length = mode_specific_args["max_length"]
            with torch.inference_mode():
                tokenized_input = self.tokenizer(text=text, truncation=True, return_tensors="pt", max_length=max_length) # Batch size 1
                # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
                input_ids = tokenized_input["input_ids"].to(self.model.device)
                input_len = tokenized_input.attention_mask.sum()
                if input_len >= max_length:
                    mode_specific_args.pop("max_length", None)
                    mode_specific_args["max_new_tokens"] = MANDATORY_NEW_LEN
                if sampling == "naive":
                    outputs = self.model.generate(input_ids, return_dict_in_generate=True, 
                                            output_scores=True, early_stopping=True,
                                            pad_token_id=self.tokenizer.eos_token_id, **mode_specific_args)
                    result = process_naive_output(input_ids.shape[-1], outputs, self.tokenizer)
                elif sampling == "topk":
                    outputs = self.model.generate(input_ids, return_dict_in_generate=True, 
                                            output_scores=True, early_stopping=True,
                                            pad_token_id=self.tokenizer.eos_token_id, do_sample=True, top_k=30, **mode_specific_args)
                    result = process_sampleonly_output(input_ids.shape[-1], outputs, self.tokenizer)
            return result
        elif mode == "code":
            # text is prompt_contents in https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/lm_eval/utils.py#L12
            task = mode_specific_args["task"]
            temp_args = {key: mode_specific_args[key] for key in mode_specific_args if key != "task"}
            mode_specific_args = copy.deepcopy(temp_args)
            mode_specific_args["task"] = task
            gen_kwargs = mode_specific_args["gen_kwargs"]
            prefix = mode_specific_args["prefix"]
            #max_length = mode_specific_args["max_length"]
            max_length = gen_kwargs["max_length"]
            if "max_new_tokens" in gen_kwargs:
                gen_kwargs.pop("max_length", None)
            batch_size = mode_specific_args["batch_size"]
            postprocess = mode_specific_args["postprocess"] # True or False
            sample_idx = mode_specific_args["sample_idx"] # Idx of the instance in the dataset
            if isinstance(text, str):
                infill = False
                prompt = prefix + text
            elif isinstance(text, dict):
                assert set(text.keys()) == {"prefix", "suffix"}
                infill = True
                prompt = self._make_infill_prompt(
                                    **text, preprefix=prefix
                                )
            else:
                raise ValueError("mode must be either nlp or code")
            if infill:
                return_token_type_ids = False
            else:
                return_token_type_ids = None
            with torch.inference_mode():
                tokenized_input = self.tokenizer(
                    prompt,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                    return_token_type_ids=return_token_type_ids,
                )
                input_ids = tokenized_input["input_ids"].to(self.model.device)
                input_len = tokenized_input.attention_mask.sum()
                if input_len >= max_length:
                    gen_kwargs.pop("max_length", None)
                    gen_kwargs["max_new_tokens"] = MANDATORY_NEW_LEN
                gen_kwargs["stopping_criteria"][0].start_length = input_len
                outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=tokenized_input.attention_mask,
                                pad_token_id=self.tokenizer.eos_token_id,
                                num_return_sequences=batch_size,
                                return_dict_in_generate=True,
                                output_scores=True,
                                **gen_kwargs,
                            )
                collected_outputs = process_naive_output(input_len, outputs, self.tokenizer)
                generated_tokens = outputs.sequences.cpu().numpy()[0] # Only one sample
                result = list()
                #del input_ids
                gen_code = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                if infill or self.tokenizer.eos_token in task.stop_words:
                    if infill:
                        gen_code = parse_infill(gen_code, self.tokenizer)
                if not infill:
                    gen_code = gen_code[len(prefix) :] # Prefix is for the prompt template
                if postprocess:
                    result.append(
                        task.postprocess_generation(gen_code, int(sample_idx))
                        )
                else:
                    result.append(gen_code)
                collected_outputs["final_result"] = result
            return collected_outputs
        #probs = torch.stack(outputs.scores, dim=1).softmax(-1)
        #gen_sequences = outputs.sequences[:, input_ids.shape[-1]:]
        #gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

        #if normalize_logits:
        #    scores = scores.reshape(-1, self.vocab_size, scores.shape[-1])
        #    scores = torch.nn.functional.log_softmax(scores, dim=1)
        #    scores = scores.reshape(-1, scores.shape[-1])

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

def load_tokenizer(model_type, model_path=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if model_type == "gpt2":
        from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    elif model_type == "llama":
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_type == "llama2":
        from transformers import AutoTokenizer
        # Please follow the official instructions to download the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    elif model_type == "codegen":
        
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-mono")
    elif model_type =="santacoder":
        
        tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
    elif model_type == "incoder":
        
        tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-6B")
    elif model_type == "codellama":
         
         tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    elif model_type == "deepseekcoder":
        model_id = "deepseek-ai/deepseek-coder-6.7b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    elif model_type == "codeqwen1.5":
        model_id = "Qwen/CodeQwen1.5-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_type == "starcoder2":
        model_id = "bigcode/starcoder2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_type == "llama3":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_type == "phi3":
        model_id = "microsoft/Phi-3-small-8k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    elif model_type == "gemma2":
        model_id = "google/gemma-2-9b"
        # https://huggingface.co/google/gemma-2-9b for bfloat16
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        raise Exception(NotImplementedError)
    return tokenizer

def load_opensource_model(model_type, model_path=None, parallel=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if parallel:
        args = {"device_map": "auto"}
    else:
        args = {}
    if model_type == "gpt2":
        from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        model.eval()
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "llama":
        assert model_path is not None
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, **args)
        model.eval()
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "llama2":
        # Please follow the official instructions to download the model
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, **args)
        model.eval()
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "codegen":
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-mono")
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-6B-mono", torch_dtype=torch.float16, **args)
        model.eval()
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type =="santacoder":
        tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
        # Do not turn the model into float16
        # We encoure issues similar to:
        # https://github.com/huggingface/transformers/issues/15169#issuecomment-1346736724
        model = AutoModelForCausalLM.from_pretrained("bigcode/santacoder", trust_remote_code=True, **args)
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "incoder":
        model = AutoModelForCausalLM.from_pretrained("facebook/incoder-6B", revision="float16", torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, **args)
        tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-6B")
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "codellama":
        model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf", torch_dtype=torch.float16, **args)
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "deepseekcoder":
        model_id = "deepseek-ai/deepseek-coder-6.7b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, **args)
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "codeqwen1.5":
        model_id = "Qwen/CodeQwen1.5-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **args)
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "starcoder2":
        model_id = "bigcode/starcoder2-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **args)
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "llama3":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct for bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **args)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "phi3":
        model_id = "microsoft/Phi-3-small-8k-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2", 
                                                     trust_remote_code=True, **args)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        wrapper_model = LLMInference(model, tokenizer)
    elif model_type == "gemma2":
        model_id = "google/gemma-2-9b"
        # https://huggingface.co/google/gemma-2-9b for bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **args)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        wrapper_model = LLMInference(model, tokenizer)
    else:
        raise Exception(NotImplementedError)
    return (wrapper_model, tokenizer)

import logging
class FileFilter(logging.Filter):
    def __init__(self, filename):
        super(FileFilter, self).__init__()
        self.filename = filename
    def filter(self, record):
        return record.filename == self.filename

def setupLogging(log_file, file_filter=None, debug=False, multiprocess=False):
    import sys
    if multiprocess:
        import multiprocessing
        logger = multiprocessing.get_logger()
    else:
        logger = logging.getLogger()
    logLevel = logging.DEBUG if debug else logging.INFO
    logger.setLevel(logLevel)
    logFormat = "%(asctime)s [%(levelname)s] %(message)s"
    formatter = logging.Formatter(logFormat)
    #logging.basicConfig( stream=sys.stderr, level=logLevel, format=logFormat )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logLevel)
    file_handler.setFormatter(formatter)
    if file_filter is not None:
        file_handler.addFilter(file_filter)
    logger.addHandler(file_handler)
    return (logger, file_handler)

def divide_list_equally(lst, divider):
    length = len(lst)
    if length < divider:
        return [[i] for i in range(length)]
    chunk_size = length // divider
    remainder = length % divider

    divided_list = []
    start = 0

    for i in range(divider):
        chunk_size_with_remainder = chunk_size + 1 if i < remainder else chunk_size
        end = start + chunk_size_with_remainder
        divided_list.append(lst[start:end])
        start = end

    return divided_list

def compute_VR(inf_result, dist_func, w=1):
    # inf_result: the result returned from MC dropout
    # inf_result: [batch_size, inference_time, ...]
    batch_size = len(inf_result)
    inf_time = len(inf_result[0])
    result = []
    for batch_idx in range(batch_size):
        final_score = 0
        for first_T in range(inf_time):
            similarity = 0
            for second_T in range(inf_time):
                if first_T == second_T:
                    continue
                dist = dist_func(
                        inf_result[batch_idx][first_T],
                        inf_result[batch_idx][second_T])
                similarity +=  (1 - dist)
            similarity /= (inf_time - 1)
            final_score += w * similarity
        result.append(1 - final_score / inf_time)
    return result


def compute_VRO(inf_result, lm_result, dist_func):
    # inf_result: the result returned from MC dropout [batch_size, infer_time, ...]
    # lm_result: the result returned from normal inference. [batch_size, ...]
    batch_size = len(inf_result)
    inf_time = len(inf_result[0])
    result = []
    for batch_idx in range(batch_size):
        final_score = 0
        if type(lm_result[batch_idx]) is list:
            assert len(lm_result[batch_idx]) == 1
            referece_result = lm_result[batch_idx][0]
        else:
            referece_result = lm_result[batch_idx]
        for T in range(inf_time):
            try:
                final_score += (1 -
                                dist_func(inf_result[batch_idx][T], referece_result))
            except:
                print(inf_result)
                print(inf_result.shape)
                print(lm_result.shape)
        final_score /= inf_time
        result.append(1 - final_score)
    return result

def load_shelve_and_resume(dir_path):
    """
    Looping through the files under the dir_path. If there is more than
    one shelve file, raise an Exception. Otherwise, record the already
    saved count using len(loaded_data) and return the next index
    to resume.
    """
    import os
    import shelve
    # Get list of all shelve files in the directory
    shelve_files = [f for f in os.listdir(dir_path) if f.endswith('.db') or f.endswith('.dat')]

    # If there's more than one shelve file, raise an exception
    if len(shelve_files) > 1:
        raise Exception("More than one shelve file found in the directory.")
    
    # If no shelve file is found, return 0 as the starting index
    if not shelve_files:
        return 0
    
    # Load the shelve file
    shelve_file = os.path.splitext(shelve_files[0])[0]
    path = os.path.join(dir_path, shelve_file)
    with shelve.open(path) as db:
        loaded_data = list(db.keys())
    
    # Return the next index to resume
    return len(loaded_data), path