import sys
sys.path.append('../')
from eval_utils import Evaluator
from sentence_transformers import SentenceTransformer
import spacy
import shelve
import pandas as pd
import numpy as np
import argparse
import utils
from GPT3_utils import get_feature, get_response


"""
python3 GPT3_evaluation.py --output_path ../data/qa/wiki_qa_test --dataset_path ../data/qa/wiki_qa_test.csv \
    --mode qa --log_file log/gpt3_wiki_qa_test.log

"""



"""
Use this script to evaluate the performance of GPT-3 on given tasks and datasets
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--simplified", type=bool, default=True)
    parser.add_argument("--dataset_path", type=str, default="eli5_category_test.csv")
    parser.add_argument("--print_freq", type=int, default=10, help="Every length/n evaluation will be printed")
    parser.add_argument("--mode", type=str, default='qa', choices=["qa", "summarization", "translation", "code"])
    parser.add_argument("--log_file", type=str, default="GPT3_evaluation.log")
    parser.add_argument("--type", type=str, default="naive", choices=["naive", "sample", "perturbation"])
    parser.add_argument("--instance_num", type=float, default=100)
    parser.add_argument("--api_key", type=str)
    args = parser.parse_args()

    file_filter = None
    logger, file_handler = utils.setupLogging(args.log_file, file_filter)
    logger.info("Running %s" % " ".join(sys.argv))
    logger.info("Using GPT-3 model")

    nlp = spacy.load(args.spacy_model)
    sentencemodel = SentenceTransformer(args.sentence_model)
    # sentencemodel.to(args.eval_device)
    evaluator = Evaluator(nlp, sentencemodel, mode=args.mode)
    
    new_df = pd.read_csv(args.dataset_path)
    new_df = new_df.loc[0:args.instance_num]
    dataset_len = len(new_df)

    if args.mode == "qa":
        mode_specific_args = {"api_key": args.api_key,
                            "max_tokens": 100,
                            "temp": 0.0}
    elif args.mode == "summarization" or args.mode == "translation":
        mode_specific_args = {"api_key": args.api_key,
                            "max_tokens": 650,
                            "temp": 0.0}
    elif args.mode == "code":
        if args.dataset == "mbpp":
            from code_analysis import mbpp
            logger.info("Evaluating MBPP")
            task = mbpp.MBPP()
        elif args.dataset == "humaneval":
            from code_analysis import human_eval
            logger.info("Evaluating Human Eval")
            task = human_eval.HumanEval()
            
        dataset = task.get_dataset()
        dataset_len = len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(dataset_len)]
        mode_specific_args = {"api_key": args.api_key,
                                "max_tokens": 650,
                                "temp": 0.0}
        
    save_path = args.output_path
    ans_recored = shelve.open(save_path)
    print_threshold = dataset_len // args.print_freq

    if args.mode in ["qa", "summarization", "translation"]:
        for index, row in new_df.iterrows():
            question = row["question"]
            answer = row["answer"]
            response = get_response(question, 
                                    temperature=mode_specific_args["temp"], 
                                    API_key=mode_specific_args["api_key"],
                                    max_tokens=mode_specific_args["max_tokens"])
            infer_output = dict()
            infer_output["evaluation"] = evaluator.compute_passage_score(response["text"], answer)
            infer_output["evaluation"]["task_metric"] = evaluator.compute_task_specific_score(response["text"], answer)
            infer_output["raw_response"] = response
            
            if args.simplified:
                feature_extractor = get_feature(response)
                infer_output["entropy"] = feature_extractor.get_entropy()
                # infer_output.pop('scores')
            infer_output['evaluation']['input'] = row["question"]
            infer_output['evaluation']['gt'] = row["answer"]
            ans_recored[str(index)] = infer_output
            if index % print_threshold == 0:
                logger.info(f"Processed {index}/{dataset_len} samples")
                
    elif args.mode == "code":
        for index in range(dataset_len):
            mode_specific_args["sample_idx"] = index
            prompt = task.get_prompt(dataset[index])
            reference = references[index]
            response = get_response(prompt, 
                                    temperature=mode_specific_args["temp"], 
                                    API_key=mode_specific_args["api_key"],
                                    max_tokens=mode_specific_args["max_tokens"])
            infer_output = dict()
            infer_output["evaluation"] = task.process_results([response["text"]], [reference])
            if args.simplified:
                feature_extractor = get_feature(response)
                infer_output["entropy"] = feature_extractor.get_entropy()
                # infer_output.pop('scores')
            infer_output['evaluation']['input'] =prompt
            infer_output['evaluation']['gt'] = reference
            infer_output['evaluation']['response'] = response["text"]
            infer_output["raw_response"] = response
            ans_recored[str(index)] = infer_output
            if index % print_threshold == 0:
                logger.info(f"Processed {index}/{dataset_len} samples")
    else:
        print("Task not found")
    
    ans_recored.close()
    logger.info("Processing finished, saving to: {}".format(save_path))
    file_handler.close()