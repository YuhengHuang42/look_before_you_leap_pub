import shelve
import json
import csv


model_list = ['llama', 'llama2', 'text-curie-001', 'text-davinci-003']
uncertainty_method_list = ['raw_inference', 'sample_uncertainty']


def get_adv_dataset_length(dataset, task):
    dataset_path = f'../../DecodingTrust/data/adv-glue-plus-plus/data/{dataset}.json'
    if dataset == 'original': dataset_path = f'../../DecodingTrust/data/adv-glue-plus-plus/data/alpaca.json'
    with open(dataset_path) as f:
        datasets = json.load(f)
    dataset = datasets[task]
    return len(dataset)


def check_adv_result():
    print('-----checking adv robust result-----')
    task_list = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']
    dataset_list = ['alpaca', 'stable-vicuna', 'vicuna', 'original']
    for model in model_list:
        print(f'for model: {model}')
        for dataset in dataset_list:
            print(f'    for dataset: {dataset}')
            for task in task_list:
                print(f'        for task: {task}')
                for uncertainty_method in uncertainty_method_list:
                    output_path = f'result/adv_robust/{dataset}/{task}/{model}/{uncertainty_method}'
                    try:
                        result = shelve.open(output_path, 'r')
                        result_length = len(result)
                        result.close()
                    except:
                        result_length = 0
                    end = ' '*9 if uncertainty_method == 'raw_inference' else ' '*4
                    print(f'            for uncertainty_method: {uncertainty_method}', end=end)
                    print(f'{result_length}/{get_adv_dataset_length(dataset,task)} completed')


def get_ood_dataset_length(dataset, task):
    dataset_question = []
    if task == 'knowledge':
        data_file = f'../../DecodingTrust/data/ood/knowledge/{args.dataset}.jsonl'
        with open(data_file) as f:
            for line in f.readlines():
                dataset_question.append(json.loads(line))
    elif task == 'style':
        data_file = f'../../DecodingTrust/data/ood/styles/dev_{dataset}.tsv'
        if dataset == 'original': data_file = f'../DecodingTrust/data/ood/styles/dev.tsv'
            with open(data_file) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t')
                for row in reader:
                    dataset_question.append(row)
    return len(dataset_question)

def check_ood_result():
    print('-----checking ood robust result-----')
    task_list = ["style", "knowledge"]
    for model in model_list:
        print(f'for model: {model}')
        for task in task_list:
            print(f'    for task: {task}')
            if task == 'style':
                dataset_list = ['original', 'augment', 'bible_p0.6', 'bible_p0', 'romantic_p0.6', 'romantic_p0', 
                                'shake_p0.6', 'shake_p0', 'shake_w', 'tweet_p0.6', 'tweet_p0']
            elif task == 'knowledge':
                dataset_list = ['qa_2020', 'qa_2023']
            for dataset in dataset_list:
                print(f'        for dataset: {dataset}')
                for uncertainty_method in uncertainty_method_list:
                    output_path = f'result/ood_robust/{dataset}/{task}/{model}/{uncertainty_method}'
                    try:
                        result = shelve.open(output_path, 'r')
                        result_length = len(result)
                        result.close()
                    except:
                        result_length = 0
                    end = ' '*9 if uncertainty_method == 'raw_inference' else ' '*4
                    print(f'            for uncertainty_method: {uncertainty_method}', end=end)
                    print(f'{result_length}/{get_adv_dataset_length(dataset,task)} completed')


if __name__ == '__main__':
    check_adv_result()
    print('\n\n\n')
    check_ood_result()