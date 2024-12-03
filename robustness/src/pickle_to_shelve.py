import pandas as pd
import shelve


mode_list = ['adv_robust', 'ood_robust']
model_list = ['text-curie-001', 'text-davinci-003']

# get all result path list
result_path_list = []
for mode in mode_list:
	if mode == 'adv_robust':
		task_list = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']
		dataset_list = ['alpaca', 'stable-vicuna', 'vicuna', 'original']
		for dataset in dataset_list:
            for task in task_list:
            	for model in model_list:
            		result_path_list.append(f'result/{mode}/{dataset}/{task}/{model}')
	elif mode == 'ood_robust':
		task_list = ["style", "knowledge"]
		for task in task_list:
			if task == 'style':
	            dataset_list = ['original', 'augment', 'bible_p0.6', 'bible_p0', 'romantic_p0.6', 'romantic_p0', 
	                                'shake_p0.6', 'shake_p0', 'shake_w', 'tweet_p0.6', 'tweet_p0']
	        elif task == 'knowledge':
	            dataset_list = ['qa_2020', 'qa_2023']
	        for dataset in dataset_list:
	        	result_path_list.append(f'result/{mode}/{dataset}/{task}/{model}')


def get_non_nan_indices(result):
	non_nan_indices = []
	for idx in range(len(result)):
		if not pd.isna(result.at[idx, 'prompt']):
			non_nan_indices.append(idx)
	return idx


for result_path in result_path_list:
	result = pd.read_pickle(result_path)
	get_non_nan_indices(result)