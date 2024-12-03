# LLM_analysis

This repo contains code that are split into two parts: OpenAI related model analysis and open-source LLM analysis. Under the root directory, there are also two files `code_result_analysis.py` and `nlp_result_analysis.py` that can help with the analysis of the results. Besides these two files:

`eval_utils.py` provides evaluation utils for both parts.

`code_analysis` provides necessary utilitis from https://github.com/bigcode-project/bigcode-evaluation-harness to evaluate code generation tasks.


## OpenAI related model analysis

Please refer to the directory under `openAI_analysis`, especially the code under `GPT3_utils.py`.

Once the sampled/perturbed examples are collectedm both open-source models and close-source models can be analzed together using the functions in `utils.py`, such as `compute_VR` and `compute_VRO`.

## Open-source LLM analysis

Under the directory, `code_evaluation.py` and `llm_evaluation.py` are used to evaluate the target LLMs and store the corresponding results. 

`uncertainty_perturbation.py` is used to evaluate the perturbation-based uncertainty estimation.

`uncertainty_sample.py` is used to evaluate the sample-based uncertainty estimation.

Please refer to each file to check the meaning of each parameter.

