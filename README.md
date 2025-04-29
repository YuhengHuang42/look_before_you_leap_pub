# LLM_analysis

This repo contains code that is split into two parts: OpenAI-related model analysis and open-source LLM analysis. Under the root directory, there are also two files `code_result_analysis.py` and `nlp_result_analysis.py` that can help with the analysis of the results. Besides these two files:

`eval_utils.py` provides evaluation utils for both parts.

`code_analysis` provides necessary utilities from https://github.com/bigcode-project/bigcode-evaluation-harness to evaluate code generation tasks.

More results and experiment details are provided on our website: https://sites.google.com/view/llm-uncertainty 

## OpenAI related model analysis

Please refer to the directory under `openAI_analysis`, especially the code under `GPT3_utils.py`.

Once the sampled/perturbed examples are collectedm both open-source models and close-source models can be analzed together using the functions in `utils.py`, such as `compute_VR` and `compute_VRO`.

## Open-source LLM analysis

Under the directory, `code_evaluation.py` and `llm_evaluation.py` are used to evaluate the target LLMs and store the corresponding results. 

`uncertainty_perturbation.py` is used to evaluate the perturbation-based uncertainty estimation.

`uncertainty_sample.py` is used to evaluate the sample-based uncertainty estimation.

Please refer to each file to check the meaning of each parameter.



## Citation

If you find the related resources and code useful, please cite our paper at TSE, thank you!

```
@ARTICLE{huang2025lbyp,
  author={Huang, Yuheng and Song, Jiayang and Wang, Zhijie and Zhao, Shengming and Chen, Huaming and Juefei-Xu, Felix and Ma, Lei},
  journal={IEEE Transactions on Software Engineering}, 
  title={Look Before You Leap: An Exploratory Study of Uncertainty Analysis for Large Language Models}, 
  year={2025},
  volume={51},
  number={2},
  pages={413-429},
  keywords={Uncertainty;Estimation;Codes;Hidden Markov models;Adaptation models;Artificial intelligence;Training;Risk management;Electronic mail;Transformers;Large language models;deep neural networks;uncertainty estimation;software reliability},
  doi={10.1109/TSE.2024.3519464}
  }
```

