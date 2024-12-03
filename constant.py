import os

gpt2_prefix = "/data/huangyuheng/QA/gpt2/"
GPT2_RAW_DATA = {
    "cnn_daily": os.path.join(gpt2_prefix, "cnn_daily/greedy/raw_inference"),
    "eli5_category": os.path.join(gpt2_prefix, "eli5_category/greedy/raw_inference"),
    "wiki": os.path.join(gpt2_prefix, "wiki/greedy/raw_inference"),
    "wmt14": os.path.join(gpt2_prefix, "wmt14_test/greedy/raw_inference")}

llama_prefix = "/data/huangyuheng/QA/llama/"
LLAMA_RAW_DATA = {
    "cnn_daily": os.path.join(llama_prefix, "cnn_daily/greedy/raw_inference"),
    "eli5_category": os.path.join(llama_prefix, "eli5_category/greedy/raw_inference"),
    "wiki": os.path.join(llama_prefix, "wiki/greedy/raw_inference"),
    "wmt14": os.path.join(llama_prefix, "wmt14_test/greedy/raw_inference")
}


llama2_prefix = "/data/huangyuheng/QA/llama2/"
LLAMA2_RAW_DATA = {
    "cnn_daily": os.path.join(llama2_prefix, "cnn_daily/greedy/raw_inference"),
    "eli5_category": os.path.join(llama2_prefix, "eli5_category/greedy/raw_inference"),
    "wiki": os.path.join(llama2_prefix, "wiki/greedy/raw_inference"),
    "wmt14": os.path.join(llama2_prefix, "wmt14_test/greedy/raw_inference")
}
LLAMA2_NO_TEMPLETE = {
    "cnn_daily": os.path.join(llama2_prefix, "cnn_daily/greedy/notemplate"),
    "eli5_category": os.path.join(llama2_prefix, "eli5_category/greedy/notemplate"),
    "wiki": os.path.join(llama2_prefix, "wiki/greedy/notemplate"),
    "wmt14": os.path.join(llama2_prefix, "wmt14_test/greedy/notemplate")
}

llama3_prefix = "/data/huangyuheng/QA/llama3/"

LLAMA3_RAW_DATA = {
    "cnn_daily": os.path.join(llama3_prefix, "cnn_daily/greedy/raw_inference"),
    "eli5_category": os.path.join(llama3_prefix, "eli5_category/greedy/raw_inference"),
    "wiki": os.path.join(llama3_prefix, "wiki/greedy/raw_inference"),
    "wmt14": os.path.join(llama3_prefix, "wmt14_test/greedy/raw_inference")
}

phi3_prefix = "/data/huangyuheng/QA/phi3/"

PHI3_RAW_DATA = {
    "cnn_daily": os.path.join(phi3_prefix, "cnn_daily/greedy/raw_inference"),
    "eli5_category": os.path.join(phi3_prefix, "eli5_category/greedy/raw_inference"),
    "wiki": os.path.join(phi3_prefix, "wiki/greedy/raw_inference"),
    "wmt14": os.path.join(phi3_prefix, "wmt14_test/greedy/raw_inference")
}

gemma2_prefix = "/data/huangyuheng/QA/gemma2/"

GEMMA2_RAW_DATA = {
    "cnn_daily": os.path.join(gemma2_prefix, "cnn_daily/greedy/raw_inference"),
    "eli5_category": os.path.join(gemma2_prefix, "eli5_category/greedy/raw_inference"),
    "wiki": os.path.join(gemma2_prefix, "wiki/greedy/raw_inference"),
    "wmt14": os.path.join(gemma2_prefix, "wmt14_test/greedy/raw_inference")
}

gpt2_uncertain_prefix = "/data/huangyuheng/QA/uncertainty/gpt2"
llama_uncertain_prefix = "/data/huangyuheng/QA/uncertainty/llama"
llama2_uncertain_prefix = "/data/huangyuheng/QA/uncertainty/llama2"
llama3_uncertain_prefix = "/data/huangyuheng/QA/uncertainty/llama3"
phi3_uncertain_prefix = "/data/huangyuheng/QA/uncertainty/phi3"
gemma2_uncertain_prefix = "/data/huangyuheng/QA/uncertainty/gemma2"

# Raw_data_paths: {original_raw_data_path: [[perturbation_path, result_save_path], [sample_based_uncertainty_path, sample_based_save_path]]}

RAW_DATA_PATHS = {LLAMA_RAW_DATA["eli5_category"]: [[os.path.join(llama_uncertain_prefix, "eli5_category_alter")],
                                                   [os.path.join(llama_uncertain_prefix, "sample_uncertain_eli5_category")]], 
                  LLAMA_RAW_DATA["wiki"]: [[os.path.join(llama_uncertain_prefix, "wiki_alter")],
                                          [os.path.join(llama_uncertain_prefix, "sample_uncertain_wiki")]], 
                  LLAMA_RAW_DATA["wmt14"]: [[os.path.join(llama_uncertain_prefix, "wmt14_alter")],
                                          [os.path.join(llama_uncertain_prefix, "sample_uncertain_wmt14")]],
                  LLAMA_RAW_DATA["cnn_daily"]: [
                      [os.path.join(llama_uncertain_prefix, "llama_cnn_daily_alter")],
                      [os.path.join(llama_uncertain_prefix, "sample_uncertain_cnn_dailymail")]
                  ],
                  GPT2_RAW_DATA['eli5_category']: [
                      [os.path.join(gpt2_uncertain_prefix, "eli5_alter")],
                      [os.path.join(gpt2_uncertain_prefix, "sample_uncertain_eli5_category")]
                  ], 
                  GPT2_RAW_DATA["wmt14"]: [[os.path.join(gpt2_uncertain_prefix, "wmt14_alter")],
                                          [os.path.join(gpt2_uncertain_prefix, "sample_uncertain_wmt14")]], 
                  GPT2_RAW_DATA["wiki"]: [[os.path.join(gpt2_uncertain_prefix, "wiki_alter")],
                                          [os.path.join(gpt2_uncertain_prefix, "sample_uncertain_wiki")]],
                  GPT2_RAW_DATA["cnn_daily"]: [
                      [os.path.join(gpt2_uncertain_prefix, "gpt2_cnn_daily_alter")],
                      [os.path.join(gpt2_uncertain_prefix, "sample_uncertain_cnn_dailymail")]
                  ],
                  LLAMA2_RAW_DATA["eli5_category"]: [[os.path.join(llama2_uncertain_prefix, "eli5_category_alter")],
                                                   [os.path.join(llama2_uncertain_prefix, "sample_uncertain_eli5_category")]], 
                  LLAMA2_RAW_DATA["wiki"]: [[os.path.join(llama2_uncertain_prefix, "wiki_alter")],
                                          [os.path.join(llama2_uncertain_prefix, "sample_uncertain_wiki")]], 
                  LLAMA2_RAW_DATA["wmt14"]: [[os.path.join(llama2_uncertain_prefix, "wmt14_alter")],
                                          [os.path.join(llama2_uncertain_prefix, "sample_uncertain_wmt14")]],
                  LLAMA2_RAW_DATA["cnn_daily"]: [
                      [os.path.join(llama2_uncertain_prefix, "cnn_daily_alter")],
                      [os.path.join(llama2_uncertain_prefix, "sample_uncertain_cnn_dailymail")]
                  ],
                  LLAMA2_NO_TEMPLETE["eli5_category"]: [[os.path.join(llama2_uncertain_prefix, "notemplate_eli5_category_alter")],
                                                   [os.path.join(llama2_uncertain_prefix, "notemplate_sample_uncertain_eli5_category")]], 
                  LLAMA2_NO_TEMPLETE["wiki"]: [[os.path.join(llama2_uncertain_prefix, "notemplate_wiki_alter")],
                                          [os.path.join(llama2_uncertain_prefix, "notemplate_sample_uncertain_wiki")]], 
                  LLAMA2_NO_TEMPLETE["wmt14"]: [[os.path.join(llama2_uncertain_prefix, "notemplate_wmt14_alter")],
                                          [os.path.join(llama2_uncertain_prefix, "notemplate_sample_uncertain_wmt14")]],
                  LLAMA2_NO_TEMPLETE["cnn_daily"]: [
                      [os.path.join(llama2_uncertain_prefix, "notemplate_cnn_daily_alter")],
                      [os.path.join(llama2_uncertain_prefix, "notemplate_sample_uncertain_cnn_dailymail")]
                  ],
                  LLAMA3_RAW_DATA["eli5_category"]: [[os.path.join(llama3_uncertain_prefix, "eli5_category_alter")],
                                                   [os.path.join(llama3_uncertain_prefix, "sample_uncertain_eli5_category")]], 
                  LLAMA3_RAW_DATA["wiki"]: [[os.path.join(llama3_uncertain_prefix, "wiki_alter")],
                                          [os.path.join(llama3_uncertain_prefix, "sample_uncertain_wiki")]], 
                  LLAMA3_RAW_DATA["wmt14"]: [[os.path.join(llama3_uncertain_prefix, "wmt14_alter")],
                                          [os.path.join(llama3_uncertain_prefix, "sample_uncertain_wmt14")]],
                  LLAMA3_RAW_DATA["cnn_daily"]: [
                      [os.path.join(llama3_uncertain_prefix, "cnn_daily_alter")],
                      [os.path.join(llama3_uncertain_prefix, "sample_uncertain_cnn_dailymail")]
                  ],
                  PHI3_RAW_DATA["eli5_category"]: [[os.path.join(phi3_uncertain_prefix, "eli5_category_alter")],
                                                   [os.path.join(phi3_uncertain_prefix, "sample_uncertain_eli5_category")]], 
                  PHI3_RAW_DATA["wiki"]: [[os.path.join(phi3_uncertain_prefix, "wiki_alter")],
                                          [os.path.join(phi3_uncertain_prefix, "sample_uncertain_wiki")]], 
                  PHI3_RAW_DATA["wmt14"]: [[os.path.join(phi3_uncertain_prefix, "wmt14_alter")],
                                          [os.path.join(phi3_uncertain_prefix, "sample_uncertain_wmt14")]],
                  PHI3_RAW_DATA["cnn_daily"]: [
                      [os.path.join(phi3_uncertain_prefix, "cnn_daily_alter")],
                      [os.path.join(phi3_uncertain_prefix, "sample_uncertain_cnn_dailymail")]
                  ],
                  GEMMA2_RAW_DATA["eli5_category"]: [[os.path.join(gemma2_uncertain_prefix, "eli5_category_alter")],
                                                   [os.path.join(gemma2_uncertain_prefix, "sample_uncertain_eli5_category")]], 
                  GEMMA2_RAW_DATA["wiki"]: [[os.path.join(gemma2_uncertain_prefix, "wiki_alter")],
                                          [os.path.join(gemma2_uncertain_prefix, "sample_uncertain_wiki")]], 
                  GEMMA2_RAW_DATA["wmt14"]: [[os.path.join(gemma2_uncertain_prefix, "wmt14_alter")],
                                          [os.path.join(gemma2_uncertain_prefix, "sample_uncertain_wmt14")]],
                  GEMMA2_RAW_DATA["cnn_daily"]: [
                      [os.path.join(gemma2_uncertain_prefix, "cnn_daily_alter")],
                      [os.path.join(gemma2_uncertain_prefix, "sample_uncertain_cnn_dailymail")]
                  ],
                }

# ======== code path ========
codegen_prefix = "/data/huangyuheng/code/codegen"
CODEGEN_RAW_DATA = {
    "humaneval": os.path.join(codegen_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(codegen_prefix, "mbpp/greedy/raw_inference"),
}

incoder_prefix = "/data/huangyuheng/code/incoder"
INCODER_RAW_DATA = {
    "humaneval": os.path.join(incoder_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(incoder_prefix, "mbpp/greedy/raw_inference")
}

santacoder_prefix = "/data/huangyuheng/code/santacoder/"
SANTACODER_RAW_DATA = {
    "humaneval": os.path.join(santacoder_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(santacoder_prefix, "mbpp/greedy/raw_inference")
}

codellama_prefix = "/data/huangyuheng/code/codellama"
CODELLAMA_RAW_DATA = {
    "humaneval": os.path.join(codellama_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(codellama_prefix, "mbpp/greedy/raw_inference")
}

#llama3_code_prefix = "/data/huangyuheng/code/llama3/"
#LLAMA3_CODE_RAW_DATA = {
#    "humaneval": os.path.join(llama3_code_prefix, "humaneval/greedy/raw_inference"),
#    "mbpp": os.path.join(llama3_code_prefix, "mbpp/greedy/raw_inference")
#}

deepseekcoder_prefix = "/data/huangyuheng/code/deepseekcoder/"
DEEPSEEKCODER_RAW_DATA = {
    "humaneval": os.path.join(deepseekcoder_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(deepseekcoder_prefix, "mbpp/greedy/raw_inference")
}

codeqwen_prefix = "/data/huangyuheng/code/codeqwen1.5/"
CODEQWEN_RAW_DATA = {
    "humaneval": os.path.join(codeqwen_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(codeqwen_prefix, "mbpp/greedy/raw_inference")
}

starcoder2_prefix = "/data/huangyuheng/code/starcoder2/"

STARCODER2_RAW_DATA = {
    "humaneval": os.path.join(starcoder2_prefix, "humaneval/greedy/raw_inference"),
    "mbpp": os.path.join(starcoder2_prefix, "mbpp/greedy/raw_inference")
}



perturbation_prefix = "/data/huangyuheng/code/uncertainty/result/perturbation/"
sample_prefix = "/data/huangyuheng/code/uncertainty/result/sample_uncertain/"

CODE_RAW_DATA_PATHS = {
    CODEGEN_RAW_DATA["humaneval"]: [
        [
            os.path.join(perturbation_prefix, "codegen_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_codegen_humaneval"),
        ]
    ],
    CODEGEN_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "codegen_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_codegen_mbpp"),
        ]
    ],
    INCODER_RAW_DATA["humaneval"]: [
        [
            os.path.join(perturbation_prefix, "incoder_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_incoder_humaneval"),
        ]
    ],
    INCODER_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "incoder_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_incoder_mbpp"),
        ]
    ],
    SANTACODER_RAW_DATA["humaneval"]: [
        [
            os.path.join(perturbation_prefix, "santacoder_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_santacoder_humaneval"),
        ]
    ],
    SANTACODER_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "santacoder_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_santacoder_mbpp"),
        ]
    ],
    CODELLAMA_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "codellama_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_codellama_mbpp"),
        ]
    ],
    CODELLAMA_RAW_DATA["humaneval"]: [
        [
             os.path.join(perturbation_prefix, "codellama_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_codellama_humaneval"),
        ]
    ],
    DEEPSEEKCODER_RAW_DATA["humaneval"]: [
        [
            os.path.join(perturbation_prefix, "deepseekcoder_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_deepseekcoder_humaneval"),
        ]
    ],
    DEEPSEEKCODER_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "deepseekcoder_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_deepseekcoder_mbpp"),
        ]
    ],
    CODEQWEN_RAW_DATA["humaneval"]: [
        [
            os.path.join(perturbation_prefix, "codeqwen1.5_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_codeqwen1.5_humaneval"),
        ]
    ],
    CODEQWEN_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "codeqwen1.5_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_codeqwen1.5_mbpp"),
        ]
    ],
    STARCODER2_RAW_DATA["humaneval"]: [
        [
            os.path.join(perturbation_prefix, "starcoder2_humaneval_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_starcoder2_humaneval"),
        ]
    ],
    STARCODER2_RAW_DATA["mbpp"]: [
        [
            os.path.join(perturbation_prefix, "starcoder2_mbpp_alter"),
        ],
        [
            os.path.join(sample_prefix, "sample_uncertain_starcoder2_mbpp"),
        ]
    ],
    
}


# =================


openai_prefix = "data/experiment/result/"

NLP_DAVINCI_RAW_DATA = {
 "wmt14": os.path.join(openai_prefix, 'wmt14_test_new_98_text-davinci-003.pkl'),
 "cnn_daily": os.path.join(openai_prefix, 'cnn_dailymail_3_test_100_text-davinci-003.pkl'),
 "eli5_category": os.path.join(openai_prefix, 'eli5_category_test_100_text-davinci-003.pkl'),
 "wiki": os.path.join(openai_prefix, 'wiki_qa_test_new_100_text-davinci-003.pkl'),
}

NLP_CURIE_RAW_DATA = {
 "wmt14":  os.path.join(openai_prefix, 'wmt14_test_new_100_text-curie-001.pkl'),
 "eli5_category":  os.path.join(openai_prefix, 'eli5_category_test_100_text-curie-001.pkl'),
 "wiki":  os.path.join(openai_prefix, 'wiki_qa_test_new_100_text-curie-001.pkl'),
 "cnn_daily":  os.path.join(openai_prefix, 'cnn_dailymail_3_test_97_text-curie-001.pkl')
}


CODE_DAVINCI_RAW_DATA = {
    "humaneval": os.path.join(openai_prefix, "humaneval_new_40_text-davinci-003.pkl"),
    "mbpp": os.path.join(openai_prefix, "mbpp_new_125_text-davinci-003.pkl")
}
