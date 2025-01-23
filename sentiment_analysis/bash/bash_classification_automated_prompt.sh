#!/bin/bash

LLM_IS_DIR="/home/$USER/IC_MGI/MGI_Teste/sentiment_analysis/"
cd "$LLM_IS_DIR"

#PROMPT_FILES=("prompt1" "prompt2" "prompt3" "prompt4")
#EXAMPLE_COUNTS=(1 2 3 4 5 6 7)
PROMPT_FILES=("prompt4")
EXAMPLE_COUNTS=(1)

for NUMBER_OF_EXAMPLES in "${EXAMPLE_COUNTS[@]}"; do
    for PROMPT_FILE in "${PROMPT_FILES[@]}"; do
        for model in Llama3.1-I; do
            python3.10 run_llms_classifiication.py --number_of_examples $NUMBER_OF_EXAMPLES \
                                                --inputdir '../data/SUS_Simulador_Aposentadoria_pre_processado.csv' \
                                                --llm_method $model \
                                                --outputdir "resources/outLLM/sentiment_analysis/${PROMPT_FILE}/${NUMBER_OF_EXAMPLES}_few_shot" \
                                                --prompt_dir "resources/prompt/${PROMPT_FILE}.json"
        done
    done
done