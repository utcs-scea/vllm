#!/bin/bash

models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-3.1-8B')

device_num=0
output_dir=$(pwd)/results_no_mem
#export HF_HOME=/var/local/tkim/huggingface

gpu_mem=('0.9')

for model in "${models[@]}"
do 
    for gpu_mem_util in "${gpu_mem[@]}"
    do
        #export CUDA_VISIBLE_DEVICES=$(($device_num%2))
        python3 benchmark_prefix_caching.py \
            --dataset-path $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --swap-space 64 \
            --preemption_mode swap \
            --gpu-memory-utilization ${gpu_mem_util} \
            --num-prompts 20 \
            --input-length-range 128:256 \
            --repeat-count 5 \
            --enable-prefix-caching 
            #--max-model-len 93000 \
            #--num-gpu-blocks-override 1 \
#        echo "Done, sleeping 30 seconds for cooling down GPU"
#        sleep 30 
    done
done

echo "Done"
