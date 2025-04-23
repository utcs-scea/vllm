#!/bin/bash

models=('facebook/opt-125m')
models=('google/gemma-2-9b')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('openai-community/gpt2-xl')
models=('mistralai/Mistral-7B-v0.1')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('meta-llama/Llama-3.1-8B')

gpu_mem=('0.9')

device_num=0

#export HF_HOME=/work/10000/tlkim/hf_cache
#export HF_HOME=/var/local/tkim/huggingface
output_dir=$(pwd)/results_test

for model in "${models[@]}"
do 
    for gpu_mem_util in "${gpu_mem[@]}"
    do
        #export CUDA_VISIBLE_DEVICES=$(($device_num%2))
        python3 benchmark_kv_access.py \
            --backend vllm \
            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --swap-space 4 \
            --preemption_mode swap \
            --gpu-memory-utilization ${gpu_mem_util} \
            --enable-prefix-caching \
            --num-prompts 4
#            --output-json $output_dir/${model}-${gpu_mem_util}.json |& tee $output_dir/${model}-${gpu_mem_util}.log
        echo "Done, sleeping 30 seconds for cooling down GPU"
    done
done
