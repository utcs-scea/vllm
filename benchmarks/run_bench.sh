#!/bin/bash

models=('meta-llama/Llama-2-7b-chat-hf')
models=('meta-llama/Llama-2-13b-hf')
models=('google/gemma-2-9b')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'google/gemma-2-9b')
models=('meta-llama/Llama-3.1-8B')

gpu_mem=('0.9' '0.85' '0.8' '0.75' '0.7')
gpu_mem=('0.9')

device_num=0

output_dir=$(pwd)/results_throughput

for model in "${models[@]}"
do 
    for gpu_mem_util in "${gpu_mem[@]}"
    do
        export CUDA_VISIBLE_DEVICES=$(($device_num%2))
        python benchmark_throughput.py \
            --backend vllm \
            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --swap-space 0 \
            --gpu-memory-utilization ${gpu_mem_util} \
            --num-prompts 1000 \
            --output-json $output_dir/${gpu_mem_util}.json 
        ((device_num++))
        echo "Done, sleeping 10 seconds for cooling down GPU"
        sleep 10
    done
done
