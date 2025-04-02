#!/bin/bash

models=('google/gemma-2-9b')
models=('meta-llama/Llama-3.1-8B')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'google/gemma-2-9b')
models=('meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('openai-community/gpt2-xl')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('mistralai/Mistral-7B-v0.1')
models=('meta-llama/Llama-3.1-8B')
models=('meta-llama/Llama-2-13b-hf')

gpu_mem=('0.9' '0.85' '0.8' '0.75' '0.7' '0.65' '0.6' '0.5' '0.55')
gpu_mem=('0.7' '0.65' '0.6' '0.55' '0.5' '0.45' '0.4' '0.35' '0.3')
gpu_mem=('0.9' '0.85' '0.8' '0.75' '0.7')
gpu_mem=('0.95' '0.9' '0.85')
gpu_mem=('0.95' '0.9' '0.85')
gpu_mem=('0.9' '0.85' '0.8' '0.75' '0.7' '0.65' '0.6' '0.55' '0.5')
gpu_mem=('0.95' '0.9' '0.85' '0.8' '0.75' '0.7' '0.65' '0.6' '0.55' '0.5' '0.45' '0.4')
gpu_mem=('0.9' '0.8' '0.7' '0.6' '0.5')
gpu_mem=('0.9')

device_num=0

#output_dir=$(pwd)/results_throughput
#output_dir=$(pwd)/results_peak_usage
output_dir=$(pwd)/results_test

for model in "${models[@]}"
do 
    for gpu_mem_util in "${gpu_mem[@]}"
    do
        export CUDA_VISIBLE_DEVICES=$(($device_num%2))
        python benchmark_throughput.py \
            --backend vllm \
            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --gpu-memory-utilization ${gpu_mem_util} \
            --num-prompts 200 \
            --preemption-mode swap \
            --output-json $output_dir/${model}-${gpu_mem_util}-test.json |& tee $output_dir/${model}-${gpu_mem_util}.log
        ((device_num++))
        echo -n "${gpu_mem_util} " >> $output_dir/${model}-peak-cache-usage.log
        grep "Throughput:" $output_dir/${model}-${gpu_mem_util}.log | cut -f2 -d ' '| tr '\n' ' ' >> $output_dir/${model}-peak-cache-usage.log
        grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${gpu_mem_util}.log | sort -n | tail -n 1 >> $output_dir/${model}-peak-cache-usage.log

        echo "Done, sleeping 30 seconds for cooling down GPU"
        sleep 30 
    done
done

# Mistral
            #--max-model-len 22464 \

# Llama 3.1
            #--max-model-len 28240 \
            #--max-model-len 14064 \
