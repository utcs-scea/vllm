#!/bin/bash

models=('meta-llama/Llama-2-13b-hf')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'google/gemma-2-9b')
models=('facebook/opt-125m')
models=('google/gemma-2-9b')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('openai-community/gpt2-xl')
models=('meta-llama/Llama-2-13b-hf')
models=('mistralai/Mistral-7B-v0.1')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-3.1-8B')

gpu_mem=('0.95' '0.9' '0.85' '0.8' '0.75' '0.7' '0.65' '0.6' '0.55' '0.5' '0.45' '0.4' '0.35' '0.3' '0.25' '0.2')
gpu_mem=('0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3')
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
        python3 benchmark_throughput.py \
            --backend vllm \
            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --swap-space 4 \
            --preemption_mode swap \
            --gpu-memory-utilization ${gpu_mem_util} \
            --num-prompts 1000 \
            --output-json $output_dir/${model}-${gpu_mem_util}.json |& tee $output_dir/${model}-${gpu_mem_util}.log
        echo -n "${gpu_mem_util} " >> $output_dir/${model}-peak-cache-usage.log
        grep "Throughput:" $output_dir/${model}-${gpu_mem_util}.log | cut -f2 -d ' ' | tr '\n' ' ' >> $output_dir/${model}-peak-cache-usage.log 
        grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${gpu_mem_util}.log | sort -n | tail -n 1 >> $output_dir/${model}-peak-cache-usage.log
        ((device_num++))
        echo -n "${gpu_mem_util} " >> $output_dir/${model}-peak-cache-usage.log
        grep "Throughput:" $output_dir/${model}-${gpu_mem_util}.log | cut -f2 -d ' '| tr '\n' ' ' >> $output_dir/${model}-peak-cache-usage.log
        grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${gpu_mem_util}.log | sort -n | tail -n 1 >> $output_dir/${model}-peak-cache-usage.log

        echo "Done, sleeping 30 seconds for cooling down GPU"
        sleep 30 
    done
done

            #--max-model-len 22464 \
           # --max-model-len 28240 \
