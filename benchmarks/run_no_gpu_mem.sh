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


device_num=0

#output_dir=$(pwd)/results_throughput
#output_dir=$(pwd)/results_peak_usage
export HF_HOME=/var/local/tkim/huggingface
output_dir=$(pwd)/results_peak_usage
output_dir=$(pwd)/results_no_mem
#export HF_HOME=/work/10000/tlkim/hf_cache

gpu_mem=('0.7')

for model in "${models[@]}"
do 
    for gpu_mem_util in "${gpu_mem[@]}"
    do
        #export CUDA_VISIBLE_DEVICES=$(($device_num%2))
        export CUDA_VISIBLE_DEVICES=1
        python3 benchmark_throughput.py \
            --backend vllm \
            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --swap-space 64 \
            --preemption_mode swap \
            --gpu-memory-utilization ${gpu_mem_util} \
            --num-prompts 200 \
            --max-model-len 93000 \
            --output-json $output_dir/${model}-${gpu_mem_util}.json |& tee $output_dir/${model}-${gpu_mem_util}.log
            #--num-gpu-blocks-override 1 \
#        echo -n "${gpu_mem_util} " >> $output_dir/${model}-peak-cache-usage.log
#        grep "Throughput:" $output_dir/${model}-${gpu_mem_util}.log | cut -f2 -d ' ' | tr '\n' ' ' >> $output_dir/${model}-peak-cache-usage.log 
#        grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${gpu_mem_util}.log | sort -n | tail -n 1 >> $output_dir/${model}-peak-cache-usage.log
#        ((device_num++))
#        echo -n "${gpu_mem_util} " >> $output_dir/${model}-peak-cache-usage.log
#        grep "Throughput:" $output_dir/${model}-${gpu_mem_util}.log | cut -f2 -d ' '| tr '\n' ' ' >> $output_dir/${model}-peak-cache-usage.log
#        grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${gpu_mem_util}.log | sort -n | tail -n 1 >> $output_dir/${model}-peak-cache-usage.log

        echo "Done, sleeping 30 seconds for cooling down GPU"
        sleep 30 
    done
done

            #--max-model-len 22464 \
           # --max-model-len 28240 \
