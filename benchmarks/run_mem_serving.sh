#!/bin/bash

models=('meta-llama/Llama-2-13b-hf')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'google/gemma-2-9b')
models=('facebook/opt-125m')
models=('google/gemma-2-9b')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-3.1-8B')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('deepseek-ai/deepseek-llm-7b-chat' 'meta-llama/Llama-2-13b-hf')
models=('openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat' 'meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-3.1-8B')
models=('deepseek-ai/deepseek-llm-7b-chat' 'meta-llama/Llama-2-13b-hf')
models=('openai-community/gpt2-xl')
models=('mistralai/Mistral-7B-v0.1')

#gpm_mem=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100')
gpu_mem=('0.9' '0.8' '0.7' '0.6' '0.5')
request_rates=('1' '16' 'inf')
request_rates=('inf')

device_num=0

output_dir=$(pwd)/results_serving/mem
#export HF_HOME=/work/10000/tlkim/hf_cache
export HF_HOME=/var/local/tkim/huggingface


for model in "${models[@]}"
do 
    for request_rate in "${request_rates[@]}"
    do
            for mem in "${gpu_mem[@]}"
            do
            export CUDA_VISIBLE_DEVICES=$(($device_num%2))
            vllm serve ${model} \
                --swap-space 0 \
                --max-model-len 22464 \
                --disable-log-requests \
                --gpu-memory-utilization ${mem} |& tee $output_dir/${model}-${mem}-${request_rate}-server.log &
            echo "Waiting for launching server..."
            sleep 80

            python3 benchmark_serving.py \
                --disable-tqdm \
                --backend vllm \
                --dataset-name sharegpt \
                --dataset-path $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
                --model ${model} \
                --request-rate ${request_rate} \
                --num-prompts 1000 |& tee $output_dir/${model}-${mem}-${request_rate}-client.log
            echo -n "${mem} " >> $output_dir/${model}-${request_rate}.log
            sleep 5
            grep "Request throughput" $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Mean TTFT" $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "P99 TTFT"  $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Mean TPOT" $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "P99 TPOT"  $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Mean ITL" $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "P99 ITL"  $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Output token throughput" $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f5 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Total Token throughput" $output_dir/${model}-${mem}-${request_rate}-client.log | tr -s ' ' | cut -f5 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${mem}-${request_rate}-server.log | sort -n | tail -n 1 >> $output_dir/${model}-${request_rate}.log

            echo "Done, sleeping 10 seconds for cooling down GPU"
            sleep 10
            pkill vllm
            echo "Killing vllm server..."
            sleep 10
            ((device_num++))
        done
    done
done


           # --max-model-len 28240 \
