#!/bin/bash

models=('meta-llama/Llama-2-13b-hf')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'google/gemma-2-9b')
models=('facebook/opt-125m')
models=('google/gemma-2-9b')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-3.1-8B')
models=('mistralai/Mistral-7B-v0.1')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('deepseek-ai/deepseek-llm-7b-chat' 'meta-llama/Llama-2-13b-hf')
models=('openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat' 'meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-2-13b-hf')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('openai-community/gpt2-xl')
models=('meta-llama/Llama-3.1-8B')

sm_counts=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100')
request_rates=('1' '16' 'inf')
request_rates=('inf')

device_num=0

output_dir=$(pwd)/results_serving/mps
#export HF_HOME=/work/10000/tlkim/hf_cache
export HF_HOME=/var/local/tkim/huggingface


for model in "${models[@]}"
do 
    for request_rate in "${request_rates[@]}"
    do
            for count in "${sm_counts[@]}"
            do

            export CUDA_VISIBLE_DEVICES=$(($device_num%2))
            echo "Running MPS on Background"
            sudo nvidia-cuda-mps-control -d
            echo "Set active percentage as ${count}"
            echo set_default_active_thread_percentage ${count} | nvidia-cuda-mps-control

            vllm serve ${model} \
                --swap-space 0 \
                --max-model-len 14064 \
                --disable-log-requests \
                --gpu-memory-utilization 0.9 |& tee $output_dir/${model}-${count}-${request_rate}-server.log &
            echo "Waiting for launching server..."
            sleep 100

            python3 benchmark_serving.py \
                --disable-tqdm \
                --backend vllm \
                --dataset-name sharegpt \
                --dataset-path $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
                --model ${model} \
                --request-rate ${request_rate} \
                --num-prompts 1000 |& tee $output_dir/${model}-${count}-${request_rate}-client.log
            echo -n "${count} " >> $output_dir/${model}-${request_rate}.log
            sleep 5
            grep "Request throughput" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Mean TTFT" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "P99 TTFT"  $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Mean TPOT" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "P99 TPOT"  $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Mean ITL" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "P99 ITL"  $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f4 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Output token throughput" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f5 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep "Total Token throughput" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f5 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
            grep -oP " GPU KV cache usage:\s+\K\w+" $output_dir/${model}-${count}-${request_rate}-server.log | sort -n | tail -n 1 >> $output_dir/${model}-${request_rate}.log

            echo "Done, sleeping 10 seconds for cooling down GPU"
            sleep 10
            pkill vllm
            echo "Killing vllm server..."
            sleep 10

            echo "Done running benchmarks... terminating MPS..."
            sleep 5
            sudo sh -c "echo quit | nvidia-cuda-mps-control"
            sleep 10

#            ((device_num++))
        done
    done
done


            #--max-model-len 22464 \
           # --max-model-len 28240 \
