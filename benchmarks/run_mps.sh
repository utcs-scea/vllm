#!/bin/bash

models=('google/gemma-2-9b')
models=('meta-llama/Llama-3.1-8B')
models=('mistralai/Mistral-7B-v0.1' 'meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'google/gemma-2-9b')
models=('meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-13b-hf')
models=('meta-llama/Llama-2-7b-chat-hf')
models=('openai-community/gpt2-xl')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/Llama-2-13b-hf')
models=('mistralai/Mistral-7B-v0.1')
models=('meta-llama/Llama-3.1-8B')

sm_counts=('100')

device_num=0

export HF_HOME=/var/local/tkim/huggingface
output_dir=$(pwd)/results_corun

for model in "${models[@]}"
do 
    instance=0
    #for gpu_mem_util in "${gpu_mem[@]}"
    for count in "${sm_counts[@]}"
    do
        export CUDA_VISIBLE_DEVICES=$(($device_num%2))
        echo "Running MPS on Background"
        sudo nvidia-cuda-mps-control -d

        echo "Set active percentage as ${count}"
        echo set_default_active_thread_percentage ${count} | nvidia-cuda-mps-control

#        python benchmark_throughput.py \
#            --backend vllm \
#            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
#            --model ${model} \
#            --swap-space 16 \
#            --preemption_mode swap \
#            --gpu-memory-utilization 0.45 \
#            --num-prompts 200 \
#            --max-model-len 10240 \
#            --output-json $output_dir/${model}-${count}sm-45mem-corun1.json &

        python benchmark_throughput.py \
            --backend vllm \
            --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model ${model} \
            --swap-space 16 \
            --preemption_mode swap \
            --gpu-memory-utilization 0.9 \
            --num-prompts 400 \
            --max-model-len 10240 \
            --output-json $output_dir/${model}-${count}sm-90mem-400.json 

        ((device_num++))
        echo "Done running benchmarks... terminating MPS..."
        sleep 5
        sudo sh -c "echo quit | nvidia-cuda-mps-control"
        sleep 5
    done
done

# Mistral
            #--max-model-len 22464 \

# Llama 3.1
            #--max-model-len 28240 \
            #--max-model-len 14064 \
