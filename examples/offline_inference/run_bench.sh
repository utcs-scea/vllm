#!/bin/bash

models=('meta-llama/llama-2-13b-hf')
models=('mistralai/mistral-7b-v0.1' 'meta-llama/llama-2-7b-chat-hf' 'meta-llama/llama-2-13b-hf' 'google/gemma-2-9b')
models=('facebook/opt-125m')
models=('google/gemma-2-9b')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/llama-2-7b-chat-hf')
models=('deepseek-ai/deepseek-llm-7b-chat')
models=('openai-community/gpt2-xl')
models=('meta-llama/llama-2-13b-hf')
models=('meta-llama/llama-3.1-8b')
models=('mistralai/mistral-7b-v0.1')
models=('mistralai/mistral-7b-v0.1' 'meta-llama/llama-3.1-8b' 'meta-llama/llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')
models=('meta-llama/llama-2-13b-hf' 'openai-community/gpt2-xl' 'deepseek-ai/deepseek-llm-7b-chat')

gpu_mem=('0.95' '0.9' '0.85' '0.8' '0.75' '0.7' '0.65' '0.6' '0.55' '0.5' '0.45' '0.4' '0.35' '0.3' '0.25' '0.2')
gpu_mem=('0.9' '0.8' '0.7' '0.6' '0.5' '0.4' '0.3')
gpu_mem=('0.9')


model=('meta-llama/llama-3.1-8b')

#output_dir=$(pwd)/results_throughput
#output_dir=$(pwd)/results_peak_usage
export HF_HOME=/var/local/tkim/huggingface
#export hf_home=/work/10000/tlkim/hf_cache

python3 disaggregated_prefill.py \
    --backend vllm \
    --dataset $(pwd)/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model ${model} \
    --swap-space 4 \
    --preemption_mode swap \
    --num-prompts 1000 
echo "Done, sleeping 30 seconds for cooling down GPU"

