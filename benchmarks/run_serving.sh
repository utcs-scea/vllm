#!/bin/bash

models=('meta-llama/Llama-2-7b-chat-hf')

export CUDA_VISIBLE_DEVICES=0

python benchmark_serving.py \
    --backend vllm \
    --dataset-path /home/tkim/workspace/vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model meta-llama/Llama-2-7b-chat-hf \
    --num-prompts 1000 \
    --request-rate 1 \
    --result-filename $models-1qps-0.9mem \
    --result-dir /home/tkim/workspace/vllm/benchmarks/result_serving
#    --swap-space 0 \
#    --gpu-memory-utilization 0.9 \

