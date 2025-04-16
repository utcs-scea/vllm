#!/bin/bash

models=('deepseek-ai/deepseek-llm-7b-chat' 'openai-community/gpt2-xl' 'meta-llama/Llama-3.1-8B' 'meta-llama/Llama-2-13b-hf')
output_dir=$(pwd)/results_serving/mps

sm_counts=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100')
request_rate=('inf')

for model in "${models[@]}"
do
    for count in "${sm_counts[@]}" 
    do
        echo -n "${count} " >> $output_dir/${model}-${request_rate}.log
        grep "Output token throughput" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f5 -d ' ' | tr '\n' ' ' >>  $output_dir/${model}-${request_rate}.log
        grep "Total Token throughput" $output_dir/${model}-${count}-${request_rate}-client.log | tr -s ' ' | cut -f5 -d ' ' >>  $output_dir/${model}-${request_rate}.log
    done
done
