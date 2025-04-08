# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""
import argparse
import dataclasses
import os
import time
import json
import random
from typing import Dict, List, Optional, Tuple
from multiprocessing import Event, Process

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.config import KVTransferConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g.
            images).
        lora_request: Optional LoRARequest specifying the LoRA to use. 
    """
    prompt: str
    prompt_len: int
    expected_output_len: int


def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace) -> List[SampleRequest]:

    dataset_path: str = args.dataset
    num_requests: int = args.num_prompts
    fixed_output_len: Optional[int] = args.output_len
    model: str = args.model
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Shuffle the dataset.
    random.shuffle(dataset)
    print(len(dataset))

    # (taeklim): Count long sequence on dataset
    long_context_cnt = 0
    too_long_context_cnt = 0
    small_context_cnt = 0

    # Filter out sequences that are too long or too short
    filtered_dataset: List[SampleRequest] = []
    for data in tqdm(dataset,
                     total=len(filtered_dataset),
                     desc="sampling requests"):
        if len(filtered_dataset) == num_requests:
            break

        # Only keep the first two turns of each conversation.
        prompt = data["conversations"][0]["value"]
        completion = data["conversations"][1]["value"]

        request_tokenizer = tokenizer

        # Tokenize the prompts and completions.
        prompt_token_ids = request_tokenizer(prompt).input_ids
        completion_token_ids = request_tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len

        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            SampleRequest(prompt=prompt,
                          prompt_len=prompt_len,
                          expected_output_len=output_len))
    return filtered_dataset


def run_prefill(prefill_done, args: argparse.Namespace):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(tokenizer, args)

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
#    prompts = [
#        "Hello, my name is",
#        "Hi, your name is",
#        # The decode node will actually "prefill" this request.
#        "Tell me a very long story",
#    ]

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    )
    args.kv_transfer_config = ktc
    #args.model = "meta-llama/Meta-Llama-3.1-8B"
    args.max_model_len = 2000

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    # Add the requests to the engine.
    prompts: List[TextPrompt] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(
            TextPrompt(prompt=request.prompt))
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            ))

    print("Before prefill")
    start = time.perf_counter()
    llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    elapsed_time = end - start
    print("Prefill node is finished.")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, ")

    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done, args: argparse.Namespace):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(tokenizer, args)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    )

    args.kv_transfer_config = ktc
    #args.model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    args.max_model_len = 2000

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    # Add the requests to the engine.
    prompts: List[TextPrompt] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(
            TextPrompt(prompt=request.prompt))
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            ))

    # Wait for the producer to start the pipe
#    print("Waiting for prefill node to finish...")
#    prefill_done.wait()

    start = time.perf_counter()
    llm.generate(prompts, sampling_params)
    end = time.perf_counter()

    elapsed_time = end - start
    print(f"Decode node is finished. {len(requests)}")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, ")

#    # At this point when the prefill_done is set, the kv-cache should have been
#    # transferred to this decode node, so we can start decoding.
#    outputs = llm.generate(prompts, sampling_params)
#    for output in outputs:
#        prompt = output.prompt
#        generated_text = output.outputs[0].text
#        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset. The dataset is expected to "
                        "be a json in form of List[Dict[..., conversations: "
                        "List[Dict[..., value: <prompt_or_response>]]]]")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=200,
                        help="Number of prompts to process.")
#    parser.add_argument("--async-engine",
#                        action='store_true',
#                        default=False,
#                        help="Use vLLM async engine rather than LLM class.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, args, ))
    decode_process = Process(target=run_decode, args=(prefill_done, args, ))

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Terminate the prefill node when decode is finished
    decode_process.join()
    prefill_process.terminate()
