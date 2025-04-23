"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import random
import time
from functools import cache
from typing import Dict, List, Optional, Tuple

import torch
import uvloop
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.inputs import TextPrompt
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import BeamSearchParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer
from vllm.utils import FlexibleArgumentParser, merge_async_iterators

#import os
#os.environ['CURL_CA_BUNDLE'] = ''
#os.environ['REQUESTS_CA_BUNDLE'] = ''

#import requests
#from huggingface_hub import configure_http_backend
#
#def backend_factory() -> requests.Session:
#    session = requests.Session()
#    session.verify = False
#    return session

#configure_http_backend(backend_factory=backend_factory)

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
    multi_modal_data: Optional[MultiModalDataDict] = None
    lora_request: Optional[LoRARequest] = None


def _get_prompt_for_image_model(question: str, *, model: str) -> str:
    """Prepend and append special tokens around the question to form a prompt.

    Args:
        question: The input question text to wrap with special tokens
        model: The name of the model being used, to determine which special
            tokens to add

    Returns:
        The formatted prompt string with appropriate special tokens for the
            model

    Raises:
        ValueError: If an unsupported model name is provided
    """
    model = model.lower()
    if "pixtral" in model:
        return f"<s>[INST]{question}\n[IMG][/INST]"
    raise ValueError(f"Unsupported model {model}")


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


lora_tokenizer_cache: Dict[int, AnyTokenizer] = {}


def profile_dataset(tokenizer: PreTrainedTokenizerBase,
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

    # Count long sequence on dataset
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

        multi_modal_data: Optional[MultiModalDataDict] = None
        if "image" in data:
            multi_modal_data = multi_modal_data or {}
            image_path = data["image"]
            # TODO(vllm-project/vllm/issues/9778): Support multiple images.
            assert isinstance(image_path,
                              str), "Only support single image input"
            try:
                multi_modal_data["image"] = Image.open(image_path).convert(
                    "RGB")
            except FileNotFoundError:
                # Ignore datapoint where asset is missing
                continue
            prompt = _get_prompt_for_image_model(question=prompt, model=model)

        request_tokenizer = tokenizer
        lora_request: Optional[LoRARequest] = None

        # Tokenize the prompts and completions.
        prompt_token_ids = request_tokenizer(prompt).input_ids
        completion_token_ids = request_tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
#        if prompt_len > 12064:
#            too_long_context_cnt += 1
        if prompt_len > 1024:
            long_context_cnt += 1
        else:
            small_context_cnt += 1

        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            SampleRequest(prompt=prompt,
                          prompt_len=prompt_len,
                          expected_output_len=output_len,
                          multi_modal_data=multi_modal_data,
                          lora_request=lora_request))
    print(f"Long context prompt's count {long_context_cnt}")
    print(f"Too Long context prompt's count {too_long_context_cnt}")
    print(f"Small context prompt's count {small_context_cnt}")

    return filtered_dataset

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

    # Count long sequence on dataset
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

        #if prompt_len < 4 or output_len < 4:
        if prompt_len < 1024:
            # Prune too short sequences.
            continue
        #if prompt_len > 1024 or prompt_len + output_len > 2048:
        if prompt_len > 131072/4:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            SampleRequest(prompt=prompt,
                          prompt_len=prompt_len,
                          expected_output_len=output_len))
    print(f"filtered_dataset length: {len(filtered_dataset)}")
    return filtered_dataset


def run_vllm(
    requests: List[SampleRequest],
    n: int,
    engine_args: EngineArgs,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(**dataclasses.asdict(engine_args))

    # Add single request to the engine and iterate.
    prompts: List[TextPrompt] = []
    sampling_params: List[SamplingParams] = []
    for request in requests:
        prompts.append(
            TextPrompt(prompt=request.prompt))
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
            ))
        if len(prompts) >= 1:
            start = time.perf_counter()
            llm.generate(prompts,
                         sampling_params,
                         use_tqdm=True)
            end = time.perf_counter()
            print(f"e2e time: {end-start} input_len: {request.prompt_len}")
            prompts.clear()
            sampling_params.clear()

    return end - start


async def run_vllm_async(
    requests: List[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> float:
    from vllm import SamplingParams

    async with build_async_engine_client_from_engine_args(
            engine_args, disable_frontend_multiprocessing) as llm:

        # Add the requests to the engine.
        prompts: List[TextPrompt] = []
        sampling_params: List[SamplingParams] = []
        lora_requests: List[Optional[LoRARequest]] = []
        for request in requests:
            prompts.append(
                TextPrompt(prompt=request.prompt,
                           multi_modal_data=request.multi_modal_data))
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=request.expected_output_len,
                ))
            lora_requests.append(request.lora_request)

        generators = []
        start = time.perf_counter()
        for i, (prompt, sp,
                lr) in enumerate(zip(prompts, sampling_params, lora_requests)):
            generator = llm.generate(prompt,
                                     sp,
                                     lora_request=lr,
                                     request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(tokenizer, args)

    is_multi_modal = any(request.multi_modal_data is not None
                         for request in requests)
    if args.backend == "vllm":
        if args.async_engine:
            elapsed_time = uvloop.run(
                run_vllm_async(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    args.disable_frontend_multiprocessing,
                ))
        else:
            elapsed_time = run_vllm(requests, args.n,
                                    EngineArgs.from_cli_args(args))
    total_num_tokens = sum(request.prompt_len + request.expected_output_len
                           for request in requests)
    total_output_tokens = sum(request.expected_output_len
                              for request in requests)


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
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    # LoRA
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to the lora adapters to use. This can be an absolute path, "
        "a relative path, or a Hugging Face model identifier.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
    if args.enable_lora:
        assert args.lora_path is not None

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        # (taeklim): modifed for HF 
        if args.enable_lora is True:
            raise ValueError("LoRA benchmarking is only supported for vLLM"
                             " backend")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
        if args.enable_lora is not None:
            raise ValueError("LoRA benchmarking is only supported for vLLM"
                             " backend")
    main(args)
