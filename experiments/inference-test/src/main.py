# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_diagnostics():
    rank = int(os.environ.get("RANK", 0))
    prefix = f"[Rank {rank}] "
    print(f"{prefix}Diagnostics:")
    print(f"{prefix}PyTorch version: {torch.__version__}")
    print(f"{prefix}CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{prefix}CUDA version: {torch.version.cuda}")
        print(f"{prefix}cuDNN version: {torch.backends.cudnn.version()}")
        print(f"{prefix}Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"{prefix}Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"{prefix}  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"{prefix}  Multi-processor count: {props.multi_processor_count}")
            print(f"{prefix}  Compute capability: {props.major}.{props.minor}")
    print(f"{prefix}" + "-" * 20)

def main():
    parser = argparse.ArgumentParser(description="Run simple inference benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID to use")
    args = parser.parse_args()

    # Initialize distributed environment if applicable
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Rank {rank}] Initialized process group. World size: {world_size}")
    else:
        rank = 0
        print("Not running in distributed mode.")

    print_diagnostics()

    model_id = args.model
    print(f"[Rank {rank}] Loading model: {model_id}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    prompt = "What is Raleigh Scattering?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print(f"Prompt: {prompt}")
    print("Starting generation...")
    start_time = time.time()
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    end_time = time.time()
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\nResponse:")
    print(response)
    
    # Calculate metrics
    num_tokens = sum(len(ids) for ids in generated_ids)
    duration = end_time - start_time
    tokens_per_second = num_tokens / duration if duration > 0 else 0
    
    print("\nMetrics:")
    print(f"Tokens generated: {num_tokens}")
    print(f"Duration: {duration:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
