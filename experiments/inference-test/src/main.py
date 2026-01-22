import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_diagnostics():
    print("Diagnostics:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")
            print(f"  Compute capability: {props.major}.{props.minor}")
    print("-" * 20)

def main():
    parser = argparse.ArgumentParser(description="Run simple inference benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID to use")
    args = parser.parse_args()

    print_diagnostics()

    model_id = args.model
    print(f"Loading model: {model_id}")

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

if __name__ == "__main__":
    main()
