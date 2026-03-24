"""DeepSeek-V3 standalone — load checkpoint and generate text."""

import time
import torch
from config import DEEPSEEK_V3_CONFIG, load_config_from_hf
from model import DeepSeekV3ForCausalLM
from load_weights import load_weights
from tokenizer import DeepSeekTokenizer
from generate import generate_stream


CHECKPOINT_DIR = "/path/to/deepseek-v3-checkpoint"
TOKENIZER_PATH = f"{CHECKPOINT_DIR}/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def main():
    # 1. Load config
    print("Loading config...")
    cfg = load_config_from_hf(CHECKPOINT_DIR)

    # 2. Build model
    print("Building model...")
    model = DeepSeekV3ForCausalLM(cfg)

    # 3. Load weights
    print("Loading weights...")
    model = load_weights(model, CHECKPOINT_DIR, device=DEVICE, dtype=DTYPE)
    model.eval()

    # 4. Load tokenizer
    tokenizer = DeepSeekTokenizer(TOKENIZER_PATH)

    # 5. Encode prompt
    prompt = "Hello, how are you?"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {len(input_ids)}")
    print("-" * 40)

    # 6. Generate (streaming)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token in generate_stream(
        model=model,
        input_ids=input_tensor,
        max_new_tokens=256,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
    ):
        generated_tokens += 1
        print(tokenizer.decode(token.squeeze(0).tolist()), end="", flush=True)

    # 7. Report
    elapsed = time.perf_counter() - start_time
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n\nGenerated {generated_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.2f} tokens/sec)")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"GPU memory used: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
