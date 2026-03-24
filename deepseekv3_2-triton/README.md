# DeepSeek-V3 Triton Kernel Implementation

## Overview

Pure Triton implementation of all DeepSeek-V3 compute kernels. This variant uses OpenAI Triton for all custom kernels, providing good portability across NVIDIA GPUs (Ampere, Hopper, Blackwell) while maintaining near-optimal performance.

## Architecture Reference

DeepSeek-V3 671B (arXiv: 2412.19437):
- **Backbone**: 61 decoder layers (3 dense + 58 MoE)
- **Attention**: Multi-head Latent Attention (MLA) with absorbed projections
  - 128 attention heads
  - q_lora_rank=1536, kv_lora_rank=512
  - qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
- **MoE**: 256 routed experts + 1 shared expert
  - Top-8 selection with grouped routing (n_group=8, topk_group=4)
  - Auxiliary-loss-free load balancing
  - moe_intermediate_size=2048
- **Positional**: YaRN RoPE (factor=40, original_max=4096, extends to 163K)
- **MTP**: 1 Multi-Token Prediction layer
- **Quantization**: FP8 E4M3 with 128x128 block scaling
- **No DSA**: Standard causal attention (no Dynamic Sparse Attention)

## Module Structure

```
deepseekv3_2-triton/
  __init__.py              # Package initialization with architecture docstring
  config.py                # DeepSeek-V3 configuration dataclass
  model.py                 # Full model: embeddings, decoder layers, LM head
  mla_attention.py         # MLA with Triton FlashAttention kernel
  mtp.py                   # Multi-Token Prediction layer
  rope_partial.py          # YaRN RoPE implementation
  cache.py                 # KV cache with paged allocation
  validate.py              # Validation utilities
  dsa_sparse_attention.py  # STUB: DSA not used in DeepSeek-V3
  dsa_indexer.py           # STUB: DSA not used in DeepSeek-V3
  unsloth_rms_layernorm.py # Triton fused RMSNorm kernel
  unsloth_swiglu.py        # Triton fused SwiGLU kernel
  unsloth_cross_entropy_loss.py  # Triton fused cross-entropy
  unsloth_fast_lora.py     # Triton LoRA kernels
  unsloth_utils.py         # Shared utilities
  PRECISION.md             # Precision tracking for this variant
  unsloth_moe/             # MoE kernel package
    __init__.py
    autotune_cache.py      # Autotune result caching
    grouped_gemm/          # Triton grouped GEMM implementation
      __init__.py
      interface.py         # User-facing API
      kernels/
        __init__.py
        forward.py         # Forward pass Triton kernel
        backward.py        # Backward pass Triton kernels
        autotuning.py      # Triton autotune configurations
        tuning.py          # Kernel config dataclasses
      reference/
        __init__.py
        moe_block.py       # PyTorch reference MoE block
        moe_ops.py         # Reference MoE operations
        layers/
          qwen3_moe.py     # Qwen3 MoE reference
          llama4_moe.py    # LLaMA4 MoE reference
```

## Key Components

### MLA Attention (mla_attention.py)

Triton kernel for Multi-head Latent Attention. The absorbed form operates on:
- Compressed query: `q_nope` (128-dim) + `q_pe` (64-dim) per head
- Compressed KV: `c_kv` (512-dim latent) + `k_pe` (64-dim RoPE)
- Output in latent space (512-dim), then up-projected to value space

The Triton attention kernel implements online softmax (FlashAttention-style) with:
- Block tiling over sequence dimension
- Causal masking (no DSA masking)
- Separate nope/rope score accumulation fused into single kernel

### MoE Grouped GEMM (unsloth_moe/)

Triton grouped GEMM for the 256-expert MoE layer:
- Forward: per-expert gate_up projection (7168 -> 4096) and down projection (2048 -> 7168)
- SwiGLU activation fused between gate and up projections
- Backward: separate dX and dW kernels with autotuning
- Supports both contiguous (prefill) and masked (decode) modes

### YaRN RoPE (rope_partial.py)

Frequency-dependent interpolation for extended context:
```
For dimension d with frequency freq_d:
  If freq_d is "high frequency" (> beta_fast): no scaling
  If freq_d is "low frequency" (< beta_slow): linear interpolation by factor
  Otherwise: smooth interpolation between the two
```

### Multi-Token Prediction (mtp.py)

Single MTP layer that predicts the next token beyond the standard LM head:
- Input: hidden states from the last decoder layer
- Processing: RMSNorm + linear projection + shared LM head
- Output: logits for position t+1 (used as auxiliary loss during training)

## Usage

```python
from importlib import import_module

# Import the Triton variant
triton_model = import_module("deepseekv3_2-triton.model")
triton_config = import_module("deepseekv3_2-triton.config")

# Create model with tiny config for testing
cfg = triton_config.make_tiny_config(num_layers=2)
model = triton_model.DeepSeekV3ForCausalLM(cfg).cuda().eval()

# Forward pass
input_ids = torch.randint(0, 256, (1, 32), device="cuda")
with torch.no_grad():
    loss, logits, cache = model(input_ids=input_ids, use_cache=True)

# Decode step
next_token = logits[:, -1:, :].argmax(dim=-1)
_, new_logits, cache = model(input_ids=next_token, past_key_values=cache, use_cache=True)
```

## Benchmarking

```bash
# Compare Triton variant against others
python3 benchmark_head_to_head.py --backends triton flashinfer flashmla-deepgemm --quick

# Full Triton benchmark suite
python3 -m benchmark.run_all --suite micro component --output-dir results/triton

# Validate Triton output against PyTorch reference
python3 -m deepseekv3_2-triton.validate
```

## Differences from Other Variants

| Feature | Triton | FlashInfer | FlashMLA+DeepGEMM |
|---------|--------|-----------|-------------------|
| Attention kernel | Triton FA | FlashInfer FA3 | FlashMLA |
| MoE GEMM | Triton grouped | Triton grouped | DeepGEMM FP8 |
| GPU support | Ampere+ | Hopper+ | Hopper+ |
| FP8 support | Software | Hardware | Hardware |
| Portability | High | Medium | Low |
| Performance (H100) | Good | Very Good | Best |

## Paper Reference

DeepSeek-V3 Technical Report: arXiv 2412.19437
