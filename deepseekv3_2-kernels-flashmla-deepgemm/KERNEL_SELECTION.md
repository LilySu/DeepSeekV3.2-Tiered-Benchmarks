# Kernel Selection: Why FlashMLA + DeepGEMM for DeepSeek-V3

## Executive Summary

**FlashMLA + DeepGEMM** is the **highest-performance** kernel combination
for DeepSeek-V3 inference and fine-tuning because they directly accelerate
the model's two most compute-intensive operations with hardware-native
Hopper GPU support.

## The Two Bottlenecks

DeepSeek-V3 (671B) spends the vast majority of its compute on:

1. **MLA Attention** (~35% of FLOPs)
   - 128 heads, 192-dim QK, 128-dim V
   - Compressed KV cache (512-dim latent + 64-dim RoPE)
   - Partial RoPE (only 64 of 192 QK dims)

2. **MoE Expert Dispatch** (~55% of FLOPs)
   - 256 routed experts + 1 shared expert
   - Top-8 selection per token
   - 3 matmuls per expert (gate, up, down)
   - 58 MoE layers (layers 3-60)

## Why FlashMLA for Attention

### The Problem
Standard attention kernels (FlashAttention-2, SDPA) do not understand MLA's
compressed KV cache format. They would require:
1. Expanding the 512-dim compressed KV to full 128*(128+128) = 32768 dims
2. Splitting and re-applying partial RoPE
3. Running attention on the expanded tensors

This negates MLA's memory savings and wastes bandwidth.

### The Solution
FlashMLA was **designed specifically by DeepSeek** for MLA attention:
- Reads compressed KV (512-dim) directly from cache
- Handles partial RoPE internally
- Fused paged KV cache access
- Optimised for Hopper's SM90 architecture

### Performance Impact
| Kernel | Prefill TFLOPS | Decode TFLOPS | Memory |
|--------|---------------|---------------|---------|
| Eager PyTorch | 15-20 | 1-3 | High (expanded KV) |
| FlashAttention-2 | 150-200 | 20-40 | High (expanded KV) |
| **FlashMLA** | **200-300** | **50-80** | **Low (compressed)** |

## Why DeepGEMM for MoE

### The Problem
DeepSeek-V3 has 256 experts x 3 matmuls = 768 individual GEMM operations
per MoE layer. With 58 MoE layers, that is 44,544 GEMMs per forward pass.

Standard approaches:
- **Sequential matmuls**: 768 kernel launches per layer (launch-bound)
- **Batched matmul (BMM)**: Requires padding all experts to same size
- **Triton grouped GEMM**: Good, but no FP8 support, no TMA

### The Solution
DeepGEMM provides **FP8 grouped GEMM** on Hopper:
- Single kernel launch for all experts in a layer
- FP8 (E4M3) computation doubles arithmetic throughput vs BF16
- Hopper TMA (Tensor Memory Accelerator) for async data movement
- Per-block (128x128) scaling for accuracy

### Performance Impact
| Approach | Kernel Launches | Throughput | Memory |
|----------|----------------|-----------|---------|
| Sequential BF16 | 768/layer | 50-100 TFLOPS | Baseline |
| BMM BF16 | 3/layer | 150-200 TFLOPS | Padding waste |
| Triton grouped BF16 | 3/layer | 200-250 TFLOPS | Efficient |
| **DeepGEMM FP8** | **3/layer** | **400-600 TFLOPS** | **50% less** |

## Why NOT Other Combinations

### FlashAttention-2 + cuBLAS
- FlashAttention-2 does not understand MLA compression
- cuBLAS has no grouped GEMM -- requires sequential calls or padding

### Triton + Triton
- Triton attention kernels lack MLA-native support
- Triton grouped GEMM lacks FP8 and TMA support
- Achievable but significantly slower than FlashMLA + DeepGEMM

### vLLM PagedAttention + CUTLASS
- PagedAttention does not handle partial RoPE natively
- CUTLASS grouped GEMM lacks the FP8 E4M3 + TMA combination
- Closer to optimal but still ~30% slower on Hopper

## Architecture Compatibility

### What DeepSeek-V3 Does NOT Have
- **No DSA**: Dynamic Sparse Attention is GLM5-specific, not in DeepSeek-V3
- **No auxiliary routing loss**: Uses `noaux_tc` (no aux loss)
- **No standard MHA/GQA**: Uses MLA exclusively

### DeepSeek-V3 Specific Features
- **MLA**: Compressed KV cache with partial RoPE
- **YaRN RoPE**: Factor=40 context extension
- **Group-restricted routing**: n_group=8, topk_group=4
- **Sigmoid routing**: Instead of softmax
- **MTP**: 1 multi-token prediction layer

## Hardware Requirements

| Feature | Minimum | Recommended |
|---------|---------|-------------|
| GPU | A100 (degraded) | H100/H200 |
| SM Version | SM80 (no FP8/TMA) | SM90+ |
| Memory | 80GB (TP=8) | 80GB x 8 (full model) |
| FlashMLA | Required for optimal attention | -- |
| DeepGEMM | Required for FP8 MoE | -- |

### Fallback Behavior
- **No FlashMLA**: Falls back to eager attention (5-10x slower)
- **No DeepGEMM**: Falls back to sequential PyTorch matmuls (4-8x slower)
- **No Hopper**: FP8 simulated, no TMA, significant performance loss

## References

1. DeepSeek-V3 Technical Report (arXiv 2412.19437)
2. FlashMLA: https://github.com/deepseek-ai/FlashMLA
3. DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
4. Hopper Architecture Whitepaper (NVIDIA, 2022)
