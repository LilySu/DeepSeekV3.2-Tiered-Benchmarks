# Kernel Implementations by Architecture for DeepSeek-V3

## Overview

This document provides architecture-specific kernel implementation guidance for the four most critical DeepSeek-V3 operations: MLA attention, MoE dispatch, SwiGLU FFN, and RMSNorm. Each section covers NVIDIA CUDA, Huawei Ascend (CANN), and general optimization strategies.

## 1. MLA Attention Kernel

### 1.1 NVIDIA CUDA (H100)

The H100 implementation leverages FlashAttention-3 with modifications for MLA's compressed KV:

**Approach:** Modify FlashAttention-3's kernel to accept a compressed KV cache (d_c=512) and perform on-the-fly up-projection within the attention tile computation. The key insight is that the up-projection weights (W_uk, W_uv) are constant across positions, so they can be loaded into shared memory once per tile and reused.

**Tile strategy:**
- Q tiles: 128 tokens x 192 dims (nope + rope) per head
- KV tiles: loaded as compressed latent (128 tokens x 512 dims), then up-projected in shared memory to K (128 x 192) and V (128 x 128) per head
- Shared memory usage: 128 x 512 x 2 (KV latent) + 512 x 192 (W_uk tile) + 512 x 128 (W_uv tile) = ~320 KB

This exceeds H100's 228 KB shared memory per SM, so the implementation must use either:
1. Multi-pass: load K and V in separate passes
2. Reduced tile size: use 64-token tiles instead of 128
3. Absorbed attention: compute attention in the compressed space without explicit up-projection

**TMA (Tensor Memory Accelerator):** H100's TMA enables asynchronous data loading of KV cache tiles, overlapping memory access with computation of the previous tile.

**Warp specialization:** Use producer warps to load and up-project KV tiles while consumer warps compute attention scores and accumulate output.

### 1.2 Huawei Ascend (CANN)

The Ascend implementation uses Cube Units for projections and Vector Units for attention:

**Approach:** Separate the MLA computation into distinct Cube and Vector phases, using the Unified Buffer (UB) for inter-unit data transfer:

Phase 1 (Cube): KV compression -- matmul x @ W_dkv -> c_kv
Phase 2 (Cube): K up-projection -- matmul c_kv @ W_uk -> k_nope
Phase 3 (Cube): V up-projection -- matmul c_kv @ W_uv -> v
Phase 4 (Cube): Q projection -- matmul x @ W_q -> q
Phase 5 (Cube): Attention scores -- matmul q @ k^T
Phase 6 (Vector): Causal mask + softmax
Phase 7 (Cube): Attention output -- matmul attn_probs @ v
Phase 8 (Cube): Output projection -- matmul attn_out @ W_o

The primary optimization is pipelining: while Phase 5 computes attention scores for one block, Phase 2-3 can up-project the next KV block. The Ascend 910B's dual Cube Units (in newer versions) enable this overlap.

**Memory management:** The compressed KV cache (512 dims) fits in L2 for sequences up to ~32K tokens (32K * 512 * 2 bytes = 32 MB, within L2 capacity). Beyond 32K, the KV cache spills to HBM.

### 1.3 General Optimization: Absorbed MLA

The absorbed MLA variant avoids explicit KV up-projection:

```
Standard: score = q @ (c_kv @ W_uk)^T = q @ W_uk^T @ c_kv^T
Absorbed: score = (q @ W_uk^T) @ c_kv^T = q_absorbed @ c_kv^T
```

By absorbing the up-projection weight into the query, the attention computation operates directly on the compressed latent (d_c=512) instead of the expanded KV (n_h * d_nope). This reduces the compute for attention from O(S^2 * d_qk) to O(S^2 * d_c / n_h * ...) but requires careful handling of the RoPE component.

The nope and rope portions must be handled separately since RoPE is position-dependent and cannot be simply absorbed. The absorbed approach works for the nope portion (d_nope=128) but the rope portion (d_rope=64) still requires explicit key computation.

## 2. MoE Dispatch Kernel

### 2.1 NVIDIA CUDA

The dispatch kernel is a critical path for MoE performance:

**Token sorting approach:** Sort all token-expert assignments by expert ID to create contiguous expert batches:
1. Compute routing decisions (gating + grouped top-K)
2. Build (expert_id, token_id, weight) tuples
3. Radix sort by expert_id
4. Build cumulative count array (expert offsets)
5. Gather tokens into expert-contiguous buffers
6. Launch expert GEMMs with variable batch sizes

On H100, the radix sort and gather operations use CUB library primitives. The variable-batch expert GEMMs use either grouped GEMM (CUTLASS) or padding to uniform batch size.

**Shared-memory gather:** For small top-K (8), each warp can handle one token's dispatch by loading the token into shared memory and using warp-level scatter to the appropriate expert buffer.

### 2.2 Huawei Ascend

On Ascend, the dispatch maps to DMA-based data movement:

**Approach:** Use the Scalar Unit for routing computation and DMA engine for token dispatch:
1. Scalar Unit: compute gating logits, grouped top-K
2. DMA: async scatter tokens to per-expert buffers in L2/HBM
3. Cube Unit: process each expert's token batch as a standard GEMM
4. DMA: async gather expert outputs back to original token order
5. Vector Unit: weighted combination of expert outputs

The key optimization is double-buffering: while one batch of tokens is being processed by experts, the next batch's dispatch is happening via DMA.

## 3. SwiGLU FFN Kernel

### 3.1 NVIDIA CUDA

SwiGLU: out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

**Fused approach:** Concatenate W_gate and W_up into a single matrix [W_gate; W_up] of shape (H, 2*I):
1. Single GEMM: x @ [W_gate; W_up] -> [gate_out; up_out] of shape (tokens, 2*I)
2. Fused kernel: SiLU(gate_out) * up_out -> hidden of shape (tokens, I)
3. Single GEMM: hidden @ W_down -> output of shape (tokens, H)

The fused SiLU-multiply kernel (step 2) is element-wise and memory-bound. On H100, it achieves near-peak memory bandwidth. The key optimization is to keep the intermediate in registers between the split and the multiply, avoiding a store/load cycle.

### 3.2 Huawei Ascend

On Ascend, the SwiGLU computation chains Cube and Vector operations:

1. Cube: x @ [W_gate; W_up] -> [gate_out; up_out]
2. Vector: SiLU(gate_out) -> activated (in UB, no HBM round-trip)
3. Vector: activated * up_out -> hidden (in UB)
4. Cube: hidden @ W_down -> output

The Unified Buffer (UB) enables steps 2-3 to execute without writing to HBM, provided the intermediate size (2 * I = 4096 elements per token in BF16 = 8 KB) fits in UB.

## 4. RMSNorm Kernel

### 4.1 NVIDIA CUDA

RMSNorm is a memory-bound kernel (low arithmetic intensity):

```
rms = sqrt(mean(x^2) + eps)
output = x / rms * weight
```

**Optimization:** For hidden_size=7168, each token's RMSNorm requires:
- Read x (7168 * 2 = 14 KB)
- Compute sum of squares (7168 multiplies + adds)
- Read weight (7168 * 2 = 14 KB)
- Write output (14 KB)
- Total: 42 KB memory, 14K FLOPs -> arithmetic intensity = 0.33 FLOP/byte

This is deeply memory-bound. The key optimization is to fuse RMSNorm with the subsequent GEMM (as described in the fusion section) to eliminate the output write.

On H100, a standalone RMSNorm kernel uses a single threadblock per token, with warp-level reduction for the sum-of-squares computation.

### 4.2 Huawei Ascend

On Ascend, RMSNorm runs entirely on the Vector Unit:
1. Load x from HBM to L1
2. Compute x^2 using Vector multiply
3. Reduce sum using Vector add with tree reduction
4. Compute rsqrt using Vector special function unit
5. Multiply x * rsqrt * weight using Vector multiply
6. Store result to UB (for fusion with subsequent Cube operation) or HBM

The Ascend Vector Unit's 256-bit SIMD width processes 16 BF16 elements per cycle. For hidden_size=7168, this requires 448 cycles for the multiply pass, making RMSNorm extremely fast on Ascend.

## Performance Comparison Summary

| Kernel | H100 (BF16) | H100 (FP8) | Ascend 910B (BF16) | Ascend 910B (INT8) |
|--------|-------------|------------|--------------------|--------------------|
| MLA attention (prefill, bs=4, seq=2048) | ~2 ms | ~1.2 ms | ~6 ms | ~4 ms |
| MoE dispatch + compute | ~3 ms | ~2 ms | ~10 ms | ~7 ms |
| SwiGLU FFN (2048 tokens) | ~0.5 ms | ~0.3 ms | ~1.5 ms | ~1.0 ms |
| RMSNorm (2048 tokens) | ~0.02 ms | ~0.02 ms | ~0.05 ms | ~0.05 ms |

These are estimated per-layer latencies. Actual performance depends heavily on implementation quality and hardware-specific optimizations.
