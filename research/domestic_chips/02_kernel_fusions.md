# Kernel Fusion Strategies for DeepSeek-V3

## Overview

Kernel fusion is critical for DeepSeek-V3 performance on all hardware platforms. The model's architecture creates multiple opportunities for fusing sequential operations to eliminate intermediate memory round-trips. This document analyzes fusion opportunities specific to DeepSeek-V3's MLA, MoE, and MTP components.

## MLA Fusion Opportunities

### Fusion 1: RMSNorm + KV Down-Projection

The MLA layer begins with RMSNorm followed by the KV compression projection. These can be fused:

```
Standard: x -> RMSNorm(x) -> store -> load -> W_dkv @ x_norm -> c_kv
Fused:    x -> RMSNorm_GEMM(x, W_dkv) -> c_kv
```

The fused kernel computes RMSNorm in registers/shared memory and immediately feeds the normalized result to the GEMM without writing to global memory. Savings: one full read+write of the hidden state (7168 dims per token). On H100, this saves ~14KB per token of HBM bandwidth. For a batch of 4096 tokens, that is 57 MB of eliminated traffic.

On domestic chips (Ascend), this fusion maps to the Unified Buffer: the Vector Unit computes RMSNorm, places the result in UB, and the Cube Unit reads from UB for the GEMM.

### Fusion 2: KV Down-Project + K/V Up-Project

The MLA latent c_kv (512 dims) is up-projected to both K_nope (128 * 128 = 16384 dims) and V (128 * 128 = 16384 dims). These two up-projections can share the same input without reloading c_kv:

```
Standard: c_kv -> store -> load -> W_uk @ c_kv -> k_nope
          c_kv -> load -> W_uv @ c_kv -> v  (second load of c_kv)
Fused:    c_kv -> [W_uk; W_uv] @ c_kv -> [k_nope; v]  (single load)
```

This concatenation fusion works when the combined output dimension (32768) fits the tiling strategy. On H100, the concatenated weight matrix is (512, 32768) which is 32MB in BF16, fitting in L2 cache.

### Fusion 3: Attention Score Computation

In MLA, the attention score combines nope and rope components:

```
q = [q_nope; q_rope]  (128 + 64 = 192 dims per head)
k = [k_nope; k_rope]  (128 + 64 = 192 dims per head)
score = q @ k^T = q_nope @ k_nope^T + q_rope @ k_rope^T
```

Rather than computing two separate dot products, a fused kernel handles both in a single pass over the sequence dimension. This is a natural extension of FlashAttention that simply uses a 192-dim head instead of a standard 128-dim head.

### Fusion 4: Softmax + Value Projection

After computing attention scores, the softmax and value projection can be fused in a FlashAttention-style online reduction:

```
For each query position:
  running_max = -inf
  running_sum = 0
  running_output = 0
  For each key block:
    scores = q @ k_block^T
    local_max = max(scores)
    correction = exp(running_max - new_max)
    running_output = running_output * correction + exp(scores - new_max) @ v_block
    running_sum = running_sum * correction + sum(exp(scores - new_max))
  output = running_output / running_sum
```

This eliminates the materialization of the full (seq x seq) attention matrix.

## MoE Fusion Opportunities

### Fusion 5: Gating + Top-K + Dispatch

The MoE routing pipeline involves three steps that can be partially fused:

```
Standard: x @ W_gate -> logits -> reshape(groups) -> group_topk -> expert_topk -> scatter
Fused:    x @ W_gate -> fused_grouped_topk_scatter(logits, n_group=8, topk_group=4, top_k=8)
```

The fused kernel computes the GEMM for gating, then immediately performs the grouped top-K selection in registers. The scatter (dispatch) can be initiated from the same kernel using asynchronous DMA, overlapping the dispatch of early tokens with the routing computation of later tokens.

On NVIDIA GPUs, this uses cooperative groups and shared memory for the group scoring. On Ascend, the Scalar Unit handles the control flow while DMA initiates the transfers.

### Fusion 6: Expert SwiGLU

Each expert's FFN is a SwiGLU: gate_proj, up_proj, element-wise multiply, down_proj. The first two projections share the input:

```
Standard: x -> W_gate @ x -> silu -> store -> load -> elementwise_mul
          x -> W_up @ x -> store -> load -> elementwise_mul
Fused:    x -> [W_gate; W_up] @ x -> silu_gate_fused -> down_proj
```

The concatenated gate+up GEMM (7168 -> 4096 concatenated) feeds directly into a fused SiLU-multiply kernel. This eliminates intermediate stores of the gate and up projections.

### Fusion 7: Expert Combine + Residual

After expert computation, the weighted combination and residual addition can be fused:

```
Standard: expert_outputs -> weighted_sum(outputs, weights) -> store -> load -> residual_add
Fused:    expert_outputs -> fused_combine_residual(outputs, weights, residual)
```

## MTP Fusion

### Fusion 8: MTP Projection + LM Head

The MTP module projects hidden states and then applies the (shared) LM head:

```
Standard: hidden -> W_mtp @ hidden -> store -> load -> W_lm_head @ mtp_hidden -> logits
Fused:    hidden -> W_mtp @ hidden -> W_lm_head @ result -> logits (chained GEMM)
```

Since the MTP projection output (7168 dims) is the same size as the input to the LM head, these two GEMMs can be chained with the intermediate result staying in shared memory/L2 cache.

## Cross-Platform Fusion Implementation

### NVIDIA H100 (CUDA/Triton)
- FlashAttention-3 already provides fusions 3 and 4
- Triton kernels for fusions 1, 5, 6
- CUTLASS for fused GEMM pairs (fusions 2, 7, 8)

### Huawei Ascend (CANN)
- Unified Buffer enables fusions 1, 6 between Cube and Vector Units
- DMA engine supports fusion 5 (async dispatch)
- CANN custom operators for fusions 2-4, 7-8

### AMD MI300X (ROCm)
- Composable Kernel framework for attention fusions
- hipBLASLt for fused GEMM pairs
- Custom HIP kernels for MoE dispatch

## Expected Impact

| Fusion | Memory Savings | Latency Reduction | Complexity |
|--------|---------------|-------------------|------------|
| 1. RMSNorm + GEMM | 14 KB/token | 5-10% per layer | Medium |
| 2. KV Down + Up | 1 KB/token | 3-5% per layer | Low |
| 3. Nope + Rope attn | 0 (compute) | 2-3% attention | Medium |
| 4. Flash-style MLA | seq^2 * 2B | 30-50% attention | High |
| 5. Gate + TopK + Scatter | 2 KB/token | 10-15% routing | High |
| 6. Expert SwiGLU | 16 KB/token | 5-8% per expert | Medium |
| 7. Combine + Residual | 14 KB/token | 3-5% per layer | Low |
| 8. MTP chain | 14 KB/token | 5-8% MTP module | Low |

Combined, these fusions can reduce per-layer latency by 20-40% depending on batch size and sequence length. Memory-bound decode benefits more from fusions (since they reduce memory traffic), while compute-bound prefill benefits less.
