# FlashInfer Kernel Integration Plan for DeepSeek-V3

## Executive Summary

This document outlines the integration plan for FlashInfer as the primary attention kernel backend for DeepSeek-V3 (arXiv: 2412.19437). FlashInfer's FA3 backend is selected because DeepSeek-V3's `qk_nope_head_dim=128` natively matches FlashInfer's supported head dimensions, eliminating the need for monkey-patching that would be required with FlashMLA.

## Phase 1: Core MLA Attention Integration (Week 1-2)

### 1.1 BatchMLAPagedAttentionWrapper Setup

The primary entry point is `flashinfer.mla.BatchMLAPagedAttentionWrapper`:

```
Input tensors:
  q_nope: (batch, num_heads=128, d_ckv=512)     -- compressed query (nope part)
  q_pe:   (batch, num_heads=128, d_kpe=64)       -- query rope part
  ckv:    (total_kv_tokens, 1, d_ckv=512)         -- compressed KV cache
  kpe:    (total_kv_tokens, 1, d_kpe=64)          -- key rope embeddings

Output:
  o:      (batch, num_heads=128, d_ckv=512)       -- attention output (in latent space)
```

The wrapper requires a `plan()` call before `run()`, which builds the internal scheduling:
- `qo_indptr`: cumulative query offsets per batch element
- `kv_indptr`: cumulative KV offsets per batch element
- `kv_indices`: page indices for paged KV cache
- `kv_lens`: per-sequence KV lengths

### 1.2 KV Cache Format

FlashInfer's paged KV cache format:
- Pages: `[num_pages, page_size, 576]` (512 ckv + 64 kpe concatenated)
- FP8 variant: same layout with `torch.float8_e4m3fn` dtype, external scale tensor
- Page size: configurable (1 for decode, 64 for prefill)

Memory per token per layer: 576 bytes (FP8) or 1152 bytes (BF16)
Total KV cache for 128K context, 61 layers: 576 * 128K * 61 = 4.3 GB (FP8) or 8.6 GB (BF16)

### 1.3 Scaling Factor

The attention scaling factor for MLA with absorbed projections:
```
sm_scale = 1.0 / sqrt(qk_nope_head_dim + qk_rope_head_dim)
         = 1.0 / sqrt(128 + 64)
         = 1.0 / sqrt(192)
         ≈ 0.0722
```

### 1.4 Causal Masking

DeepSeek-V3 uses standard causal attention (no DSA). FlashInfer's `causal=True` flag in `plan()` handles this natively. No custom mask logic is needed.

## Phase 2: FP8 KV Cache Integration (Week 2-3)

### 2.1 Quantization Flow

```
Forward pass:
  1. Compute c_kv = W_dkv @ x_norm                  [BF16, shape: (B, kv_lora_rank=512)]
  2. Compute k_pe = RoPE(W_kpe @ c_kv)              [BF16, shape: (B, qk_rope_head_dim=64)]
  3. Concatenate: kv_entry = cat(c_kv, k_pe, dim=-1) [BF16, shape: (B, 576)]
  4. Quantize to FP8: kv_fp8, scale = quant(kv_entry) [FP8 E4M3]
  5. Store in paged cache

Decode attention:
  1. Load page from FP8 cache
  2. Dequantize with stored scale
  3. Split into ckv (512) and kpe (64)
  4. Run FlashInfer FA3 MLA kernel
```

### 2.2 Scale Management

Two options for FP8 scale storage:
- **Per-page scale**: One scale per page (simple, slight precision loss)
- **Per-head scale**: One scale per head per page (better precision, more memory)

Recommendation: Per-page scale for decode (bandwidth-bound anyway), per-head for prefill.

### 2.3 Block Quantization Alignment

DeepSeek-V3 uses 128x128 block FP8 for training. For KV cache, we use per-entry (576-dim) quantization since the KV entries are independent. The 576-dim entry does not align to 128-block, so we use a global scale per entry.

## Phase 3: Prefill Optimization (Week 3-4)

### 3.1 Chunked Prefill

For long context (up to 128K tokens), chunked prefill is necessary:
- Chunk size: 8192 tokens (fits in SM L2)
- Each chunk processes: attention scores against all prior chunks + current
- FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` handles this

### 3.2 Prefill-Decode Pipeline

During generation, the system alternates between:
- **Prefill**: Process new prompt tokens (compute-bound, uses FA3 prefill)
- **Decode**: Generate one token at a time (memory-bound, uses FA3 decode)

FlashInfer supports both modes with the same KV cache, switching via the wrapper.

## Phase 4: CUDA Graph Integration (Week 4-5)

### 4.1 Graph Capture Requirements

For decode, CUDA graph capture requires:
- Static tensor shapes (fixed batch size + max sequence length)
- FlashInfer's `begin_forward()` / `end_forward()` pattern
- Pre-allocated workspace buffer (128MB recommended)

### 4.2 Graph-Compatible KV Cache Updates

The KV cache append operation must be graph-compatible:
- Use `flashinfer.page.append_paged_kv_cache()` which supports graph capture
- Pre-allocate page tables with maximum expected pages

### 4.3 Expected Speedup

Based on FlashInfer benchmarks:
- Decode (B=1, seq=4096): 2-3x speedup from graph capture
- Decode (B=32, seq=4096): 1.5-2x speedup
- Prefill: No graph capture benefit (already compute-bound)

## Phase 5: Integration Testing (Week 5-6)

### 5.1 Correctness Tests

1. Compare FlashInfer FA3 output against PyTorch eager MLA
2. Compare FP8 KV cache against BF16 KV cache (cosine sim > 0.99)
3. Verify causal masking (future tokens have zero attention weight)
4. Test variable sequence lengths in a batch
5. Test page boundary crossing (sequence spanning multiple pages)

### 5.2 Performance Tests

1. Decode latency vs batch size (B=1..128)
2. Decode latency vs KV length (128..128K)
3. Prefill throughput vs sequence length
4. Memory usage vs context length
5. TMA utilization (bandwidth > 2TB/s on H100)

### 5.3 Determinism Tests

1. Same input produces identical output across 100 runs
2. Determinism across CUDA streams
3. Determinism with CUDA graphs

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| flashinfer | >= 0.2.0 | FA3 MLA kernels |
| torch | >= 2.4.0 | Framework |
| triton | >= 3.0.0 | Autotuned MoE kernels |
| deep-gemm | latest | FP8 grouped GEMM |

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FlashInfer API change | Medium | High | Pin version, abstract behind wrapper |
| FP8 precision regression | Low | Medium | Continuous precision monitoring |
| CUDA graph incompatibility | Medium | Medium | Fallback to eager for problematic paths |
| Memory fragmentation | Low | High | Page-based allocation, defrag pass |

## Milestones

- [ ] Phase 1: MLA attention produces correct output (cosine sim > 0.999 vs eager)
- [ ] Phase 2: FP8 KV cache roundtrip error < 1%
- [ ] Phase 3: Prefill supports 128K context without OOM on 80GB H100
- [ ] Phase 4: CUDA graph decode achieves > 2x speedup over eager
- [ ] Phase 5: All tests pass, performance meets targets
