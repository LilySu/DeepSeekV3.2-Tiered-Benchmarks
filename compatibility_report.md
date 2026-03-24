# DeepSeek-V3 Kernel Compatibility Report

## Executive Summary

DeepSeek-V3 (671B, arXiv 2412.19437) is a Mixture-of-Experts model with Multi-head Latent Attention (MLA). This report assesses kernel compatibility for H100 inference and training.

**Key finding:** DeepSeek-V3 has EXCELLENT kernel support because the model's creators (DeepSeek) released both FlashMLA and DeepGEMM specifically for this architecture.

## Component Compatibility Matrix

| Component | FlashMLA | FlashInfer | DeepGEMM | Triton | PyTorch |
|-----------|----------|------------|----------|--------|---------|
| MLA Attention (absorbed) | Native | Supported | FP8 logits | Custom | Eager |
| MLA Attention (non-absorbed) | No | Partial | No | Custom | Eager |
| MoE Expert GEMM (BF16) | N/A | N/A | No | Grouped | Loop |
| MoE Expert GEMM (FP8) | N/A | N/A | Native | Custom | No |
| MoE Routing (sigmoid+group) | N/A | N/A | N/A | Fused | Eager |
| RMSNorm | N/A | N/A | N/A | Fused | Eager |
| SwiGLU | N/A | N/A | N/A | Fused | Eager |
| YaRN RoPE | N/A | N/A | N/A | Standard | Standard |
| MTP | N/A | N/A | N/A | Standard | Standard |
| Cross-Entropy | N/A | N/A | N/A | Chunked | Standard |

## Critical Differences from GLM-5

### 1. No DSA (Dynamic Sparse Attention)
DeepSeek-V3 uses standard causal attention. The DSA indexer, sparse masks, and associated kernels from GLM-5 are NOT needed. This significantly simplifies the attention kernel requirements.

### 2. Group-Based Routing
DeepSeek-V3 uses n_group=8, topk_group=4 routing, which divides 256 experts into 8 groups of 32. This is more complex than GLM-5's flat routing (n_group=1) but has a dedicated fused kernel in SGLang.

### 3. YaRN RoPE
DeepSeek-V3 extends context from 4096 to 163,840 positions using YaRN scaling. This only affects the initialization of RotaryEmbedding (computing modified inv_freq), not the runtime RoPE application.

### 4. FP8 Quantization
DeepSeek-V3 natively supports FP8 (e4m3) with 128x128 block quantization. This is directly supported by DeepGEMM's grouped GEMM kernels.

## Recommended Kernel Stack

### Inference (highest throughput)
- **Attention:** FlashMLA (absorbed mode)
- **MoE dispatch:** DeepGEMM FP8 grouped GEMM
- **Routing:** SGLang fused gate (noaux_tc)
- **Normalization:** Triton fused RMSNorm
- **Activation:** Triton fused SwiGLU
- **Loss:** N/A (inference only)

### Training (with gradients)
- **Attention:** PyTorch eager (for gradient through kv_b_proj) or FlashInfer
- **MoE dispatch:** Triton grouped GEMM (unsloth) in BF16
- **Routing:** PyTorch eager (for gradient through router)
- **Normalization:** Triton fused RMSNorm (backward supported)
- **Activation:** Triton fused SwiGLU (backward supported)
- **Loss:** Triton chunked cross-entropy

## Hardware Requirements

### Minimum (inference, FP8)
- 2x H100 80GB (FP8 weights ~150GB + KV cache)
- NVLink 4.0 for tensor parallelism

### Recommended (inference, BF16)
- 8x H100 80GB (BF16 weights ~1.3TB)
- NVLink 4.0 + NVSwitch

### Training
- 64+ H100 80GB (3D parallelism: TP + EP + PP)
