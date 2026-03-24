# DeepSeek-V3 Benchmark Report Template

## Overview

This document serves as a template for DeepSeek-V3 671B benchmark reports. Fill in the sections below with actual benchmark results after running the suite.

**Model:** DeepSeek-V3 671B (arXiv: 2412.19437)
**Hardware:** NVIDIA H100 SXM5 80GB
**Date:** [FILL IN]
**Benchmark version:** 0.1.0

## 1. Environment

| Item | Value |
|------|-------|
| GPU | [GPU model and count] |
| Driver | [NVIDIA driver version] |
| CUDA | [CUDA version] |
| PyTorch | [PyTorch version] |
| Python | [Python version] |

## 2. Micro-benchmark Results

### 2.1 GEMM Throughput

| Shape (M, N, K) | Operation | Mean (ms) | TFLOPS | MFU (%) | Bound |
|-----------------|-----------|-----------|--------|---------|-------|
| (2048, 512, 7168) | MLA down-proj | | | | |
| (2048, 16384, 512) | MLA K up-proj | | | | |
| (2048, 24576, 7168) | MLA Q proj | | | | |
| (64, 2048, 7168) | MoE expert | | | | |
| (2048, 256, 7168) | MoE gating | | | | |
| (2048, 129280, 7168) | LM head | | | | |

### 2.2 Kernel Throughput

| Kernel | Config | Mean (ms) | Notes |
|--------|--------|-----------|-------|
| Softmax | seq=2048, heads=128 | | |
| RoPE (YaRN) | seq=2048, d_rope=64 | | |
| SwiGLU | tokens=2048, I=2048 | | |
| RMSNorm | tokens=2048, H=7168 | | |

## 3. Component-level Results

### 3.1 MLA Layer

| Batch | SeqLen | Mean (ms) | MFU (%) | Tokens/s | Peak Mem (MB) |
|-------|--------|-----------|---------|----------|---------------|
| 1 | 512 | | | | |
| 1 | 2048 | | | | |
| 4 | 512 | | | | |
| 4 | 2048 | | | | |

**Key observations:**
- [Note MLA's memory efficiency from KV compression]
- [Note compute vs memory bound transition point]

### 3.2 MoE FFN Layer

| Batch | SeqLen | Mean (ms) | MFU (%) | Tokens/s | Peak Mem (MB) |
|-------|--------|-----------|---------|----------|---------------|
| 1 | 512 | | | | |
| 1 | 2048 | | | | |
| 4 | 512 | | | | |
| 4 | 2048 | | | | |

**Key observations:**
- [Note expert dispatch overhead]
- [Note load balance quality under grouped routing]

### 3.3 Dense FFN Layer

| Batch | SeqLen | Mean (ms) | MFU (%) | Tokens/s | Peak Mem (MB) |
|-------|--------|-----------|---------|----------|---------------|
| 1 | 2048 | | | | |
| 4 | 2048 | | | | |

## 4. MoE Sweep Results

### 4.1 Expert Count Impact

| Experts | Top-K | Groups | Mean (ms) | MFU (%) | Load CV |
|---------|-------|--------|-----------|---------|---------|
| 64 | 8 | 2 | | | |
| 128 | 8 | 4 | | | |
| 256 | 8 | 8 | | | |

### 4.2 Load Balance Analysis

| Groups | TopK_Group | CV | Max/Min Ratio | Efficiency |
|--------|------------|----|---------------|------------|
| 1 (flat) | 1 | | | |
| 4 | 2 | | | |
| 8 | 4 | | | |
| 16 | 8 | | | |

## 5. FP8 Pareto Frontier

### 5.1 Format Comparison

| Format | Block Size | TFLOPS | MFU (%) | RMSE | SNR (dB) |
|--------|-----------|--------|---------|------|----------|
| BF16 | N/A | | | baseline | |
| FP8 e4m3 | 128x128 | | | | |
| FP8 e4m3 | 64x64 | | | | |
| FP8 e4m3 | per-tensor | | | | |
| FP8 e5m2 | 128x128 | | | | |

### 5.2 Block Size Impact on Accuracy

| Block Size | RMSE | SNR (dB) | Scale Overhead (%) |
|-----------|------|----------|--------------------|
| Per-tensor | | | 0.00 |
| 256x256 | | | |
| 128x128 | | | |
| 64x64 | | | |
| 32x32 | | | |

## 6. MFU Ceiling Analysis

### 6.1 Roofline Model

| Operation | AI (FLOP/B) | Bound | Achievable TFLOPS | Efficiency (%) |
|-----------|-------------|-------|-------------------|----------------|
| MLA KV down | | | | |
| MLA Q proj | | | | |
| Attention QKT | | | | |
| MoE gating | | | | |
| MoE expert FFN | | | | |
| LM head | | | | |

### 6.2 Full Model MFU Estimate

| Batch | SeqLen | Est. MFU (%) | Dominant Component | Time (ms) |
|-------|--------|--------------|--------------------|-----------|
| 1 | 512 | | | |
| 4 | 2048 | | | |
| 16 | 4096 | | | |

## 7. End-to-End Results

| Scale | Batch | SeqLen | Mean (ms) | Tokens/s | MFU (%) | Peak Mem (MB) |
|-------|-------|--------|-----------|----------|---------|---------------|
| 0.01 | 1 | 2048 | | | | |
| 0.01 | 4 | 2048 | | | | |
| 0.05 | 1 | 2048 | | | | |

## 8. Key Findings

1. **MLA efficiency:** [Summarize MLA KV compression benefit]
2. **MoE scaling:** [Summarize MoE performance characteristics]
3. **FP8 benefit:** [Summarize FP8 speedup with accuracy impact]
4. **Bottlenecks:** [Identify primary bottlenecks]

## 9. Optimization Recommendations

1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

## 10. Methodology Notes

- All timings use CUDA events with bootstrap 95% confidence intervals (1000 resamples)
- Warmup: 10 iterations discarded before measurement
- Benchmark iterations: 100 per configuration
- Memory measured via torch.cuda.max_memory_allocated()
- MFU calculated against H100 SXM5 peak BF16 (989.5 TFLOPS) or FP8 (1979 TFLOPS)
