# DeepSeek-V3 Academic Benchmark Research

## Overview

This document tracks the academic benchmarking methodology used for evaluating
DeepSeek-V3 kernel implementations, following FlashAttention-3 and
MoE-Inference-Bench reporting standards.

## Benchmark Standards

### FlashAttention-3 Methodology (Tri Dao, 2024)
- **Metric:** MFU (Model FLOPs Utilization) as percentage of H100 peak
- **Reference:** FA3 achieves 75% MFU on H100 for standard attention
- **Warmup:** 10 iterations (JIT compilation, thermal stabilization)
- **Measurement:** 100 iterations with CUDA events
- **Statistics:** Median, p99, bootstrap 95% CI

### MoE-Inference-Bench (SC '25)
- **Sweep dimensions:** batch_size x seq_len x num_experts x top_k x ffn_dim
- **Hardware:** 4x H100 SXM5 with NVLink
- **Metrics:** Throughput (tokens/s), latency (ms), MFU, memory
- **Roofline analysis:** Compute-bound vs memory-bound classification

### MLPerf v5.1 SLA Compliance
- **TTFT (Time to First Token):** p99 < 2000 ms
- **TPOT (Time Per Output Token):** p99 < 80 ms

## DeepSeek-V3 Specific Benchmarks

### MLA Attention Benchmarks
- **Prefill:** Varying seq_len (128, 512, 2048, 8192, 32768) at batch=1
- **Decode:** Varying batch (1, 16, 64, 256) at context_len=4096
- **Absorbed vs non-absorbed:** Compare compressed KV cache performance
- **YaRN positions:** Test at 4K, 32K, 128K, 163K positions

### MoE Expert Dispatch Benchmarks
- **Expert sweep:** 8, 64, 128, 256 experts
- **Top-k sweep:** 2, 4, 8 active experts
- **FFN dim sweep:** 1024, 2048, 4096
- **Batch sweep:** 1, 16, 32, 64 concurrent sequences
- **FP8 vs BF16:** Throughput and precision comparison

### Group Routing Benchmarks
- **Group configurations:** n_group=1 (flat), 4, 8, 16
- **Topk_group sweep:** 1, 2, 4 groups selected
- **Load balance metrics:** Expert utilization variance, token drop rate

### MTP Benchmarks
- **Acceptance rate:** Average tokens accepted per speculation step
- **Latency tradeoff:** Speculation overhead vs verification savings
- **Batch interaction:** How MTP acceptance scales with batch size

## Research Questions

1. **MLA compression ratio vs quality:** How does kv_lora_rank=512 compare to
   higher ranks (1024, 2048) in terms of attention quality and latency?

2. **Group routing efficiency:** Does n_group=8 provide meaningful load balance
   improvement over flat routing for different batch sizes?

3. **FP8 block size sensitivity:** Is 128x128 optimal, or would 64x64 or
   256x256 provide better precision-throughput Pareto?

4. **YaRN vs NTK-aware scaling:** How does YaRN compare to alternative RoPE
   scaling methods at very long contexts (>100K)?

5. **MTP draft quality:** Does the 1-layer MTP provide sufficient draft quality
   for efficient speculative decoding at batch>1?
