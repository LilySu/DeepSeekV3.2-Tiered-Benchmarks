# Per-Platform Specifications for DeepSeek-V3 Deployment

## Hardware Comparison Matrix

| Specification | NVIDIA H100 SXM5 | NVIDIA H800 | Huawei Ascend 910B | Cambricon MLU590 | Biren BR100 |
|--------------|-------------------|-------------|--------------------|--------------------|-------------|
| FP16 TFLOPS | 989.5 | 989.5 | 320 | 256 | 512 |
| BF16 TFLOPS | 989.5 | 989.5 | 320 | 256 | 512 |
| FP8 TFLOPS | 1979 | 1979 | N/A (emulated) | N/A | N/A |
| INT8 TOPS | 1979 | 1979 | 640 | 512 | 1024 |
| HBM Type | HBM3 | HBM2e | HBM2e | HBM2e | HBM2e |
| HBM Capacity | 80 GB | 80 GB | 64 GB | 48 GB | 64 GB |
| HBM Bandwidth | 3.35 TB/s | 2.0 TB/s | 1.6 TB/s | 1.2 TB/s | 1.5 TB/s |
| Interconnect | NVLink 4.0 | NVLink (limited) | HCCS | MLU-Link | N/A |
| Interconnect BW | 900 GB/s | 400 GB/s | 56 GB/s | 192 GB/s | PCIe 5.0 |
| L2 Cache | 50 MB | 50 MB | 32 MB | 24 MB | 48 MB |
| TDP | 700W | 700W | 350W | 300W | 550W |
| Process Node | TSMC 4N | TSMC 4N | 7nm | 7nm | 7nm |

## Per-Platform DeepSeek-V3 Deployment Analysis

### NVIDIA H100 SXM5 (Reference Platform)

**Minimum chips:** 8 (FP8 weights: 671B bytes / 8 = 84 GB; 8 chips = 640 GB)
**Optimal chips:** 8-16 for expert parallelism aligned with 8 groups

The H100 is the reference platform with full FP8 support. DeepSeek-V3's FP8 e4m3 training directly maps to H100's 4th-gen tensor cores. Key advantages:
- Native FP8 compute at 1979 TFLOPS
- 3.35 TB/s HBM bandwidth enables large-batch decode
- 900 GB/s NVLink for efficient expert-parallel all-to-all
- FlashAttention-3 with TMA support for MLA attention

### NVIDIA H800 (Training Platform)

**Note:** DeepSeek-V3 was trained on 2048 H800 GPUs. The H800 is the export-compliant version of H100 with reduced NVLink bandwidth (400 GB/s vs 900 GB/s). For inference, the lower interconnect bandwidth increases MoE dispatch latency by approximately 2.25x for cross-GPU expert communication. MLA computation is unaffected as it is per-GPU.

### Huawei Ascend 910B

**Minimum chips:** 12 (INT8 weights: 84 GB; 12 chips = 768 GB available after overhead)
**Optimal chips:** 16 for expert-parallel alignment

The Ascend 910B requires custom kernel development for optimal DeepSeek-V3 performance. Key considerations:

Memory: 64 GB HBM2e at 1.6 TB/s is 48% of H100 bandwidth. This directly impacts decode throughput for memory-bound scenarios. MLA's compressed KV cache (512 dims vs ~32K for standard MHA) helps mitigate this by reducing KV cache bandwidth requirements.

Compute: 320 TFLOPS FP16 is 32% of H100. For INT8, 640 TOPS is 32% of H100 as well. The gap is consistent, suggesting a 3x latency multiplier for compute-bound operations.

Interconnect: 56 GB/s HCCS is 16x slower than NVLink. This makes expert-parallel MoE prohibitively expensive across chips. The mitigation is to replicate experts across chips or use pipeline parallelism instead of expert parallelism.

Software: CANN 7.0+ supports custom operators with acceptable performance. The primary development effort is custom MLA attention kernels and optimized MoE dispatch.

### Cambricon MLU590

**Minimum chips:** 14 (INT8: 84 GB; 14 chips = 672 GB)
**Optimal chips:** 16

The MLU590 provides 256 TFLOPS FP16 and 512 TOPS INT8. Key considerations:

The BANG programming model is lower-level than CUDA, requiring explicit memory management and pipeline scheduling. MLA kernel development requires understanding of the MLU's pipeline stages and memory hierarchy.

MLU-Link provides 192 GB/s interconnect bandwidth, better than HCCS but still 4.7x slower than NVLink. This makes expert parallelism feasible with careful optimization but not at H100 efficiency levels.

The 48 GB HBM capacity means DeepSeek-V3 requires at least 14 chips for INT8 weights alone, with additional chips needed for KV cache and activations. A 16-chip deployment provides headroom.

### Biren BR100

**Status:** Early-stage software ecosystem. The BR100 provides competitive INT8 throughput (1024 TOPS) but lacks the software stack for complex model deployments like DeepSeek-V3.

The primary challenge is the absence of high-bandwidth chip-to-chip interconnect (relies on PCIe 5.0), making distributed MoE inference impractical. The BR100 is better suited for smaller models or single-chip inference of quantized models.

## Memory Budget Analysis

For a 16-chip deployment targeting low-latency inference:

| Component | Size (INT8) | Size (BF16) | Notes |
|-----------|------------|------------|-------|
| Model weights | 84 GB | 168 GB | 671B params |
| KV cache (4K ctx) | 0.5 GB | 1.0 GB | 61 layers x 512 dims x 4096 tokens |
| KV cache (32K ctx) | 4.0 GB | 8.0 GB | 61 layers x 512 dims x 32768 tokens |
| KV cache (128K ctx) | 16.0 GB | 32.0 GB | 61 layers x 512 dims x 131072 tokens |
| Activations | 2-8 GB | 4-16 GB | Depends on batch size |
| MoE routing buffers | 0.5 GB | 1.0 GB | Token dispatch/combine |
| **Total (32K ctx)** | **~91 GB** | **~193 GB** | |

MLA's compressed KV cache is a significant advantage: at 32K context, the KV cache is only 4 GB in INT8, compared to approximately 128 GB for standard MHA with 128 heads and 256-dim KV. This 32x reduction is the single most impactful architectural decision for deployment memory efficiency.

## Performance Projections

Expected throughput for DeepSeek-V3 inference (tokens/second, batch=1, seq=2048):

| Platform | Chips | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | Notes |
|----------|-------|-----------------|----------------|-----------|-------|
| H100 SXM5 | 8 | ~12,000 | ~40 | ~170 | FP8, expert parallel |
| H800 | 8 | ~11,000 | ~38 | ~186 | Reduced NVLink impact |
| Ascend 910B | 16 | ~4,000 | ~15 | ~510 | INT8, custom kernels |
| MLU590 | 16 | ~3,000 | ~12 | ~680 | INT8, custom kernels |
| BR100 | 16 | ~2,000 | ~8 | ~1000 | Limited by PCIe |

These projections assume optimized kernels and efficient parallelism strategies. Actual performance may vary significantly based on software optimization maturity.
