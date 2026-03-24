# Overview: Domestic Chip Landscape for DeepSeek-V3 Inference

## Context

The deployment of DeepSeek-V3 (671B parameters, arXiv: 2412.19437) on domestic Chinese accelerators presents unique challenges due to the model's architectural innovations: Multi-head Latent Attention (MLA), Mixture of Experts with grouped routing (256 experts, top-8 selection), and FP8 training with 128x128 block quantization. While the model was trained on NVIDIA H800 GPUs, inference deployment increasingly targets domestic hardware due to export restrictions and strategic considerations.

## Domestic Accelerator Landscape

The primary domestic accelerators under consideration for DeepSeek-V3 inference include:

**Huawei Ascend 910B/910C:** The most mature domestic AI accelerator platform. The Ascend 910B provides up to 320 TFLOPS FP16 performance with 64GB HBM2e bandwidth of 1.6 TB/s. The Ascend architecture uses a Da Vinci core design with specialized Cube Units for matrix operations and Vector Units for element-wise computations. The CANN (Compute Architecture for Neural Networks) software stack provides operator-level programmability through the AscendCL API.

**Cambricon MLU370/MLU590:** Cambricon's data center accelerators provide INT8/FP16 compute with custom BANG language for kernel development. The MLU590 targets 256 TFLOPS FP16 with HBM2e memory. Cambricon's MagicMind inference framework supports standard model formats but requires custom kernel development for MLA's non-standard attention pattern.

**Biren BR100/BR104:** Biren's GPGPU architecture provides 1024 TFLOPS INT8 with PCIe Gen5 connectivity. While impressive on paper, software ecosystem maturity lags behind Huawei Ascend. The BR104 targets edge/cloud inference scenarios.

**Enflame DTU (GCU):** Enflame's General Compute Unit targets data center inference with a dataflow architecture. The DTU's explicit memory hierarchy may benefit from MLA's structured data access patterns, but custom kernel development is required.

## Key Challenges for DeepSeek-V3 on Domestic Hardware

The primary challenges stem from four architectural aspects:

First, MLA's compressed KV cache mechanism requires non-standard attention kernels. The standard attention implementations available on domestic accelerators assume full multi-head KV, not the compressed latent representation. Adapting FlashAttention-style kernels for MLA on non-CUDA architectures is a significant engineering effort.

Second, the MoE layer with 256 experts and grouped routing creates complex data movement patterns. The all-to-all communication required for expert dispatch must be efficiently mapped to each platform's interconnect topology. Huawei Ascend uses HCCS (Huawei Cache Coherent System) for intra-node communication, which has different bandwidth characteristics than NVLink.

Third, FP8 support varies across platforms. DeepSeek-V3's FP8 quantization scheme (e4m3 with 128x128 blocks) requires either native FP8 hardware support or efficient emulation. Most domestic accelerators support INT8 but not FP8 natively, necessitating software-level quantization to INT8 or mixed-precision approaches.

Fourth, the software ecosystem gap means that many optimized kernels (FlashAttention, Triton-based MoE dispatch, DeepGEMM FP8) are CUDA-specific and must be rewritten for each domestic platform's programming model.

## Deployment Strategy

Given these challenges, we recommend a phased approach:

Phase 1: Deploy with INT8 weight quantization on Huawei Ascend 910B using standard attention (without absorbed MLA optimization). This sacrifices some throughput but provides a working baseline.

Phase 2: Develop custom MLA kernels for Ascend's Cube Units, implementing the compressed KV attention pattern natively. This requires deep knowledge of AscendCL and CANN operator development.

Phase 3: Optimize MoE dispatch for the target platform's interconnect, implement grouped routing in hardware-aware fashion, and add FP8 emulation or conversion to INT8 with minimal quality loss.

## Performance Expectations

Based on preliminary analysis, we expect the following performance ratios relative to H100:

| Platform | vs H100 (FP16) | vs H100 (INT8) | Memory (GB) | Notes |
|----------|----------------|-----------------|-------------|-------|
| Ascend 910B | ~40-50% | ~45-55% | 64 | Best software ecosystem |
| MLU590 | ~30-40% | ~35-45% | 48 | BANG kernel development needed |
| BR100 | ~25-35% | ~40-50% | 64 | Immature software stack |
| Enflame GCU | ~20-30% | ~30-40% | 32 | Dataflow advantages for MLA |

These estimates account for both raw hardware capabilities and software optimization maturity. The gap with H100 is primarily software-driven rather than hardware-limited.
