# Research Directions for DeepSeek-V3 Optimization

This document outlines open research directions for optimizing DeepSeek-V3 inference and training on current and next-generation hardware.

## 1. MLA Kernel Optimization

### 1.1 Fused MLA Projection Kernels
The MLA mechanism involves multiple sequential projections: input to latent (down-projection), latent to K/V (up-projection), and the separate RoPE key projection. Fusing these into a single kernel would eliminate intermediate memory round-trips. The challenge is that the down-projection produces a 512-dimensional latent that fans out to 16384 (128 heads times 128 dim) for K_nope and separately for V, making the fusion non-trivial.

Research questions:
- Can we fuse the down-proj + up-proj into a single kernel using shared memory for the latent?
- What is the optimal tiling strategy for the asymmetric shapes (7168->512->16384)?
- How does FlashAttention-3 need to be modified to accept pre-compressed KV inputs?

### 1.2 Absorbed MLA Optimization
The DeepSeek-V3 paper describes an "absorbed" variant where the up-projection weight is absorbed into the attention computation, avoiding explicit materialization of the full K/V. This reduces memory bandwidth but changes the attention kernel's compute pattern. Benchmarking the absorbed vs. non-absorbed variants across different batch sizes and sequence lengths would quantify the crossover point.

### 1.3 KV Cache Compression During Decode
During autoregressive decode, the KV cache stores the compressed latent (512 dims per layer per token) rather than full KV heads. This is already a 32x reduction, but further compression via quantizing the latent to INT4 or even binary could push the boundaries of how many tokens can be cached on a single GPU.

## 2. MoE Optimization

### 2.1 Expert-Parallel Communication Patterns
With 256 experts and 8-way expert parallelism, the all-to-all communication pattern during MoE dispatch is the primary distributed bottleneck. Research into:
- Hierarchical all-to-all (intra-node NVLink first, inter-node InfiniBand second)
- Overlapping expert computation with communication
- Token dropping strategies that maintain quality while reducing communication volume

### 2.2 Load Balancing Without Auxiliary Loss
DeepSeek-V3 introduced auxiliary-loss-free load balancing using a bias term on the gating logits. The bias is adjusted based on observed expert loads. Open questions:
- What is the optimal bias update schedule?
- Can the bias be learned jointly with the model weights?
- How does the bias interact with FP8 quantization of the gating logits?

### 2.3 Grouped Routing Efficiency
The n_group=8, topk_group=4 configuration was chosen to balance routing diversity with efficiency. Systematic exploration of:
- Asymmetric group sizes (not all groups need the same number of experts)
- Dynamic group selection (adjust topk_group based on input difficulty)
- Hardware-aware group assignment (map groups to physical GPU topology)

### 2.4 Expert Pruning and Distillation
For deployment on resource-constrained hardware, pruning experts while maintaining quality is valuable. Research directions include:
- Identifying which experts are rarely activated and can be pruned
- Distilling 256 experts into fewer, larger experts
- Dynamic expert loading from SSD/HBM based on routing predictions

## 3. FP8 Optimization

### 3.1 Block Size Selection
DeepSeek-V3 uses 128x128 blocks for FP8 quantization. Is this optimal?
- Smaller blocks (64x64) improve accuracy but increase scaling factor overhead
- Larger blocks (256x256) reduce overhead but increase quantization error
- Adaptive block sizes based on tensor statistics could be beneficial

### 3.2 Mixed Block-Size Quantization
Different layers and projections may benefit from different block sizes:
- Attention projections with larger dynamic range may need smaller blocks
- MoE expert weights with smaller range may tolerate larger blocks
- The gating network is particularly sensitive to quantization

### 3.3 FP8 Inference Optimization
Training with FP8 does not automatically mean inference should use FP8. Research into:
- INT4/INT8 weight-only quantization for decode (memory-bound)
- FP8 activations with INT4 weights (hybrid schemes)
- GPTQ/AWQ-style post-training quantization for MLA weights

## 4. Multi-Token Prediction

### 4.1 Speculative Decoding Integration
The MTP head predicts one additional token. Combining this with speculative decoding:
- Use MTP as a draft model for the main model
- Medusa-style parallel verification of MTP predictions
- Optimal acceptance/rejection thresholds for MTP-guided speculation

### 4.2 MTP for Prefill Optimization
During prefill, MTP could process multiple tokens in parallel if the predictions are accurate. This requires:
- Confidence calibration of MTP predictions
- Efficient tree-based attention for speculative prefill
- Analysis of MTP accuracy vs. position in sequence

## 5. Hardware-Specific Optimization

### 5.1 NVIDIA H100 / H200
- Leverage H100's FP8 tensor cores (2x peak of BF16)
- Utilize H200's expanded 141GB HBM3e for larger KV caches
- NVLink 4.0 for efficient expert-parallel communication

### 5.2 AMD MI300X
- ROCm FP8 support and achievable throughput
- HIP kernel adaptation of MLA fused kernels
- 192GB HBM3 capacity advantage for KV cache

### 5.3 Domestic Accelerators (Huawei Ascend, etc.)
- Adapting MLA/MoE kernels to non-CUDA architectures
- Utilizing Cube Units and Vector Units efficiently
- Communication topology awareness for expert parallelism

## 6. System-Level Optimization

### 6.1 Continuous Batching with MoE
Standard continuous batching assumes uniform per-token compute. MoE breaks this assumption because different tokens in a batch may route to different experts. Research into:
- Expert-aware batch scheduling
- Preemption strategies when expert queues are imbalanced
- Speculative routing prediction for batch formation

### 6.2 PagedAttention with Compressed KV
PagedAttention needs adaptation for MLA's compressed KV cache:
- Block size selection for 512-dim latents (vs. standard head_dim blocks)
- Memory allocation strategy with 32x smaller per-token footprint
- Impact on copy-on-write sharing efficiency

### 6.3 Disaggregated Prefill and Decode
Separate prefill (compute-bound) and decode (memory-bound) onto different GPU types:
- Prefill on high-compute GPUs (H100)
- Decode on high-memory GPUs (H200, or CPU+GPU hybrid)
- KV cache transfer protocol between prefill and decode nodes

## 7. Metrics and Benchmarking

### 7.1 MoE-Aware MFU
Standard MFU treats all parameters equally. For MoE, only 8/256 experts are active per token. MoE-aware MFU should account for:
- Active parameter count (not total)
- Expert dispatch overhead (not just FLOP)
- Communication overhead in distributed settings

### 7.2 Roofline Model Extension
The standard roofline model needs extension for MoE:
- Multiple operational intensity regimes within a single layer
- Communication bandwidth as a third axis
- Expert load imbalance as an efficiency multiplier

## References

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-V2 (MLA introduction): https://arxiv.org/abs/2405.04434
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- FlashAttention-3: https://arxiv.org/abs/2407.08691
