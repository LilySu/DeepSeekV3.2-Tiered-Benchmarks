# DeepSeek-V3 Implementation References

## Primary Papers

### DeepSeek-V3 Technical Report
- **Title:** DeepSeek-V3 Technical Report
- **arXiv:** 2412.19437
- **Link:** https://arxiv.org/abs/2412.19437
- **Summary:** Describes the full DeepSeek-V3 671B architecture including Multi-head Latent Attention (MLA), DeepSeekMoE with auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), and FP8 mixed-precision training. Reports training on 14.8T tokens using a cluster of 2048 H800 GPUs with only 2.788M GPU hours.

### DeepSeek-V2
- **Title:** DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- **arXiv:** 2405.04434
- **Link:** https://arxiv.org/abs/2405.04434
- **Summary:** Introduces Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture. MLA compresses KV cache via low-rank projection, reducing memory by ~32x while maintaining quality. DeepSeekMoE uses fine-grained expert segmentation with shared experts.

### DeepSeekMoE
- **Title:** DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
- **arXiv:** 2401.06066
- **Link:** https://arxiv.org/abs/2401.06066
- **Summary:** Foundational MoE design with fine-grained experts (more experts, smaller each) and shared experts that process all tokens. Demonstrates superior expert specialization compared to coarse-grained MoE.

## Kernel and System References

### DeepGEMM
- **Repository:** https://github.com/deepseek-ai/DeepGEMM
- **Summary:** DeepSeek's open-source FP8 GEMM library optimized for NVIDIA Hopper GPUs. Implements block-wise FP8 quantization with per-block scaling, achieving near-peak utilization of H100 FP8 tensor cores. Supports the 128x128 block quantization scheme used in DeepSeek-V3.

### FlashAttention-3
- **Title:** FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
- **arXiv:** 2407.08691
- **Link:** https://arxiv.org/abs/2407.08691
- **Summary:** Hardware-aware attention implementation for Hopper GPUs. Leverages TMA (Tensor Memory Accelerator) and warp specialization. Relevant for MLA attention kernel optimization, though MLA requires custom modifications to handle compressed KV inputs.

### FlashAttention-2
- **Title:** FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **arXiv:** 2307.08691
- **Link:** https://arxiv.org/abs/2307.08691
- **Summary:** Foundational IO-aware attention algorithm. Used as baseline for MLA attention benchmarks.

### Megablocks
- **Title:** MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- **arXiv:** 2211.15841
- **Link:** https://arxiv.org/abs/2211.15841
- **Summary:** Efficient MoE training framework using block-sparse operations. Relevant for understanding expert dispatch optimization strategies.

## Quantization References

### FP8 Training
- **Title:** FP8 Formats for Deep Learning
- **arXiv:** 2209.05433
- **Link:** https://arxiv.org/abs/2209.05433
- **Summary:** Defines FP8 formats (e4m3 and e5m2) and demonstrates their viability for neural network training. DeepSeek-V3 uses e4m3 for both forward and backward passes with block-wise scaling.

### GPTQ
- **Title:** GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers
- **arXiv:** 2210.17323
- **Link:** https://arxiv.org/abs/2210.17323
- **Summary:** Post-training quantization method relevant for INT4/INT8 inference deployment of DeepSeek-V3.

### AWQ
- **Title:** AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- **arXiv:** 2306.00978
- **Link:** https://arxiv.org/abs/2306.00978
- **Summary:** Weight quantization aware of activation distributions. Applicable to DeepSeek-V3's MLA projections and MoE expert weights.

## RoPE and Context Extension

### YaRN
- **Title:** YaRN: Efficient Context Window Extension of Large Language Models
- **arXiv:** 2309.00071
- **Link:** https://arxiv.org/abs/2309.00071
- **Summary:** Context window extension method used by DeepSeek-V3 to extend from 4K to 128K+ context length. Applied only to the qk_rope_head_dim=64 portion of the attention heads.

### RoPE
- **Title:** RoFormer: Enhanced Transformer with Rotary Position Embedding
- **arXiv:** 2104.09864
- **Link:** https://arxiv.org/abs/2104.09864
- **Summary:** Original Rotary Position Embedding paper. DeepSeek-V3 applies RoPE to a 64-dimensional subset of Q and K (the "rope" portion), with the remaining 128 dimensions being "nope" (no positional encoding).

## Inference Systems

### vLLM
- **Repository:** https://github.com/vllm-project/vllm
- **Summary:** High-throughput LLM serving with PagedAttention. Supports DeepSeek-V3 with adaptations for MLA's compressed KV cache.

### SGLang
- **Repository:** https://github.com/sgl-project/sglang
- **Summary:** Structured generation framework with RadixAttention. Efficient prefix caching and batch scheduling for MoE models.

### TensorRT-LLM
- **Repository:** https://github.com/NVIDIA/TensorRT-LLM
- **Summary:** NVIDIA's optimized inference runtime. FP8 support for H100 deployment of DeepSeek-V3.

## Code Repositories

### DeepSeek-V3 Official
- **Repository:** https://github.com/deepseek-ai/DeepSeek-V3
- **Weights:** https://huggingface.co/deepseek-ai/DeepSeek-V3
- **Summary:** Official model code and weights. Reference implementation for all architectural components.

### Transformers (HuggingFace)
- **Repository:** https://github.com/huggingface/transformers
- **Summary:** Includes DeepSeek-V3 model implementation in the `modeling_deepseek.py` module. Reference for standard PyTorch implementation of MLA and MoE.

## Domestic Chip References

### Huawei Ascend 910B
- **Documentation:** https://www.hiascend.com/
- **Summary:** Huawei's AI accelerator used for DeepSeek-V3 inference in China. Requires CANN toolkit and custom kernel development for MLA and MoE operations.

### Cambricon MLU370/MLU590
- **Summary:** Chinese AI accelerator with BANG language for custom kernels. Relevant for domestic deployment of DeepSeek-V3.

## Benchmarking Methodology

### MLPerf Inference
- **Repository:** https://github.com/mlcommons/inference
- **Summary:** Industry-standard inference benchmarks. Provides methodology for measuring tokens/second, time-to-first-token, and latency percentiles.

### Roofline Model
- **Reference:** Williams, Waterman, Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM, 2009.
- **Summary:** Performance analysis methodology used in our MFU ceiling analysis. Extended for MoE models with multiple operational intensity regimes.
