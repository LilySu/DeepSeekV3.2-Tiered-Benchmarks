"""
DeepSeek-V3 671B Benchmark Suite
=================================

Comprehensive benchmarking toolkit for the DeepSeek-V3 architecture (arXiv: 2412.19437).
Covers MLA (Multi-head Latent Attention), MoE (Mixture of Experts with grouped routing),
MTP (Multi-Token Prediction), FP8 quantization, and end-to-end inference profiling.

Architecture highlights benchmarked here:
  - 61 layers (3 dense + 58 MoE)
  - hidden_size=7168, num_heads=128, kv_lora_rank=512
  - 256 routed experts, top-8 selection, 8 groups with top-4 group routing
  - FP8 training with 128x128 block-wise quantization (e4m3)
  - Multi-Token Prediction with 1 MTP layer
"""

__version__ = "0.1.0"
