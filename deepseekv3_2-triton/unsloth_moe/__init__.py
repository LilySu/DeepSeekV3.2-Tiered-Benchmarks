# Unsloth MoE -- Triton grouped GEMM kernels for Mixture of Experts.
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# DeepSeek-V3 MoE config:
#   n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8
#   moe_intermediate_size: 2048, hidden_size: 7168
#   n_group: 8, topk_group: 4, topk_method: "noaux_tc"
#   routed_scaling_factor: 2.5
#   first_k_dense_replace: 3 (layers 0-2 are dense, 3-60 are MoE)
#
# The grouped GEMM kernels replace the expert-by-expert loop in MoeExperts.forward()
# with a single batched GEMM that processes all experts simultaneously.
# This provides significant speedup (up to ~9x on H100/B200) by:
#   1. Avoiding kernel launch overhead for each expert
#   2. Better GPU utilization through larger matrix sizes
#   3. Optional fused permutation/unpermutation in the GEMM prologue/epilogue
