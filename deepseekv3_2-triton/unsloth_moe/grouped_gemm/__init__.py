# Grouped GEMM for DeepSeek-V3 MoE expert dispatch.
#
# Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
# MoE config: 256 routed experts, hidden=7168, intermediate=2048, top_k=8
#
# The grouped GEMM replaces the per-expert loop with a single batched operation.
