# DeepSeek-V3 FlashInfer Kernel Tests

## Test Structure

### CPU Tests (no GPU required)

- **test_equivalence.py** - Verifies component outputs match expected behavior:
  MoE router, FP8 utilities, RMSNorm, MLA attention, full model, YaRN RoPE
- **test_components.py** - Component-level tests:
  KV cache, MoE dispatch, FP8 layout, group routing, autoregressive decode,
  gradient flow, edge cases, YaRN RoPE, DSA stub

### H100 GPU Tests (SM90 required)

| File | Category | What it Tests |
|------|----------|---------------|
| h100_bench.py | Benchmark | Per-component timing with ncu/nsys support |
| h100_bench_3way.py | Benchmark | PyTorch vs Triton vs FlashInfer comparison |
| h100_test_cuda_graph.py | Cat 1 | Graph capture, KV cache update, speedup |
| h100_test_tma.py | Cat 2 | TMA bandwidth for FlashInfer and DeepGEMM |
| h100_test_memory.py | Cat 3 | Peak memory, KV scaling, leak detection |
| h100_test_fp8_edge_cases.py | Cat 4 | Overflow, zeros, subnormals, KV scales |
| h100_test_multi_gpu.py | Cat 5 | NCCL bandwidth, TP equivalence |
| h100_test_launch_overhead.py | Cat 6 | Kernel launch cost, graph vs eager |
| h100_test_determinism.py | Cat 7 | topk, full decode, MoE routing |
| h100_test_sparse_patterns.py | Cat 8 | MoE utilization, group balance, stability |
| h100_test_precision_chain.py | Cat 9 | Chained FP8 roundtrips, full pipeline |
| h100_test_thermal.py | Cat 10 | Sustained GEMM stability, clock frequency |
| h100_test_flashinfer_kernels.py | Kernel | FA3 dense decode, paged attention |
| h100_test_deepgemm_kernels.py | Kernel | Grouped GEMM, FP8 activation |

## Running Tests

```bash
# CPU tests only
python3 -m deepseekv3_2-kernels-flashinfer.tests.run_all

# Include H100 GPU tests
python3 -m deepseekv3_2-kernels-flashinfer.tests.run_all --h100

# Individual test files
python3 -m deepseekv3_2-kernels-flashinfer.tests.test_components
python3 -m deepseekv3_2-kernels-flashinfer.tests.test_equivalence

# H100 benchmarks
python3 -m deepseekv3_2-kernels-flashinfer.tests.h100_bench --mode bench
python3 -m deepseekv3_2-kernels-flashinfer.tests.h100_bench_3way --full-dims

# Multi-GPU tests (requires torchrun)
torchrun --nproc_per_node=2 -m deepseekv3_2-kernels-flashinfer.tests.h100_test_multi_gpu
```

## Key Differences from GLM-5 Tests

1. **No DSA tests** - DeepSeek-V3 does not use Dynamic Sparse Attention
2. **Group routing tests** - Tests n_group=8, topk_group=4 hierarchical routing
3. **YaRN RoPE tests** - Verifies frequency-dependent interpolation
4. **MoE sparsity tests** - Expert utilization, group balance (replaces DSA sparse patterns)
5. **128 attention heads** - Tests scale to DeepSeek-V3's 128-head MLA
6. **61 layers** - Precision chain tests use 61 iterations (not 78)
7. **MTP tests** - Multi-Token Prediction layer verification

## Dependencies

- **Always**: torch, triton
- **FlashInfer tests**: `pip install flashinfer` (CUDA 12.0+)
- **DeepGEMM tests**: `pip install deep-gemm` (CUDA 12.8+, SM90)
- **Multi-GPU tests**: torchrun, NCCL

## Paper Reference

DeepSeek-V3 (arXiv 2412.19437)
