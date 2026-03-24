# H100 Test Plan: DeepSeek-V3 FlashMLA + DeepGEMM

## Objective

Validate correctness, performance, and stability of the FlashMLA + DeepGEMM
kernel combination for DeepSeek-V3 on NVIDIA H100 (Hopper) GPUs.

## Test Matrix

### Phase 1: Kernel Correctness

| Test | Expected | Tolerance | File |
|------|----------|-----------|------|
| FlashMLA vs eager attention | Identical (BF16) | atol=1e-3 | `h100_test_flashmla_kernels.py` |
| DeepGEMM FP8 vs BF16 GEMM | Close | rel_err < 15% | `h100_test_deepgemm_kernels.py` |
| FP8 round-trip quantisation | Bounded error | rel_err < 20% | `h100_test_fp8_edge_cases.py` |
| Full model BF16 vs FP32 | Close | rel_err < 10% | `h100_test_precision_chain.py` |

### Phase 2: Performance Benchmarks

| Metric | Baseline | Target | File |
|--------|----------|--------|------|
| Attention prefill (S=2048) | 150 TFLOPS (FA2) | 250+ TFLOPS (FlashMLA) | `h100_bench.py` |
| Attention decode (S=1) | 30 TFLOPS (FA2) | 60+ TFLOPS (FlashMLA) | `h100_bench.py` |
| MoE dispatch (T=4096) | 100 TFLOPS (seq) | 400+ TFLOPS (DeepGEMM) | `h100_bench_3way.py` |
| Kernel launch overhead | 50us (eager) | < 20us (CUDA graph) | `h100_test_launch_overhead.py` |

### Phase 3: Stability

| Test | Duration | Criterion | File |
|------|----------|-----------|------|
| Sustained attention | 30 seconds | Output deviation < 1e-4 | `h100_test_thermal.py` |
| Sustained MoE | 30 seconds | No OOM or crash | `h100_test_thermal.py` |
| Memory leak detection | 100 iterations | < 100KB growth | `h100_test_memory.py` |
| CUDA graph replay | 100 replays | Bit-exact output | `h100_test_cuda_graph.py` |
| Determinism | 5 runs | Bit-exact output | `h100_test_determinism.py` |

### Phase 4: Edge Cases

| Test | Scenario | File |
|------|----------|------|
| FP8 all zeros | Zero tensor quantisation | `h100_test_fp8_edge_cases.py` |
| FP8 very large values | Overflow handling | `h100_test_fp8_edge_cases.py` |
| FP8 sparse matrices | Mostly-zero blocks | `h100_test_fp8_edge_cases.py` |
| Empty expert groups | No tokens routed to an expert | `h100_test_sparse_patterns.py` |
| Single-token decode | seq_len=1 with KV cache | `h100_test_flashmla_kernels.py` |
| Multi-GPU tensor movement | Cross-device copies | `h100_test_multi_gpu.py` |

### Phase 5: TMA-Specific

| Test | Verification | File |
|------|-------------|------|
| SM90 capability | compute_capability >= 9.0 | `h100_test_tma.py` |
| Shared memory capacity | >= 48KB per block | `h100_test_tma.py` |
| TMA-backed GEMM correctness | vs reference | `h100_test_tma.py` |
| TMA throughput | > 400 TFLOPS for 4K x 7K x 2K | `h100_test_tma.py` |

## Hardware Requirements

- **Minimum**: 1x H100 SXM5 80GB
- **Recommended**: 8x H100 SXM5 80GB (for multi-GPU tests)
- **Software**: CUDA 12.0+, cuDNN 8.9+, FlashMLA, DeepGEMM

## Execution Order

1. Run CPU tests first to validate logic (`pytest tests/ -k "not h100"`)
2. Run H100 correctness tests (`h100_test_determinism`, `h100_test_deepgemm_kernels`, `h100_test_flashmla_kernels`)
3. Run precision tests (`h100_test_precision_chain`, `h100_test_fp8_edge_cases`)
4. Run performance benchmarks (`h100_bench`, `h100_bench_3way`)
5. Run stability tests (`h100_test_thermal`, `h100_test_memory`)
6. Run multi-GPU tests (`h100_test_multi_gpu`)

## Success Criteria

- All CPU tests pass
- All H100 correctness tests pass with specified tolerances
- FlashMLA achieves >= 1.5x speedup over SDPA
- DeepGEMM achieves >= 2x speedup over sequential BF16
- No memory leaks after 100 iterations
- Output deterministic across 5 consecutive runs
- Stable output after 30 seconds sustained load
