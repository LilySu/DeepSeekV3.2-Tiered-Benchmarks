# Test Suite: DeepSeek-V3 FlashMLA + DeepGEMM

## Test Organization

### CPU-Runnable Tests (`test_*.py`)
These tests run on CPU without any GPU requirement:

| File | Description |
|------|-------------|
| `test_equivalence.py` | PyTorch vs kernel equivalence checks |
| `test_autoregressive_decode.py` | KV cache decode correctness |
| `test_gradient_flow.py` | Gradient propagation through all components |
| `test_kv_cache.py` | KV cache allocation, update, reset |
| `test_moe_expert_dispatch.py` | MoE routing and expert dispatch |
| `test_state_dict_compat.py` | HuggingFace state dict compatibility |
| `test_fp8_layout.py` | FP8 tensor layout and quantisation |
| `test_group_routing.py` | Group-restricted top-k routing |
| `test_edge_cases.py` | Edge cases (empty batches, seq_len=1, etc.) |
| `test_mtp_prediction.py` | Multi-Token Prediction head |
| `test_yarn_rope.py` | YaRN RoPE correctness |

### H100-Specific Tests (`h100_*.py`)
These tests require an H100 (Hopper, SM90+) GPU:

| File | Description |
|------|-------------|
| `h100_bench.py` | Performance benchmarks |
| `h100_bench_3way.py` | PyTorch vs FlashMLA vs DeepGEMM comparison |
| `h100_test_cuda_graph.py` | CUDA graph capture and replay |
| `h100_test_deepgemm_kernels.py` | DeepGEMM FP8 kernel correctness |
| `h100_test_determinism.py` | Output determinism verification |
| `h100_test_flashmla_kernels.py` | FlashMLA kernel correctness |
| `h100_test_fp8_edge_cases.py` | FP8 edge cases on GPU |
| `h100_test_launch_overhead.py` | Kernel launch overhead measurement |
| `h100_test_memory.py` | GPU memory usage and leak detection |
| `h100_test_multi_gpu.py` | Multi-GPU tests |
| `h100_test_precision_chain.py` | End-to-end precision analysis |
| `h100_test_sparse_patterns.py` | MoE sparsity patterns |
| `h100_test_thermal.py` | Thermal stress tests |
| `h100_test_tma.py` | Hopper TMA tests for DeepGEMM |

## Running Tests

### Quick Start

```bash
# Run all CPU tests
pytest tests/ -k "not h100" -v

# Run all tests (requires H100)
pytest tests/ -v

# Run specific test file
pytest tests/test_equivalence.py -v

# Run specific test class
pytest tests/test_moe_expert_dispatch.py::TestMoERouterBasics -v

# Run with the test runner
python tests/run_all.py
python tests/run_all.py --h100
python tests/run_all.py --all
```

### Test Configuration

Tests use a **small model configuration** (256 hidden, 4 layers, 16 experts)
for CPU tests to keep execution fast. The full 671B configuration is only
used for parameter counting and config validation.

### Dependencies

```
pytest >= 7.0
torch >= 2.1
```

Optional (for H100 tests):
```
flash-mla
deep-gemm
triton >= 2.2
```

## Writing New Tests

1. CPU tests go in `test_*.py` files
2. H100 tests go in `h100_*.py` files
3. Use fixtures from `conftest.py` for common setup
4. Mark GPU tests with `@requires_hopper` or `@requires_cuda`
5. All tests should be deterministic (set `torch.manual_seed`)
