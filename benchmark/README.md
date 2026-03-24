# DeepSeek-V3 671B Benchmark Suite

Comprehensive benchmarking toolkit for the DeepSeek-V3 architecture (arXiv: 2412.19437).

## Architecture Summary

DeepSeek-V3 is a 671B-parameter Mixture of Experts (MoE) language model with three key innovations:

- **Multi-head Latent Attention (MLA):** Compresses KV cache via a low-rank latent (kv_lora_rank=512), reducing per-token KV cache by ~32x compared to standard MHA.
- **Grouped MoE Routing:** 256 routed experts with grouped top-K selection (8 groups, top-4 groups, then top-2 per group = 8 total active experts per token).
- **Multi-Token Prediction (MTP):** 1 additional prediction layer for speculative decoding.

### Key Model Dimensions

| Parameter | Value |
|-----------|-------|
| hidden_size | 7168 |
| num_layers | 61 (3 dense + 58 MoE) |
| num_heads | 128 |
| kv_lora_rank | 512 |
| qk_rope_head_dim | 64 |
| qk_nope_head_dim | 128 |
| v_head_dim | 128 |
| n_routed_experts | 256 |
| num_experts_per_tok | 8 |
| n_group | 8 |
| topk_group | 4 |
| moe_intermediate_size | 2048 |
| vocab_size | 129280 |
| mtp_layers | 1 |
| FP8 format | e4m3 (128x128 blocks) |

**Note:** DeepSeek-V3 does NOT use Differential State Attention (DSA). The DSA-related benchmarks from the GLM5 suite are replaced with MoE routing analysis.

## Benchmark Suites

### Micro-benchmarks (`triple_report/bench_micro.py`)
Individual kernel/operation benchmarks:
- GEMM shapes for all MLA and MoE projections
- Softmax at various sequence lengths
- RoPE with YaRN scaling
- SwiGLU activation
- RMSNorm

### Component Benchmarks (`triple_report/bench_component.py`)
Full layer benchmarks:
- MLA layer (compression + attention + output)
- MoE FFN layer (gating + routing + expert compute + combine)
- Dense FFN layer (SwiGLU)

### End-to-End Benchmarks (`triple_report/bench_e2e.py`)
Full model forward pass:
- Reduced-scale model simulation for single-GPU testing
- Profiled breakdown by layer type
- Memory tracking

### MoE Sweep (`moe_sweep/bench_moe.py`)
Parameter space exploration:
- Expert count: 64, 128, 256
- Top-K: 2, 4, 8, 16
- Group count: 1 (flat), 4, 8, 16
- Intermediate size: 1024, 2048, 4096
- Load balance analysis

### FP8 Pareto Frontier (`fp8_pareto/bench_fp8.py`)
Precision-performance trade-off:
- e4m3 vs e5m2 format
- Block sizes: 64x64, 128x128, 256x256, per-tensor
- Quantization error analysis
- Throughput measurement

### MFU Ceiling Analysis (`mfu_ceiling/bench_mfu.py`)
Theoretical performance analysis:
- Per-component MFU ceiling
- Full-model MFU estimation
- Roofline model analysis

## Usage

### Run all benchmarks
```bash
python -m benchmark.run_all --suite all
```

### Run specific suite
```bash
python -m benchmark.run_all --suite micro component
```

### Quick mode (for CI)
```bash
python -m benchmark.run_all --suite all --quick
```

### Component-only benchmarks
```bash
python -m benchmark.run_component_bench --batch-sizes 1 4 --seq-lengths 512 2048
```

### Debug before benchmarking
```bash
python -m benchmark.debug_imports
python -m benchmark.debug_all_kernels
python -m benchmark.debug_single_layer --layer-type moe
```

### Extract summary from results
```bash
python -m benchmark.extract_summary results/ --recursive
```

### Generate research report
```bash
python -m benchmark.generate_research_report_from_benchmark_results --results-dir results/
```

## Directory Structure

```
benchmark/
  __init__.py
  run_all.py                  # Master benchmark runner
  run_component_bench.py      # Quick component benchmarks
  extract_summary.py          # Result summary extraction
  generate_research_report... # Research report generation
  print_all_benchmark...      # Console result display
  debug_*.py                  # Debug and validation scripts
  fix_moe_routing.py          # MoE routing debug (replaces DSA indexer)
  shared/
    config.py                 # DEEPSEEK_V3_CONFIG, BenchConfig, BenchResult
    metrics.py                # FLOP calculations, MFU, bandwidth
    report.py                 # JSON/Markdown report generation
    timer.py                  # CUDA timer with bootstrap CI
  fp8_pareto/
    bench_fp8.py              # FP8 Pareto frontier
    precision_experiment.py   # Precision comparison
  mfu_ceiling/
    bench_mfu.py              # MFU ceiling analysis
  moe_sweep/
    bench_moe.py              # MoE parameter sweep
  triple_report/
    bench_micro.py            # Micro-benchmarks
    bench_component.py        # Component benchmarks
    bench_e2e.py              # End-to-end benchmarks
```

## Hardware Target

Primary target: NVIDIA H100 SXM5 80GB
- BF16 peak: 989.5 TFLOPS
- FP8 peak: 1979 TFLOPS
- HBM3 bandwidth: 3.35 TB/s

## References

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepGEMM (FP8 GEMM library): https://github.com/deepseek-ai/DeepGEMM
- DeepSeek-V3 model weights: https://huggingface.co/deepseek-ai/DeepSeek-V3
