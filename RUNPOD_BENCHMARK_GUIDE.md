# RunPod Benchmark Guide for DeepSeek-V3

## Quick Start

### 1. Create a RunPod Instance
- Template: `nvcr.io/nvidia/pytorch:24.07-py3`
- GPU: H100 SXM5 80GB (pod or on-demand)
- Disk: 200GB (for model weights)

### 2. Setup

```bash
# Clone repo
git clone <repo-url> /workspace/deepseekv3_2
cd /workspace/deepseekv3_2

# Install dependencies
pip install safetensors tokenizers triton>=3.0.0 einops ninja

# Install FlashMLA
pip install flash-attn --no-build-isolation

# Install FlashInfer
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/

# Install DeepGEMM
git clone https://github.com/deepseek-ai/DeepGEMM.git /tmp/deepgemm
cd /tmp/deepgemm && pip install -e . && cd /workspace/deepseekv3_2
```

### 3. Download Model Weights (optional, for full-model benchmarks)

```bash
# FP8 weights (~150GB)
huggingface-cli download deepseek-ai/DeepSeek-V3 --local-dir /workspace/models/deepseek-v3

# Or BF16 weights (~1.3TB, needs multi-GPU)
# huggingface-cli download deepseek-ai/DeepSeek-V3-Base --local-dir /workspace/models/deepseek-v3-base
```

### 4. Run Benchmarks

```bash
# Smoke test (no model weights needed)
python deepseekv3_2-raw-decoupled-from-hf/test_standalone.py

# Component benchmarks (no model weights needed)
python benchmark/run_component_bench.py

# Full benchmark suite
python benchmark/run_all.py

# Head-to-head kernel comparison
python benchmark_head_to_head.py
```

## Benchmark Configurations

### Quick Smoke Test (~5 min)
```bash
python benchmark/moe_sweep/bench_moe.py --quick
```

### Component Benchmark (~30 min)
```bash
python benchmark/triple_report/bench_component.py
python benchmark/triple_report/bench_micro.py
```

### Full Suite (~2 hours)
```bash
python benchmark/run_all.py --full
```

### FP8 Pareto Analysis (~1 hour)
```bash
python benchmark/fp8_pareto/bench_fp8.py
```

## Expected Results (H100 SXM5)

| Component | Kernel | Throughput | MFU |
|-----------|--------|------------|-----|
| MLA Prefill (S=2048) | FlashMLA | ~700 TFLOPS | ~71% |
| MLA Decode (B=32) | FlashMLA | ~660 TFLOPS | ~67% |
| MoE GEMM (FP8) | DeepGEMM | ~1550 TFLOPS | ~78% |
| MoE GEMM (BF16) | Triton | ~700 TFLOPS | ~71% |
| RMSNorm | Triton | ~2.5 TB/s | ~75% HBM |
| SwiGLU | Triton | ~2.2 TB/s | ~66% HBM |

## Cost Estimates

| Configuration | RunPod Rate | Time | Total |
|--------------|-------------|------|-------|
| 1x H100 SXM5 | ~$2.50/hr | 2 hrs | ~$5 |
| 8x H100 SXM5 | ~$20/hr | 2 hrs | ~$40 |

## Nebius Setup

For Nebius H100 instances, see:
```bash
# Similar setup, different cloud provider
# SSH into instance and follow same steps
```
