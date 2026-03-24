# RunPod H100 Setup Guide for DeepSeek-V3

## Overview

RunPod provides on-demand and spot NVIDIA H100 GPU instances for running DeepSeek-V3 kernel benchmarks and tests. This guide covers pod creation, environment configuration, and benchmark execution for the FlashMLA+DeepGEMM kernel variant.

## Pod Selection

### Recommended Configurations

| Use Case | GPU | vCPU | RAM | Disk | Template |
|----------|-----|------|-----|------|----------|
| Kernel dev/test | 1x H100 SXM 80GB | 16 | 125 GB | 200 GB | PyTorch 2.4 |
| Full benchmarks | 1x H100 SXM 80GB | 32 | 251 GB | 500 GB | PyTorch 2.4 |
| Multi-GPU tests | 4x H100 SXM 80GB | 64 | 502 GB | 1000 GB | PyTorch 2.4 |

### Cost Estimates

| Configuration | On-Demand/hr | Spot/hr | Notes |
|--------------|-------------|---------|-------|
| 1x H100 SXM | ~$3.89 | ~$2.49 | Kernel testing |
| 2x H100 SXM | ~$7.78 | ~$4.98 | Basic multi-GPU |
| 4x H100 SXM | ~$15.56 | ~$9.96 | Full TP testing |
| 8x H100 SXM | ~$31.12 | ~$19.92 | Full model fit |

For kernel development, 1x H100 spot is the most cost-effective option.

## Pod Creation

### 1. Via RunPod Web UI

1. Go to [runpod.io](https://runpod.io) and log in
2. Click "Deploy" -> "GPU Pods"
3. Select **H100 SXM 80GB**
4. Choose template: **RunPod PyTorch 2.4** (includes CUDA 12.4)
5. Set volume size: 200+ GB
6. Deploy

### 2. Via RunPod CLI

```bash
# Install RunPod CLI
pip install runpodctl

# Configure API key
runpodctl config --apikey $RUNPOD_API_KEY

# Create pod
runpodctl create pods \
  --name deepseek-v3-bench \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --gpuCount 1 \
  --volumeSize 200 \
  --containerDiskSize 50 \
  --templateId runpod-pytorch-2.4 \
  --env "JUPYTER_PASSWORD=<your-password>"
```

### 3. Via RunPod API

```python
import runpod

runpod.api_key = "YOUR_API_KEY"

pod = runpod.create_pod(
    name="deepseek-v3-bench",
    image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA H100 80GB HBM3",
    gpu_count=1,
    volume_in_gb=200,
    container_disk_in_gb=50,
    ports="8888/http,22/tcp",
    env={
        "JUPYTER_PASSWORD": "your-password",
        "PUBLIC_KEY": open(os.path.expanduser("~/.ssh/id_rsa.pub")).read(),
    },
)
print(f"Pod ID: {pod['id']}")
```

## Environment Setup

### 4. Connect to Pod

```bash
# Via SSH (if SSH port exposed)
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_rsa

# Via Web Terminal
# Open RunPod dashboard -> Click on pod -> "Connect" -> "Terminal"

# Via JupyterLab
# Open RunPod dashboard -> Click on pod -> "Connect" -> "Jupyter"
```

### 5. Verify GPU

```bash
nvidia-smi
# Expected: H100 SXM5 80GB, Driver 550+, CUDA 12.4+

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'SM version: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

### 6. Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Core framework (should already be present in RunPod template)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton>=3.0.0

# FlashInfer
pip install flashinfer

# FlashMLA (from source)
cd /workspace
git clone https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
TORCH_CUDA_ARCH_LIST="9.0" pip install -e .
cd /workspace

# DeepGEMM (from source)
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
pip install -e .
cd /workspace

# Test utilities
pip install pytest pytest-xdist matplotlib pandas scipy

# Verify all imports
python3 -c "
import torch, triton
print(f'torch {torch.__version__}, triton {triton.__version__}')
try:
    import flashinfer; print('flashinfer: OK')
except: print('flashinfer: NOT AVAILABLE')
try:
    import flash_mla; print('flash_mla: OK')
except: print('flash_mla: NOT AVAILABLE')
try:
    import deep_gemm; print('deep_gemm: OK')
except: print('deep_gemm: NOT AVAILABLE')
"
```

### 7. Clone and Setup Project

```bash
cd /workspace
git clone <your-repo-url> deepseekv3_2
cd deepseekv3_2

# Make all test scripts executable
chmod +x benchmark_head_to_head.py
chmod +x benchmark/run_all.py
```

## Running Tests

### 8. Quick Validation (5 minutes)

```bash
cd /workspace/deepseekv3_2

# CPU-only unit tests
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/ \
  -k "not h100 and not H100" -v --tb=short

# Quick GPU validation
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_determinism.py -v
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_deepgemm_kernels.py -v
```

### 9. Full H100 Test Suite (30 minutes)

```bash
# All H100 tests
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/ -k "h100" -v

# Individual categories
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_flashmla_kernels.py -v
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_fp8_edge_cases.py -v
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_memory.py -v
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_tma.py -v
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/h100_test_precision_chain.py -v
```

### 10. Benchmark Suite

```bash
# Quick benchmark
python3 benchmark_head_to_head.py --quick --output-dir /workspace/results

# Full benchmark
python3 -m benchmark.run_all --suite all --output-dir /workspace/results/full

# Specific suites
python3 -m benchmark.run_all --suite mla moe fp8 --output-dir /workspace/results/kernels
```

### 11. Save Results

```bash
# Results are in /workspace which persists across restarts
ls -la /workspace/results/

# Optionally upload to cloud storage
# pip install awscli
# aws s3 cp /workspace/results/ s3://your-bucket/deepseek-v3-results/ --recursive
```

## RunPod-Specific Tips

### Volume Persistence

- `/workspace` is on the persistent volume (survives pod restarts)
- `/root` and `/tmp` are on container disk (lost on pod stop)
- Always clone repos and save results to `/workspace`

### Saving Money

1. Use **spot** instances for benchmarks (50% cheaper, risk of preemption)
2. **Stop** pods when not in use (volume preserved, no GPU billing)
3. Use 1x H100 for development, only scale up for multi-GPU tests
4. Set up **auto-stop** timers to prevent accidentally running overnight

### Docker Template for Reproducibility

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04

RUN pip install flashinfer triton>=3.0.0 pytest matplotlib scipy

# FlashMLA
RUN git clone https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA && \
    cd /opt/FlashMLA && TORCH_CUDA_ARCH_LIST="9.0" pip install -e .

# DeepGEMM
RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM && \
    cd /opt/DeepGEMM && pip install -e .
```

### Network Ports

- 8888: JupyterLab (default)
- 22: SSH (must be explicitly exposed)
- 6006: TensorBoard (for profiling visualization)
- 8080: Custom web server (for viz app)

## Troubleshooting

### Common Issues

1. **"No H100 available"**: Try different data center region or switch to spot
2. **CUDA OOM**: Reduce batch size, use `--quick` mode
3. **FlashMLA build fails**: Ensure `TORCH_CUDA_ARCH_LIST="9.0"` is set
4. **DeepGEMM import error**: Check CUDA version >= 12.4 with `nvcc --version`
5. **Slow disk I/O**: Use container disk for temp files, volume for persistent data
6. **Pod preempted (spot)**: Save checkpoints frequently, use persistent volume
7. **Jupyter kernel dies**: Usually OOM -- check `nvidia-smi` for memory usage

### Verification Checklist

- [ ] `nvidia-smi` shows H100 SXM with 80GB memory
- [ ] CUDA >= 12.4: `nvcc --version`
- [ ] PyTorch sees GPU: `torch.cuda.is_available()`
- [ ] SM90 confirmed: `torch.cuda.get_device_properties(0).major == 9`
- [ ] FlashMLA imports: `import flash_mla`
- [ ] DeepGEMM imports: `import deep_gemm`
- [ ] CPU tests pass
- [ ] GPU tests pass
- [ ] Results directory created and populated
