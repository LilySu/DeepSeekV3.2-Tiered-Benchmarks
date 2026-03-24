# Nebius Cloud H100 Setup Guide for DeepSeek-V3

## Overview

Nebius AI Cloud provides NVIDIA H100 SXM5 instances suitable for running DeepSeek-V3 kernel benchmarks. This guide covers provisioning, environment setup, and validation for the FlashMLA+DeepGEMM kernel variant.

## Instance Selection

### Recommended Configuration

| Resource | Specification | Notes |
|----------|--------------|-------|
| GPU | 8x H100 SXM5 80GB | Required for full model; 1x sufficient for kernel tests |
| CPU | 96 vCPUs (AMD EPYC) | Sufficient for data loading |
| RAM | 1024 GB | Needed for weight loading before GPU transfer |
| Storage | 2 TB NVMe SSD | For model weights + datasets |
| Network | 400 Gbps InfiniBand | Required for multi-node only |

### Nebius Instance Types

For kernel development and testing:
- **gpu-h100-a** (1x H100): Kernel unit tests, single-GPU benchmarks
- **gpu-h100-b** (8x H100): Multi-GPU TP tests, full model benchmarks

For benchmark runs:
- **gpu-h100-b** (8x H100): Recommended for head-to-head comparisons

### Cost Estimates (as of 2025)

| Instance | GPU Count | Approx Cost/hr | Notes |
|----------|-----------|----------------|-------|
| gpu-h100-a | 1 | ~$3.50 | Kernel development |
| gpu-h100-b | 8 | ~$28.00 | Full benchmarks |

## Provisioning Steps

### 1. Create Nebius Account and Project

```bash
# Install Nebius CLI
curl -sSL https://storage.ai.nebius.cloud/nebius/install.sh | bash

# Configure authentication
nebius init
nebius iam create-profile --name deepseek-v3-bench
```

### 2. Create Compute Instance

```bash
# Create 1x H100 instance for kernel development
nebius compute instance create \
  --name deepseek-v3-dev \
  --zone eu-north1-c \
  --platform gpu-h100-a \
  --cores 24 \
  --memory 256G \
  --gpus 1 \
  --boot-disk-size 500G \
  --boot-disk-type network-ssd \
  --image-id <ubuntu-22.04-cuda-12.4> \
  --ssh-key ~/.ssh/id_rsa.pub

# Create 8x H100 instance for full benchmarks
nebius compute instance create \
  --name deepseek-v3-bench \
  --zone eu-north1-c \
  --platform gpu-h100-b \
  --cores 96 \
  --memory 1024G \
  --gpus 8 \
  --boot-disk-size 2000G \
  --boot-disk-type network-ssd \
  --image-id <ubuntu-22.04-cuda-12.4> \
  --ssh-key ~/.ssh/id_rsa.pub
```

### 3. Connect to Instance

```bash
# Get instance IP
nebius compute instance get --name deepseek-v3-dev --format json | jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address'

# SSH connect
ssh -i ~/.ssh/id_rsa ubuntu@<instance-ip>
```

## Environment Setup

### 4. System Validation

```bash
# Verify GPU availability
nvidia-smi
# Should show H100 SXM5 80GB with CUDA 12.x

# Check GPU topology (for multi-GPU)
nvidia-smi topo -m

# Verify Hopper architecture
python3 -c "import torch; print(torch.cuda.get_device_properties(0))"
# Should show: major=9, minor=0 (SM90)

# Check NVLink bandwidth (multi-GPU only)
nvidia-smi nvlink -s
```

### 5. CUDA Toolkit

```bash
# Nebius images typically include CUDA. Verify version:
nvcc --version
# Need CUDA 12.4+ for DeepGEMM

# If CUDA needs updating:
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

### 6. Python Environment

```bash
# Create virtual environment
python3 -m venv ~/deepseek-v3-env
source ~/deepseek-v3-env/bin/activate

# Core dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install triton>=3.0.0
pip install numpy scipy matplotlib pandas

# Kernel libraries
pip install flashinfer  # FlashInfer FA3 backend

# FlashMLA (build from source)
git clone https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA && pip install -e . && cd ..

# DeepGEMM (build from source)
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && pip install -e . && cd ..

# Verification
python3 -c "import torch; import triton; print(f'PyTorch {torch.__version__}, Triton {triton.__version__}')"
python3 -c "import flashinfer; print(f'FlashInfer OK')"
python3 -c "import flash_mla; print(f'FlashMLA OK')"
python3 -c "import deep_gemm; print(f'DeepGEMM OK')"
```

### 7. Clone Project

```bash
git clone <your-repo-url> ~/deepseekv3_2
cd ~/deepseekv3_2
pip install -e .  # If setup.py exists
```

## Running Benchmarks

### 8. Quick Validation

```bash
cd ~/deepseekv3_2

# CPU tests (no GPU required)
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/ -k "not h100" -v

# Single GPU tests
python3 -m pytest deepseekv3_2-kernels-flashmla-deepgemm/tests/ -k "h100" -v

# Head-to-head comparison
python3 benchmark_head_to_head.py --quick --output-dir results/nebius
```

### 9. Full Benchmark Suite

```bash
# Full benchmark (all suites)
python3 -m benchmark.run_all --suite all --output-dir results/nebius_full

# Component-specific
python3 -m benchmark.run_all --suite mla moe --output-dir results/nebius_components

# Multi-GPU tests (8x H100)
torchrun --nproc_per_node=8 \
  -m deepseekv3_2-kernels-flashmla-deepgemm.tests.h100_test_multi_gpu
```

### 10. Profiling

```bash
# NSight Compute profiling (single kernel)
ncu --set full \
    --target-processes all \
    python3 -m benchmark.debug_single_layer --mode mla

# NSight Systems profiling (full trace)
nsys profile --trace=cuda,nvtx \
    python3 -m benchmark.run_all --suite micro --quick
```

## Nebius-Specific Notes

### Storage

Nebius NVMe SSDs provide ~3GB/s read throughput. For loading 671B model weights:
- FP8 weights: ~335GB, loads in ~2 minutes from NVMe
- BF16 weights: ~670GB (may not fit on 500GB disk, use 2TB)

### Networking

For multi-node runs (if needed):
- Nebius InfiniBand provides 400Gbps per node
- Use `NCCL_IB_DISABLE=0` to enable InfiniBand for NCCL
- Set `NCCL_SOCKET_IFNAME=eth0` for the correct network interface

### Spot vs On-Demand

For long benchmark runs (>1 hour), consider on-demand instances. Spot instances are cheaper but can be preempted. Save checkpoints frequently.

### Instance Lifecycle

```bash
# Stop instance (preserves disk, stops billing for compute)
nebius compute instance stop --name deepseek-v3-bench

# Start instance
nebius compute instance start --name deepseek-v3-bench

# Delete instance (data lost)
nebius compute instance delete --name deepseek-v3-bench
```

## Troubleshooting

### Common Issues

1. **CUDA OOM on 1x H100**: Reduce batch size or use `--quick` mode
2. **DeepGEMM build failure**: Ensure CUDA 12.4+ and GCC 11+
3. **FlashMLA import error**: Build with `TORCH_CUDA_ARCH_LIST="9.0"` flag
4. **NCCL timeout**: Check that all GPUs are visible with `nvidia-smi`
5. **Slow NVMe**: Verify disk type is `network-ssd`, not `network-hdd`

### Verification Checklist

- [ ] `nvidia-smi` shows correct GPU count and type
- [ ] CUDA version >= 12.4
- [ ] PyTorch sees all GPUs: `torch.cuda.device_count()`
- [ ] SM90 confirmed: `torch.cuda.get_device_properties(0).major == 9`
- [ ] All kernel libraries import without error
- [ ] CPU tests pass: `pytest ... -k "not h100"`
- [ ] GPU tests pass: `pytest ... -k "h100"`
- [ ] Benchmark produces output files in results/
