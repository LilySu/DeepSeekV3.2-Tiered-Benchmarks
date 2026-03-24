# DeepSeek-V3 Cross-Platform Feasibility Analysis

## Target Platforms

### NVIDIA GPUs

| GPU | Architecture | FP8 Support | FlashMLA | DeepGEMM | Status |
|-----|-------------|-------------|----------|----------|--------|
| H100 SXM5 | Hopper SM90 | Native TMA | Yes | Yes | Primary |
| H100 PCIe | Hopper SM90 | Native TMA | Yes | Yes | Supported |
| H200 | Hopper SM90 | Native TMA | Yes | Yes | Supported |
| B100/B200 | Blackwell SM100 | Enhanced | Planned | Planned | Future |
| A100 | Ampere SM80 | No | No | No | FlashInfer only |
| L40S | Ada SM89 | FP8 (no TMA) | No | No | Triton only |

### Key Constraint: Hopper TMA

FlashMLA and DeepGEMM both rely on Hopper's Tensor Memory Accelerator (TMA)
for asynchronous memory copies. This means:
- **H100/H200:** Full kernel stack available
- **A100:** Must fall back to FlashInfer + Triton grouped GEMM
- **Consumer GPUs:** PyTorch eager only

### Memory Requirements

| Configuration | Precision | Memory per GPU | Min GPUs | Interconnect |
|--------------|-----------|----------------|----------|-------------|
| Full model | BF16 | ~168 GB | 8x H100 | NVLink |
| Full model | FP8 | ~84 GB | 2x H100 | NVLink |
| 4-bit GPTQ | INT4 | ~42 GB | 1x H100 | N/A |
| Expert parallel | FP8 | ~42 GB/node | 8 nodes | InfiniBand |

### Domestic Chip Considerations

See `research/domestic_chips/` for detailed analysis of:
- Huawei Ascend 910B: CANN operator compatibility
- Cambricon MLU370: Custom kernel requirements
- Biren BR100: PCIe-based inference feasibility

## Parallelism Strategies

### Inference

1. **Tensor Parallelism (TP):** Split attention heads and expert weights across GPUs
   - MLA: Split Q/K/V heads (128 heads / N GPUs)
   - MoE: All-to-all expert dispatch across TP ranks

2. **Expert Parallelism (EP):** Distribute experts across nodes
   - 256 experts / 8 nodes = 32 experts per node
   - All-to-all communication for token dispatch
   - n_group=8 limits each token to 4 groups (nodes)

3. **Pipeline Parallelism (PP):** Rarely used for inference
   - 61 layers can be split across pipeline stages
   - Increases latency, useful only for memory-constrained setups

### Training

- **3D parallelism:** TP + EP + PP
- **DeepSeek-V3 training used:** 2048 H800 GPUs
- **Communication:** All-reduce for TP, all-to-all for EP, point-to-point for PP
