# DeepSeek-V3 Precision Tracking

## Model Precision Architecture

DeepSeek-V3 (arXiv 2412.19437) supports multiple precision configurations:

### Training Precision
- **Primary:** BF16 mixed precision
- **Router:** FP32 (sigmoid, group scoring, normalization)
- **RMSNorm:** FP32 accumulation with BF16 output
- **RoPE:** FP32 trigonometric computation

### Inference Precision Options

#### BF16 Inference
| Component | Weight Precision | Compute Precision | Accumulation |
|-----------|-----------------|-------------------|--------------|
| Embeddings | BF16 | BF16 | N/A |
| Q/K/V projections | BF16 | BF16 | FP32 |
| Attention scores | BF16 | BF16 | FP32 softmax |
| RMSNorm | BF16 weights | FP32 | FP32 |
| MoE Router | FP32 | FP32 | FP32 |
| Expert gate_up_proj | BF16 | BF16 | FP32 |
| Expert down_proj | BF16 | BF16 | FP32 |
| LM Head | BF16 | BF16 | FP32 |

#### FP8 Inference (E4M3)
| Component | Weight Precision | Activation | Scale |
|-----------|-----------------|------------|-------|
| Expert gate_up_proj | FP8 (128x128 blocks) | Dynamic FP8 | Per-block |
| Expert down_proj | FP8 (128x128 blocks) | Dynamic FP8 | Per-block |
| Q/K/V projections | BF16 | BF16 | N/A |
| Attention | BF16 | BF16 | N/A |
| Router | FP32 | FP32 | N/A |

### FP8 Block Quantization Details

DeepSeek-V3 uses fine-grained block-wise FP8 quantization:
- **Block size:** 128x128 elements
- **Format:** E4M3 (range [-448, 448], precision ~0.0625)
- **Scale:** One FP32 scale per block
- **Activation:** Dynamic quantization (scale computed per-forward pass)

This is different from per-tensor or per-channel quantization:
- Better precision for non-uniform weight distributions
- Compatible with DeepGEMM's block-structured GEMM
- Memory overhead: 1 FP32 scale per 128x128 = 0.024% overhead

### YaRN Precision Considerations

YaRN RoPE modifies the frequency computation:
- Correction dim finding: FP64 (math.log, math.floor, math.ceil)
- Linear ramp: FP32
- Frequency blending: FP32
- mscale computation: FP64 (math.log)
- Final inv_freq: stored as FP32 buffer
- cos/sin at runtime: cast to model dtype (BF16)

### MTP Precision
The MTP layer operates at model precision (BF16). The auxiliary loss is computed
in FP32 for numerical stability (same as the main CE loss).
