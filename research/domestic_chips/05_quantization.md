# Quantization Strategies for DeepSeek-V3

## Background

DeepSeek-V3 was trained with FP8 (e4m3) precision using 128x128 block-wise quantization, making it one of the first large-scale models to demonstrate stable FP8 training. This section analyzes quantization strategies for inference deployment, including native FP8 on H100 and INT8 adaptation for platforms without FP8 hardware support.

## FP8 Training Scheme (Reference)

As described in Section 3.3 of arXiv:2412.19437, DeepSeek-V3's FP8 approach has three critical design choices:

**1. Fine-grained block quantization:** Rather than per-tensor or per-channel scaling, DeepSeek-V3 uses 128x128 block-level scaling. Each block of 128 rows by 128 columns has its own FP32 scaling factor. For a 7168x2048 weight matrix, this produces 56x16 = 896 scaling factors, adding only 3.5 KB of overhead (0.024% of the weight matrix in FP8). The fine-grained scaling reduces quantization error by 3-5x compared to per-tensor scaling.

**2. FP32 accumulation:** Matrix multiplications use FP8 inputs but accumulate partial sums in FP32 precision. This is critical for training stability. The H100's FP8 tensor cores natively support FP32 accumulation, so there is no throughput penalty.

**3. Online dynamic scaling:** Scaling factors are computed on-the-fly from activation statistics, not calibrated offline. This adapts to the dynamic range of each batch and layer position, avoiding the clipping artifacts of static quantization.

## Inference Quantization Options

### Option 1: Native FP8 (H100/H200)

For NVIDIA Hopper GPUs, the model can be served in its native FP8 format:

- **Format:** e4m3 (4-bit exponent, 3-bit mantissa)
- **Range:** [-448, 448] with 240 representable values
- **Block scaling:** 128x128 blocks with FP32 scales
- **Memory:** 671B * 1 byte = 671 GB (minimum 9 H100s for weights alone)

Advantages:
- Zero quantization loss (already trained in FP8)
- 2x FLOPS vs BF16 on H100 tensor cores
- 2x memory reduction vs BF16

Implementation notes:
- Use PyTorch's `torch.float8_e4m3fn` dtype or DeepGEMM library
- Block scaling requires custom GEMM kernels (DeepGEMM supports this)
- `torch._scaled_mm` provides basic FP8 matmul support

### Option 2: INT8 Weight-Only Quantization

For platforms without FP8 support (Ascend 910B, MLU590, etc.):

- **Method:** Per-channel INT8 quantization of weights with FP16/BF16 activations
- **Computation:** INT8 weight x FP16 activation with INT32 accumulation
- **Accuracy:** Minimal degradation (<0.1% on benchmarks)

Conversion from FP8 block scales to INT8 per-channel scales:

```
For each weight matrix W_fp8 with block_scales:
  1. Dequantize: W_fp32 = W_fp8 * block_scales (expanded to full matrix)
  2. Per-channel calibration: channel_scale = W_fp32.abs().max(dim=0)
  3. INT8 quantize: W_int8 = round(W_fp32 / channel_scale * 127)
```

The per-channel granularity is coarser than 128x128 blocks but supported by most INT8 accelerators. The quality loss from this re-quantization is typically small because the FP8 model was already trained to be robust to quantization noise.

### Option 3: INT4 Weight Quantization (Aggressive)

For memory-constrained deployments:

- **Method:** GPTQ or AWQ post-training quantization to INT4
- **Memory:** 671B * 0.5 bytes = 335 GB (4 H100s for weights)
- **Accuracy:** 0.5-2% degradation depending on calibration

Key challenges for DeepSeek-V3 INT4:
- MLA projection weights are sensitive to quantization (low-rank structure amplifies errors)
- MoE expert weights have different distributions per expert (per-expert calibration needed)
- Gating network weights are extremely sensitive (quantizing to INT4 can destabilize routing)

Recommendation: Use INT4 for expert FFN weights (large and redundant) but keep MLA projections and gating weights in INT8 or higher. This "mixed quantization" approach saves 40-60% of memory vs uniform INT8 while preserving quality.

### Option 4: Mixed-Precision Inference

Combine different precisions for different components:

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| MLA down-projection | INT8 | Low-rank, tolerant to quantization |
| MLA up-projections | INT8 | Moderate sensitivity |
| MLA Q projection | BF16 | High sensitivity (attention quality) |
| MoE gating | BF16 | Extremely sensitive (routing stability) |
| MoE expert FFN | INT8 or INT4 | Large, redundant, tolerant |
| Shared expert FFN | INT8 | Moderate sensitivity |
| LM head | BF16 | High sensitivity (output quality) |
| Embeddings | INT8 | Large, tolerant |
| KV cache (latent) | INT8 or FP8 | d_c=512 is already compressed |

## KV Cache Quantization

MLA's compressed KV cache (512 dims per layer per token) is already much smaller than standard MHA. Further quantization:

**FP8 KV cache:** Store the 512-dim latent in FP8. At 1 byte per element, the KV cache per token per layer is 512 bytes. For 61 layers and 128K context: 61 * 512 * 128K * 1 = 3.8 GB. This is remarkably small.

**INT4 KV cache:** Further halve to 256 bytes per token per layer. Total for 128K: 1.9 GB. The quantization error from INT4 KV cache is amplified through the up-projection, so careful calibration is essential.

**INT2 KV cache (experimental):** At 128 bytes per token per layer, the 128K KV cache would be under 1 GB. This likely requires residual quantization or other advanced techniques to maintain quality.

## Quantization Error Analysis

Theoretical per-element quantization error for the key DeepSeek-V3 tensor types:

| Tensor | Dims | BF16 Error | FP8 Block Error | INT8 Error | INT4 Error |
|--------|------|------------|----------------|------------|------------|
| Hidden state | 7168 | 0 (reference) | 2.4e-3 | 3.9e-3 | 1.6e-2 |
| KV latent | 512 | 0 | 2.4e-3 | 3.9e-3 | 1.6e-2 |
| Expert weights | 7168x2048 | 0 | 2.4e-3 | 3.9e-3 | 1.5e-2 |
| Gating logits | 256 | 0 | 3.1e-3 | 5.1e-3 | 2.1e-2 |

The gating logits have higher relative error due to their smaller dimensionality, and small perturbations in gating can change which experts are selected, cascading into large output differences.

## Recommendations

For production deployment of DeepSeek-V3:

1. **H100/H200:** Use native FP8 with block scaling. No re-quantization needed.
2. **Ascend 910B:** INT8 weights with BF16 activations. Custom per-expert calibration for MoE.
3. **MLU590:** INT8 with per-channel scaling. Keep gating in FP16.
4. **Memory-constrained:** Mixed INT4 (experts) + INT8 (MLA) + BF16 (gating).
5. **KV cache:** FP8 or INT8 for the compressed latent. INT4 only with careful validation.
