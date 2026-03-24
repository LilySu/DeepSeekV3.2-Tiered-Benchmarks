# Precision Analysis: DeepSeek-V3 FlashMLA + DeepGEMM

## Overview

DeepSeek-V3 uses a mixed-precision strategy that balances throughput with
numerical accuracy across its two most compute-intensive operations:

1. **MLA Attention** (FlashMLA): BF16 computation
2. **MoE Expert Dispatch** (DeepGEMM): FP8 (E4M3) with per-block scaling

## Precision Chain

```
Input (BF16)
  |
  v
RMSNorm (FP32 accumulation, BF16 output)
  |
  v
MLA Attention:
  Q/K/V projections (BF16 matmul)
  -> YaRN RoPE (FP32 cos/sin, BF16 output)
  -> FlashMLA kernel (BF16 accumulation, BF16 output)
  -> O projection (BF16 matmul)
  |
  v
RMSNorm (FP32 accumulation, BF16 output)
  |
  v
MoE FFN:
  Router (BF16 matmul -> sigmoid -> FP32 topk)
  -> Expert dispatch:
     FP8 quantise (E4M3, 128x128 block scales)
     -> DeepGEMM grouped GEMM (FP8 x FP8 -> BF16)
     -> SwiGLU (BF16)
     -> DeepGEMM down projection (FP8 x FP8 -> BF16)
  -> Weighted sum (BF16)
  |
  v
Residual add (BF16)
```

## FP8 Block Quantisation

DeepGEMM uses **per-block** (128x128 tile) quantisation rather than
per-tensor or per-channel:

- **Format**: E4M3 (4 exponent bits, 3 mantissa bits)
- **Range**: [-448, 448]
- **Block size**: 128x128 elements
- **Scale factor**: `absmax(block) / 448.0` (one FP32 scale per block)

### Why per-block?

Per-block quantisation provides a middle ground:
- **Better than per-tensor**: captures local magnitude variations
- **Cheaper than per-channel**: fewer scale factors, TMA-friendly alignment
- **128x128 matches TMA tile**: zero overhead for scale lookup

### Expected Error

| Component | Mean Relative Error | Notes |
|-----------|-------------------|-------|
| FP8 round-trip | < 5% | Single quantise/dequantise |
| FP8 GEMM vs BF16 GEMM | < 10% | One matrix multiply |
| MoE layer (3 GEMMs) | < 15% | gate + up + down projections |
| Full model (61 layers) | Bounded | Residual connections prevent accumulation |

## Key Design Decisions

### 1. BF16 for Attention, FP8 for MoE

- Attention softmax is sensitive to quantisation noise (exponential function)
- MoE expert dispatch involves many independent small GEMMs where FP8's
  throughput advantage outweighs its precision cost
- Residual connections after each MoE layer bound error accumulation

### 2. FP32 for Normalization

- RMSNorm computes variance in FP32 to avoid BF16 overflow on large norms
- Inverse sqrt is computed in FP32 then cast back to BF16
- This is critical for training stability (eps=1e-6 requires FP32)

### 3. FP32 for RoPE Frequencies

- YaRN frequency interpolation requires FP32 precision
- cos/sin cache is computed in FP32, cast to BF16 for application
- The mscale factor is FP32

### 4. Router in BF16

- Sigmoid scores are computed in BF16 (sufficient for routing decisions)
- Top-k selection uses FP32 for numerical tie-breaking
- Weight normalisation is in FP32

## Validation Strategy

1. **Unit tests**: FP8 round-trip error bounded per block
2. **Layer tests**: MoE layer output compared against FP32 reference
3. **End-to-end tests**: Full model BF16 vs FP32 relative error < 10%
4. **Perplexity tests**: Production perplexity within 0.1% of FP32 baseline

## References

- arXiv 2412.19437: DeepSeek-V3 Technical Report
- DeepGEMM: FP8 GEMM library for Hopper GPUs
- FlashMLA: Flash-decoding for Multi-head Latent Attention
