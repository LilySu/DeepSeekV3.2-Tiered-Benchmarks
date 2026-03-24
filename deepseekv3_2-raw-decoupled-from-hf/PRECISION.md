# DeepSeek-V3 Precision Tracking

## Numerical Precision Strategy

### Default Precision: BF16

DeepSeek-V3 uses BF16 as its primary training and inference precision.
The official checkpoint includes FP8 quantization config (e4m3 format with
dynamic activation scheme and 128x128 weight block size).

### Component-Level Precision

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Token Embeddings | BF16 | Standard |
| RMSNorm | FP32 accumulation | Variance computation needs precision |
| MLA Q/K/V projections | BF16 | Standard linear layers |
| MLA attention scores | FP32 softmax | Numerical stability |
| RoPE (YaRN) | FP32 computation | Trigonometric functions |
| MoE Router (sigmoid) | FP32 | Routing decisions are precision-sensitive |
| MoE Expert weights | BF16 (FP8 optional) | Can use FP8 with block quantization |
| Shared Expert | BF16 | Standard |
| LM Head | BF16 | Standard |
| MTP Layer | BF16 | Standard |

### FP8 Quantization (Optional)

DeepSeek-V3 supports FP8 inference with:
- Format: E4M3 (4-bit exponent, 3-bit mantissa)
- Activation scheme: Dynamic
- Weight block size: 128x128
- Applied to: MoE expert gate_up_proj and down_proj

### YaRN RoPE Precision

YaRN scaling computation (correction dimensions, linear ramp) is done in
FP32 to ensure precise frequency interpolation. The final cos/sin embeddings
are cast to the model's working precision (BF16).

### MoE Router Precision

The router computes in FP32 for all stages:
- Linear projection: FP32
- Sigmoid activation: FP32
- Group scoring and top-k selection: FP32
- Weight normalization: FP32
- Correction bias addition: FP32

Results are cast back to model precision only for expert dispatch.
