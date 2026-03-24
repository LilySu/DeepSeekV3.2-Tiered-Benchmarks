# DeepSeek-V3 FlashInfer Precision Tracking

## Architecture Overview (arXiv 2412.19437)

DeepSeek-V3 671B (37B active parameters per token):
- 61 decoder layers (3 dense + 58 MoE)
- MLA with absorbed KV: qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
- MoE: 256 experts, top-8 selection, group routing (n_group=8, topk_group=4)
- YaRN RoPE scaling factor=40 for 128K context
- NO Dynamic Sparse Attention (standard causal)

## FP8 Precision Crossings Per Forward Pass

| Component | Crossings/Layer | Total (61 layers) | Format |
|-----------|----------------|-------------------|--------|
| KV cache write | 1 | 61 | BF16 -> FP8 E4M3 |
| KV cache read  | 1 | 61 | FP8 E4M3 -> BF16 |
| MoE gate_up GEMM | 1 | 58 (MoE layers) | BF16 -> FP8 -> BF16 |
| MoE down GEMM | 1 | 58 (MoE layers) | BF16 -> FP8 -> BF16 |
| **Total** | **~4** | **~238** | |

## Quantization Formats

### FP8 E4M3 (DeepSeek-V3 native)
- Range: [-448, 448]
- Min subnormal: ~0.001953
- Used for: KV cache, activation quantization, weight quantization
- Block size: 128x128 for activations, global scale for KV cache

### FlashInfer KV Cache Format
- Layout: [num_pages, page_size, 576] (512 ckv + 64 kpe concatenated)
- Scale: external global scale factor (not inline)
- Memory: 576 bytes/token in FP8 vs 1152 bytes/token in BF16

### DeepGEMM Activation Format
- Layout: (tensor_fp8, scales) pair
- Scaling: per-block with block_size=128
- scales shape: [num_rows, ceil(dim/128)]

## Measured Precision (from h100_test_precision_chain)

### Single FP8 Roundtrip (block_size=128)
- Max relative error: <7%
- Cosine similarity: >0.999
- Suitable for inference with minimal quality loss

### Chained 61 Roundtrips (simulating full forward pass)
- Cosine similarity: >0.90 (empirically ~0.95)
- Max relative error: increases but bounded
- Error partially cancels due to zero-mean quantization noise

## Key Precision Differences from GLM-5

| Aspect | GLM-5 | DeepSeek-V3 |
|--------|-------|-------------|
| RMSNorm eps | 1e-5 | 1e-6 (tighter) |
| FP8 block size | 128 | 128x128 (2D blocks for weights) |
| Num layers | 78 | 61 (fewer crossings) |
| DSA precision loss | Yes (top-k selection) | N/A (no DSA) |
| MoE routing | Flat top-8 | Group-based (less noise) |
| Initializer range | 0.02 | 0.006 (smaller init) |

## Attention Precision

FlashInfer FA3 backend for MLA:
- qk_nope_head_dim=128 matches FlashInfer native (no patch needed)
- Scaling: 1/sqrt(192) = 1/sqrt(qk_nope + qk_rope) for absorbed MLA
- BF16 computation throughout attention (no FP8 in attention itself)
- KV cache FP8 quantization is the only precision reduction in attention

## Recommendations

1. **KV Cache**: Use FP8 for inference, BF16 for training/fine-tuning
2. **Activations**: 128-block FP8 is sufficient for inference
3. **Weights**: 128x128 block FP8 with per-block scaling maintains <1% quality loss
4. **RoPE**: Keep in BF16/FP32 (64-dim, negligible cost)
5. **Router**: Keep in FP32 for stability (sigmoid + group selection)
