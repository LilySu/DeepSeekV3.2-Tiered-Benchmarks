# DeepSeek-V3 Precision Tracking (Triton Package)

Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)

## Numerical Precision Strategy

### Default Precision: BF16

DeepSeek-V3 uses BF16 as its primary training and inference precision.
The official checkpoint includes FP8 quantization config (e4m3 format with
dynamic activation scheme and 128x128 weight block size).

### Component-Level Precision

| Component | Precision | Rationale | Triton Kernel? |
|-----------|-----------|-----------|----------------|
| Token Embeddings | BF16 | Standard | No |
| RMSNorm (eps=1e-6) | FP32 accumulation | Variance computation needs precision | Yes (unsloth_rms_layernorm) |
| MLA Q/K/V projections | BF16 | Standard linear layers | No (LoRA fused via unsloth_fast_lora) |
| MLA attention scores | FP32 softmax | Numerical stability | No (PyTorch eager) |
| RoPE (YaRN, factor=40) | FP32 computation | Trigonometric functions need precision | No (PyTorch) |
| MoE Router (sigmoid) | FP32 | Routing decisions are precision-sensitive | No |
| MoE Expert weights | BF16 (FP8 optional) | Can use FP8 with block quantization | Yes (grouped GEMM) |
| Shared Expert | BF16 | Standard SwiGLU MLP | Yes (unsloth_swiglu) |
| SwiGLU activation | FP32 internal | Sigmoid in SiLU needs FP32 | Yes (unsloth_swiglu) |
| Cross-Entropy Loss | FP32 | Log-sum-exp needs precision | Yes (unsloth_cross_entropy) |
| LM Head | BF16 | Standard | No |
| MTP Layer | BF16 | Standard | No |

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

Key YaRN parameters:
- factor: 40 (extending from 4096 to 163840 tokens)
- beta_fast: 32, beta_slow: 1
- mscale: 1.0, mscale_all_dim: 1.0
- attention_scaling: computed from mscale parameters

### MoE Router Precision

The router computes in FP32 for all stages:
- Linear projection: FP32
- Sigmoid activation: FP32
- Group scoring (n_group=8, topk_group=4): FP32
- Top-k selection within groups: FP32
- Weight normalization: FP32
- Correction bias addition: FP32
- Scaling by routed_scaling_factor (2.5): FP32

Results are cast back to model precision only for expert dispatch.

### Triton Kernel Precision Notes

1. **RMSNorm** (unsloth_rms_layernorm.py):
   - Input cast to FP32 for variance computation
   - inv_var stored as FP32 for backward pass
   - Output in model dtype (BF16)
   - eps=1e-6 (differs from GLM5's 1e-5)

2. **SwiGLU** (unsloth_swiglu.py):
   - Sigmoid computed in FP32 for numerical stability
   - e * sigmoid(e) in FP32, then cast back
   - Backward derivatives computed in FP32

3. **Cross-Entropy** (unsloth_cross_entropy_loss.py):
   - All internal computation in FP32
   - Chunked for vocab_size=129280 (2 chunks of 65536)
   - logsumexp saved in FP32 for backward pass

4. **Grouped GEMM** (unsloth_moe/):
   - Accumulator in FP32 by default
   - Input/output in model dtype
   - Supports FP8 weight quantization path

### Key Differences from GLM5

| Aspect | DeepSeek-V3 | GLM5 |
|--------|-------------|------|
| RMSNorm eps | 1e-6 | 1e-5 |
| RoPE scaling | YaRN (factor=40) | None |
| Attention type | Standard causal | DSA sparse |
| MoE groups | 8 groups, top-4 | 1 group |
| MTP layers | 1 | 3 (shared) |
| Vocab size | 129280 (2 CE chunks) | 154880 (3 CE chunks) |
