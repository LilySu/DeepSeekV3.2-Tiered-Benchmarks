# DeepSeek-V3 Kernel Selection Decisions -- H100 (SM90)

Research-backed resolution of kernel selection for each major component.
Each decision cites the specific source that settles it.

---

## RESOLVED DECISIONS

### #1. FlashMLA for MLA Attention -- CONFIRMED, native support

**Source:** FlashMLA repository (github.com/deepseek-ai/FlashMLA)
```
flash_mla_with_kvcache(): head_dim_v -- "Must be 512"
```

**Resolution:** FlashMLA was built BY DeepSeek specifically for their MLA architecture. DeepSeek-V3's v_head_dim=128 works in **absorbed mode** where the V output becomes kv_lora_rank=512 via weight absorption into kv_b_proj.

**Decision:** Use FlashMLA in absorbed mode for inference. The compressed KV cache stores kv_lora_rank=512 + qk_rope_head_dim=64 = 576 dims per token. FlashMLA handles this natively.

**Risk level:** None for inference. Low for training (PyTorch eager fallback).

---

### #2. DeepGEMM FP8 Grouped GEMM for MoE -- CONFIRMED, native support

**Source:** DeepGEMM repository (github.com/deepseek-ai/DeepGEMM)
```
grouped_gemm_fp8(A, B, C, expert_ids, num_groups, ...)
```

**Resolution:** DeepGEMM provides FP8 (e4m3) grouped GEMM using Hopper TMA instructions. DeepSeek-V3's MoE config (256 experts, top-8, intermediate=2048) is the primary target for this kernel.

**Decision:** Use DeepGEMM for MoE expert dispatch in FP8 precision. Fall back to Triton grouped GEMM (unsloth_moe) for BF16 training. The FP8 quantization uses 128x128 block size with dynamic activation scheme.

**Risk level:** None. DeepGEMM was built for exactly this model.

---

### #3. Group-based routing (n_group=8, topk_group=4) -- CONFIRMED works

**Source:** DeepSeek-V3 config.json and inference/model.py
```python
"n_group": 8,
"topk_group": 4,
"topk_method": "noaux_tc",
"scoring_func": "sigmoid",
```

**Resolution:** DeepSeek-V3 divides 256 experts into 8 groups of 32. Each token selects top-4 groups, then top-8 experts within those 4 groups. This limits each token to at most 4 expert-parallel nodes, optimizing multi-node inference.

SGLang's `moe_fused_gate` kernel supports this group-based routing via the `topk_method="noaux_tc"` codepath, which fuses the sigmoid, bias addition, group scoring, and top-k selection into a single kernel.

**Decision:** Use SGLang's fused gate kernel for the routing computation. Fall back to PyTorch eager routing (already implemented) for correctness verification.

**Risk level:** None. This is the production serving path.

---

### #4. No DSA -- CONFIRMED, not applicable

**Source:** DeepSeek-V3 Technical Report (arXiv 2412.19437)

**Resolution:** DeepSeek-V3 does NOT use Dynamic Sparse Attention (DSA). DSA is a GLM-5 innovation (arXiv 2602.15763v2). DeepSeek-V3 uses standard causal attention via MLA. The DSA indexer, sparse attention masks, and associated kernels from the GLM-5 implementation are not needed.

**Decision:** No DSA kernels needed. Standard causal attention (FlashMLA / FlashInfer) suffices.

**Risk level:** None. Architecture confirmed from paper and config.

---

### #5. YaRN RoPE -- CONFIRMED, standard implementation

**Source:** DeepSeek-V3 config.json
```json
"rope_scaling": {
    "type": "yarn",
    "factor": 40,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0
}
```

**Resolution:** YaRN (Yet another RoPE extensioN) modifies the inv_freq computation to blend between original and scaled frequencies. The mscale factor adjusts the attention softmax temperature. For DeepSeek-V3, the factor=40 extends from 4096 to 163,840 positions.

The YaRN computation is done once during model initialization and produces modified inv_freq values. No special kernel needed -- the cos/sin values are precomputed and applied identically to standard RoPE at runtime.

**Decision:** Implement YaRN in the RotaryEmbedding initialization. Use standard RoPE application at runtime. No kernel needed.

**Risk level:** None.

---

### #6. FlashInfer as alternative attention kernel -- CONFIRMED works

**Source:** FlashInfer documentation and DeepSeek-V3 inference implementations

**Resolution:** FlashInfer provides:
1. Page-attention with variable-length sequences
2. Support for MLA's compressed KV format
3. Paged KV cache for efficient memory management
4. FP8 attention for inference

FlashInfer is used by SGLang as the primary attention kernel for DeepSeek-V3 serving.

**Decision:** Use FlashInfer as the alternative to FlashMLA. It provides broader hardware compatibility (A100, H100, B200) at slightly lower peak performance compared to FlashMLA on H100.

**Risk level:** None for A100/H100. Monitor for B200 support.

---

### #7. MTP Speculative Decoding -- CONFIRMED, 1 layer

**Source:** DeepSeek-V3 config.json
```json
"num_nextn_predict_layers": 1
```

**Resolution:** DeepSeek-V3 uses 1 MTP layer (not 3 like GLM-5). The MTP layer shares the main model's embedding and uses a lightweight projection to predict the next-next token. During inference, this enables speculative decoding with ~2.55 average acceptance length.

**Decision:** Implement MTP as a thin wrapper: embed_proj(cat(hidden, next_embed)) -> norm -> head. Use for speculative decoding during inference. Add as auxiliary loss during training.

**Risk level:** Low. MTP is optional for basic inference.

---

### #8. FP8 Quantization (e4m3, 128x128 blocks) -- CONFIRMED native support

**Source:** DeepSeek-V3 config.json
```json
"quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
}
```

**Resolution:** DeepSeek-V3's official checkpoint includes FP8 quantized weights. The quantization uses:
- E4M3 format (4-bit exponent, 3-bit mantissa, 1 sign bit)
- Block-wise quantization with 128x128 blocks
- Dynamic activation quantization (scale computed per-forward)
- Applied to MoE expert weights (gate_up_proj, down_proj)

DeepGEMM natively supports this quantization format.

**Decision:** Use DeepGEMM FP8 for MoE experts. Support both BF16 and FP8 inference paths.

**Risk level:** None.

---

## KERNEL PRIORITY ORDER

For DeepSeek-V3 on H100:

1. **FlashMLA + DeepGEMM** (highest throughput): FlashMLA for MLA attention, DeepGEMM for FP8 MoE
2. **FlashInfer + Triton grouped GEMM** (broader compatibility): FlashInfer for attention, unsloth grouped GEMM for MoE
3. **Pure Triton** (fallback): Custom Triton kernels for all components
4. **Pure PyTorch** (reference): No kernel dependencies, for correctness verification

## COMPARISON WITH GLM-5 KERNEL DECISIONS

| Decision | GLM-5 | DeepSeek-V3 |
|----------|-------|-------------|
| Attention kernel | FlashMLA (absorbed) + DSA | FlashMLA (absorbed), no DSA |
| MoE kernel | DeepGEMM FP8 | DeepGEMM FP8 (same) |
| DSA indexer | DeepGEMM fp8_mqa_logits | N/A (no DSA) |
| Routing | Flat (n_group=1) | Group-based (n_group=8) |
| RoPE | Standard | YaRN (factor=40) |
| MTP | Not implemented | 1 layer (speculative decoding) |
