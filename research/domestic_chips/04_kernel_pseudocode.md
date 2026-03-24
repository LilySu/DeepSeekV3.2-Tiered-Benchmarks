# Kernel Pseudocode for MLA + MoE on DeepSeek-V3

## 1. MLA Attention Kernel (Fused)

This kernel implements the complete MLA attention computation, fusing the compressed KV handling with standard attention. It is the most performance-critical kernel in DeepSeek-V3.

### 1.1 MLA Prefill Kernel (FlashAttention-style)

```python
def mla_prefill_flash(
    x: Tensor[B, S, H],           # input hidden states
    W_dkv: Tensor[H, d_c],        # KV down-projection
    W_uk: Tensor[d_c, n_h * d_nope],  # K nope up-projection
    W_uv: Tensor[d_c, n_h * d_v],     # V up-projection
    W_rope_k: Tensor[H, d_rope],      # RoPE key projection
    W_q: Tensor[H, n_h * d_qk],       # Q projection
    W_o: Tensor[n_h * d_v, H],        # output projection
    cos_cache: Tensor[S, d_rope//2],   # YaRN RoPE cos
    sin_cache: Tensor[S, d_rope//2],   # YaRN RoPE sin
    scale: float,                       # 1/sqrt(d_qk)
) -> Tensor[B, S, H]:
    """
    Fused MLA prefill with FlashAttention-style tiling.
    No materialization of full S x S attention matrix.
    """
    # Block sizes (tuned per hardware)
    BLOCK_S = 128  # sequence block for Q
    BLOCK_K = 128  # sequence block for K/V

    for b in range(B):
        # === Phase 1: Compute all projections (pipelined with GEMM units) ===
        x_b = x[b]  # (S, H)

        # KV compression: x -> c_kv (stored in KV cache as compressed latent)
        c_kv = matmul(x_b, W_dkv)  # (S, d_c) -- THIS is what goes into KV cache

        # Q projection (full)
        q_full = matmul(x_b, W_q)  # (S, n_h * d_qk)
        q = reshape(q_full, [S, n_h, d_qk])  # (S, n_h, d_nope + d_rope)
        q_nope = q[:, :, :d_nope]
        q_rope = q[:, :, d_nope:]

        # Apply YaRN RoPE to q_rope
        q_rope = apply_rope(q_rope, cos_cache, sin_cache)

        # K projections from latent
        k_nope = matmul(c_kv, W_uk)  # (S, n_h * d_nope)
        k_nope = reshape(k_nope, [S, n_h, d_nope])

        k_rope = matmul(x_b, W_rope_k)  # (S, d_rope) -- shared across heads
        k_rope = apply_rope(k_rope, cos_cache, sin_cache)
        k_rope = broadcast(k_rope, n_h)  # (S, n_h, d_rope)

        # V from latent
        v = matmul(c_kv, W_uv)  # (S, n_h * d_v)
        v = reshape(v, [S, n_h, d_v])

        # Concatenate K
        k = concat(k_nope, k_rope, dim=-1)  # (S, n_h, d_qk)
        q = concat(q_nope, q_rope, dim=-1)

        # === Phase 2: FlashAttention-style tiled attention ===
        output = zeros(S, n_h, d_v)

        for q_start in range(0, S, BLOCK_S):
            q_end = min(q_start + BLOCK_S, S)
            q_block = q[q_start:q_end]  # (BLOCK_S, n_h, d_qk)

            # Online softmax state
            running_max = full(-inf, [q_end - q_start, n_h])
            running_sum = zeros(q_end - q_start, n_h)
            running_out = zeros(q_end - q_start, n_h, d_v)

            # Causal: only attend to positions <= q_position
            max_k_end = q_end  # causal bound

            for k_start in range(0, max_k_end, BLOCK_K):
                k_end = min(k_start + BLOCK_K, max_k_end)
                k_block = k[k_start:k_end]  # (BLOCK_K, n_h, d_qk)
                v_block = v[k_start:k_end]  # (BLOCK_K, n_h, d_v)

                # Compute attention scores
                scores = einsum("qhd,khd->hqk", q_block, k_block) * scale

                # Causal masking within block
                apply_causal_mask(scores, q_start, k_start)

                # Online softmax update
                block_max = scores.max(dim=-1)
                new_max = maximum(running_max, block_max)
                correction = exp(running_max - new_max)
                exp_scores = exp(scores - new_max.unsqueeze(-1))

                running_out = running_out * correction.unsqueeze(-1) + \
                              einsum("hqk,khd->qhd", exp_scores, v_block)
                running_sum = running_sum * correction + exp_scores.sum(dim=-1)
                running_max = new_max

            # Finalize
            output[q_start:q_end] = running_out / running_sum.unsqueeze(-1)

        # === Phase 3: Output projection ===
        output_flat = reshape(output, [S, n_h * d_v])
        result = matmul(output_flat, W_o)  # (S, H)

    return result + x  # residual connection
```

### 1.2 MLA Decode Kernel (Single-Token)

```python
def mla_decode_single_token(
    x: Tensor[B, 1, H],              # new token hidden state
    kv_cache: Tensor[B, cache_len, d_c],  # compressed KV cache
    pos: int,                          # current position
    # ... same weight matrices as prefill ...
) -> Tensor[B, 1, H]:
    """
    MLA decode for a single new token.

    Key optimization: KV cache stores the compressed latent (d_c=512),
    NOT the full K/V heads. Up-projection happens on-the-fly.

    Two strategies:
    A) "Standard": up-project cached c_kv to K, V, then standard attention.
       Pro: standard attention kernel works. Con: up-projects full cache.
    B) "Absorbed": absorb up-projection into attention computation.
       Pro: avoids materializing full K, V. Con: custom kernel needed.
    """
    # Strategy A (standard):
    q = project_q(x)                        # (B, 1, n_h, d_qk)
    c_kv_new = compress_kv(x)               # (B, 1, d_c)
    kv_cache = append(kv_cache, c_kv_new)   # (B, cache_len+1, d_c)

    # Up-project from cache (expensive for long contexts)
    k_nope = matmul(kv_cache, W_uk)         # (B, L, n_h, d_nope)
    k_rope = apply_rope(matmul(kv_cache_x, W_rope_k))  # needs original x
    v = matmul(kv_cache, W_uv)              # (B, L, n_h, d_v)

    # Standard attention (single query)
    attn = dot(q, k) * scale                # (B, 1, n_h, L)
    attn = softmax(attn)
    out = dot(attn, v)                      # (B, 1, n_h, d_v)

    return output_proj(out) + x
```

## 2. MoE Grouped Routing Kernel

```python
def moe_grouped_routing(
    x: Tensor[tokens, H],
    W_gate: Tensor[H, n_experts],
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    DeepSeek-V3 grouped routing.
    Returns (expert_indices, expert_weights, dispatch_mask)
    """
    # Step 1: Compute gating logits
    logits = matmul(x, W_gate)  # (tokens, n_experts)

    # Step 2: Group scoring
    experts_per_group = n_experts // n_group
    grouped_logits = reshape(logits, [tokens, n_group, experts_per_group])
    group_scores = grouped_logits.max(dim=-1)  # (tokens, n_group)

    # Step 3: Select top groups
    _, top_group_idx = topk(group_scores, topk_group, dim=-1)  # (tokens, topk_group)

    # Step 4: Mask non-selected groups
    group_mask = zeros(tokens, n_group, dtype=bool)
    scatter(group_mask, dim=1, index=top_group_idx, value=True)
    expert_mask = repeat_interleave(group_mask, experts_per_group, dim=1)
    masked_logits = where(expert_mask, logits, -inf)

    # Step 5: Select top-K experts from masked logits
    probs = softmax(masked_logits, dim=-1)
    expert_weights, expert_indices = topk(probs, top_k, dim=-1)

    # Step 6: Normalize weights
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    # Step 7: Build dispatch mask (for efficient scatter/gather)
    dispatch_mask = build_dispatch_mask(expert_indices, n_experts)

    return expert_indices, expert_weights, dispatch_mask
```

## 3. Expert Dispatch and Combine

```python
def expert_dispatch_combine(
    x: Tensor[tokens, H],
    expert_indices: Tensor[tokens, top_k],
    expert_weights: Tensor[tokens, top_k],
    expert_ffn_weights: List[Tuple[Tensor, Tensor, Tensor]],  # per-expert (W_gate, W_up, W_down)
) -> Tensor[tokens, H]:
    """
    Dispatch tokens to experts, compute, and combine results.
    Optimized for GPU with padding-free dispatch.
    """
    # Sort tokens by expert assignment for coalesced memory access
    flat_indices = expert_indices.reshape(-1)  # (tokens * top_k,)
    flat_weights = expert_weights.reshape(-1)

    # For each expert, gather assigned tokens
    output = zeros(tokens, H)

    for expert_id in range(n_experts):
        # Find tokens assigned to this expert
        mask = (flat_indices == expert_id)
        token_ids = mask.nonzero() // top_k  # original token indices
        weights = flat_weights[mask]

        if token_ids.numel() == 0:
            continue

        # Gather tokens
        x_expert = x[token_ids]  # (n_assigned, H)

        # Expert computation (SwiGLU)
        W_gate, W_up, W_down = expert_ffn_weights[expert_id]
        gate = silu(matmul(x_expert, W_gate))
        up = matmul(x_expert, W_up)
        expert_out = matmul(gate * up, W_down)

        # Scatter-add weighted results back
        output.scatter_add_(0, token_ids, expert_out * weights.unsqueeze(-1))

    return output
```

## 4. FP8 Block Quantization Kernel

```python
def fp8_block_quantize(
    tensor: Tensor[M, N],
    block_h: int = 128,
    block_w: int = 128,
    fp8_max: float = 448.0,  # e4m3 max
) -> Tuple[Tensor, Tensor]:
    """
    Block-wise FP8 quantization as used in DeepSeek-V3 training.
    Returns (quantized_tensor, per_block_scales)
    """
    n_blocks_h = ceil(M / block_h)
    n_blocks_w = ceil(N / block_w)

    scales = zeros(n_blocks_h, n_blocks_w)
    quantized = zeros(M, N, dtype=fp8_e4m3)

    for bh in range(n_blocks_h):
        for bw in range(n_blocks_w):
            h_start, h_end = bh * block_h, min((bh + 1) * block_h, M)
            w_start, w_end = bw * block_w, min((bw + 1) * block_w, N)

            block = tensor[h_start:h_end, w_start:w_end]

            # Per-block scaling
            amax = block.abs().max()
            scale = amax / fp8_max
            scale = max(scale, 1e-12)

            scales[bh, bw] = scale
            quantized[h_start:h_end, w_start:w_end] = round(block / scale).clamp(-fp8_max, fp8_max)

    return quantized, scales
```

## 5. RMSNorm + GEMM Fused Kernel (Platform-Agnostic)

```python
def fused_rmsnorm_gemm(
    x: Tensor[tokens, H],
    weight: Tensor[H],
    W: Tensor[H, out_dim],
    eps: float = 1e-6,
) -> Tensor[tokens, out_dim]:
    """
    Fused RMSNorm followed by GEMM.
    Eliminates intermediate store/load of normalized tensor.
    """
    # Per-token processing to keep normalized values in registers
    output = zeros(tokens, out_dim)

    # In practice, this is tiled along the token dimension
    for t in range(tokens):
        # RMSNorm in registers
        x_t = x[t]
        rms = sqrt(mean(x_t * x_t) + eps)
        x_norm = x_t / rms * weight  # stays in registers/shared memory

        # GEMM: x_norm @ W
        # x_norm is in registers, W columns are streamed from HBM
        for j in range(out_dim):
            output[t, j] = dot(x_norm, W[:, j])

    return output
```

These pseudocode implementations serve as reference specifications for porting DeepSeek-V3 kernels across different hardware platforms. The actual implementations would use hardware-specific APIs (CUDA, AscendCL, BANG, HIP) and include extensive tiling, pipelining, and memory management optimizations.
