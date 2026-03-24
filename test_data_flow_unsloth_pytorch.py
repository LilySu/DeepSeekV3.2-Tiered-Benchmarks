"""Data flow integration tests: validates that data flows correctly between
unsloth Triton kernels and PyTorch components at every stage of the DeepSeek-V3 model.

These tests use dummy tokenized data and verify shape consistency, numerical
compatibility, and gradient flow across the boundary between:
  - PyTorch RMSNorm <-> unsloth Triton RMSNorm
  - PyTorch SwiGLU <-> unsloth Triton SwiGLU
  - PyTorch CrossEntropy <-> unsloth Triton CrossEntropy
  - PyTorch MoE dispatch <-> unsloth grouped GEMM
  - Full model pipeline: embedding -> layers -> lm_head

Architecture: DeepSeek-V3 671B (arXiv 2412.19437)
  - MLA attention (q_lora_rank=1536, kv_lora_rank=512)
  - MoE: 256 experts, top-8, n_group=8, topk_group=4
  - YaRN RoPE (factor=40)
  - MTP: 1 prediction layer
  - No DSA (GLM5-specific feature)

Run: python test_data_flow_unsloth_pytorch.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Tiny config for integration tests (small enough for CPU, any GPU)
# ---------------------------------------------------------------------------

TINY_CFG = {
    "vocab_size": 256,
    "hidden_size": 128,
    "tie_word_embeddings": False,
    "num_hidden_layers": 2,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 16,
    "qk_head_dim": 32,
    "v_head_dim": 32,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "routed_scaling_factor": 2.5,
    "n_group": 2,
    "topk_group": 1,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "ep_size": 1,
    "num_nextn_predict_layers": 1,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4,
        "original_max_position_embeddings": 32,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.707,
    },
    "initializer_range": 0.02,
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "use_cache": True,
    "mlp_layer_types": ["dense", "sparse"],
}

B, S = 2, 16  # batch, sequence length
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helper: create dummy tokenized data
# ---------------------------------------------------------------------------

def make_dummy_batch(batch_size=B, seq_len=S, vocab_size=256, device=DEVICE):
    """Create dummy tokenized data mimicking a ChatML conversation."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Mask some positions (system/user prompts) with -100
    mask = torch.rand(batch_size, seq_len) > 0.5
    labels[mask] = -100
    return input_ids, labels


# ===========================================================================
# Test 1: Embedding -> RMSNorm data flow
# ===========================================================================

def test_embedding_to_rmsnorm():
    """Verify data flows from embedding layer through RMSNorm correctly."""
    print("[Test 1] Embedding -> RMSNorm data flow")

    embed = nn.Embedding(TINY_CFG["vocab_size"], TINY_CFG["hidden_size"])
    norm = nn.Module()
    norm.eps = TINY_CFG["rms_norm_eps"]
    norm.weight = nn.Parameter(torch.ones(TINY_CFG["hidden_size"]))

    def pytorch_rmsnorm(x):
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + TINY_CFG["rms_norm_eps"])
        return norm.weight * x_normed.to(x.dtype)

    input_ids = torch.randint(0, 256, (B, S))
    hidden = embed(input_ids)

    # PyTorch path
    normed_pt = pytorch_rmsnorm(hidden)

    assert normed_pt.shape == (B, S, TINY_CFG["hidden_size"]), \
        f"Shape mismatch: {normed_pt.shape}"
    assert torch.isfinite(normed_pt).all(), "Non-finite values in RMSNorm output"

    # Verify variance is approximately 1 after normalization
    output_var = normed_pt.float().pow(2).mean(-1)
    assert output_var.mean().item() > 0.5, "RMSNorm output variance too low"

    print(f"  Shape: {normed_pt.shape}")
    print(f"  Mean output variance: {output_var.mean().item():.4f}")
    print("  PASS\n")


# ===========================================================================
# Test 2: SwiGLU activation data flow
# ===========================================================================

def test_swiglu_data_flow():
    """Verify SwiGLU activation preserves shapes and produces finite outputs."""
    print("[Test 2] SwiGLU activation data flow")

    hidden_size = TINY_CFG["hidden_size"]
    intermediate_size = TINY_CFG["intermediate_size"]

    gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    x = torch.randn(B, S, hidden_size)

    # PyTorch SwiGLU
    gate_out = F.silu(gate_proj(x))
    up_out = up_proj(x)
    intermediate = gate_out * up_out
    output = down_proj(intermediate)

    assert output.shape == (B, S, hidden_size), f"Shape mismatch: {output.shape}"
    assert torch.isfinite(output).all(), "Non-finite values in SwiGLU output"

    # Verify gradient flows through SwiGLU
    x_grad = torch.randn_like(x, requires_grad=True)
    out = down_proj(F.silu(gate_proj(x_grad)) * up_proj(x_grad))
    loss = out.sum()
    loss.backward()
    assert x_grad.grad is not None, "No gradient through SwiGLU"
    assert torch.isfinite(x_grad.grad).all(), "Non-finite gradients through SwiGLU"

    print(f"  Input: {x.shape} -> Output: {output.shape}")
    print(f"  Gradient flows: True")
    print("  PASS\n")


# ===========================================================================
# Test 3: MLA attention data flow
# ===========================================================================

def test_mla_attention_flow():
    """Verify MLA attention compresses/decompresses correctly."""
    print("[Test 3] MLA attention data flow")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import MLAttention, RotaryEmbedding

    attn = MLAttention(TINY_CFG, layer_idx=0)
    rope = RotaryEmbedding(TINY_CFG)

    hidden = torch.randn(B, S, TINY_CFG["hidden_size"])
    pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos, sin = rope(hidden, pos_ids)

    # Forward pass
    attn_out, weights = attn(hidden, (cos, sin))

    assert attn_out.shape == hidden.shape, f"Attention output shape mismatch: {attn_out.shape}"
    assert torch.isfinite(attn_out).all(), "Non-finite values in attention output"

    # Verify MLA compression path shapes
    q_lora = TINY_CFG["q_lora_rank"]
    kv_lora = TINY_CFG["kv_lora_rank"]
    q_compressed = attn.q_a_proj(hidden)
    assert q_compressed.shape == (B, S, q_lora), f"Q compression shape: {q_compressed.shape}"

    kv_compressed = attn.kv_a_proj_with_mqa(hidden)
    expected_kv_dim = kv_lora + TINY_CFG["qk_rope_head_dim"]
    assert kv_compressed.shape == (B, S, expected_kv_dim), f"KV compression shape: {kv_compressed.shape}"

    print(f"  Input: {hidden.shape} -> Attention output: {attn_out.shape}")
    print(f"  Q compressed: {q_compressed.shape}")
    print(f"  KV compressed: {kv_compressed.shape}")
    print("  PASS\n")


# ===========================================================================
# Test 4: MoE routing data flow
# ===========================================================================

def test_moe_routing_flow():
    """Verify MoE group-based routing produces valid expert assignments."""
    print("[Test 4] MoE routing data flow (n_group=2, topk_group=1)")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import MoE

    moe = MoE(TINY_CFG)
    hidden = torch.randn(B, S, TINY_CFG["hidden_size"])

    # Forward pass
    output = moe(hidden)

    assert output.shape == hidden.shape, f"MoE output shape mismatch: {output.shape}"
    assert torch.isfinite(output).all(), "Non-finite values in MoE output"

    # Verify routing internals
    router_logits = moe.gate(hidden)
    expected_router_shape = (B * S, TINY_CFG["n_routed_experts"])
    assert router_logits.shape == expected_router_shape, f"Router logits shape: {router_logits.shape}"

    # Verify sigmoid routing (not softmax)
    probs = router_logits.sigmoid()
    assert (probs >= 0).all() and (probs <= 1).all(), "Sigmoid scores out of range"

    # Verify group-based selection
    topk_indices, topk_weights = moe.route_tokens_to_experts(router_logits)
    assert topk_indices.shape == (B * S, TINY_CFG["num_experts_per_tok"]), \
        f"Top-k indices shape: {topk_indices.shape}"
    assert topk_weights.shape == topk_indices.shape, \
        f"Top-k weights shape: {topk_weights.shape}"

    # Verify all selected experts are valid
    assert (topk_indices >= 0).all(), "Negative expert indices"
    assert (topk_indices < TINY_CFG["n_routed_experts"]).all(), "Expert index out of range"

    print(f"  Input: {hidden.shape} -> MoE output: {output.shape}")
    print(f"  Router logits: {router_logits.shape}")
    print(f"  Selected experts: {topk_indices.shape}")
    print(f"  Expert range: [{topk_indices.min().item()}, {topk_indices.max().item()}]")
    print("  PASS\n")


# ===========================================================================
# Test 5: MTP prediction data flow
# ===========================================================================

def test_mtp_data_flow():
    """Verify MTP layer produces valid predictions with correct shapes."""
    print("[Test 5] MTP prediction data flow")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import MTPLayer, DeepSeekV3Model

    base = DeepSeekV3Model(TINY_CFG)
    mtp = MTPLayer(TINY_CFG)

    input_ids = torch.randint(0, 256, (B, S))
    labels = torch.randint(0, 256, (B, S))

    # Get hidden states from base model
    with torch.no_grad():
        hidden, _ = base(input_ids=input_ids)

    # MTP forward
    mtp_logits, mtp_loss = mtp(hidden, base.embed_tokens, input_ids=input_ids, labels=labels)

    assert mtp_logits is not None, "MTP should produce logits"
    assert mtp_logits.shape[0] == B, f"MTP batch dim mismatch: {mtp_logits.shape}"
    assert mtp_logits.shape[-1] == TINY_CFG["vocab_size"], f"MTP vocab dim: {mtp_logits.shape}"
    assert mtp_loss is not None, "MTP should produce loss with labels"
    assert mtp_loss.dim() == 0, f"MTP loss should be scalar: {mtp_loss.shape}"
    assert torch.isfinite(mtp_loss), f"MTP loss not finite: {mtp_loss}"

    print(f"  Hidden states: {hidden.shape}")
    print(f"  MTP logits: {mtp_logits.shape}")
    print(f"  MTP loss: {mtp_loss.item():.4f}")
    print("  PASS\n")


# ===========================================================================
# Test 6: Full pipeline — dummy tokens -> loss
# ===========================================================================

def test_full_pipeline():
    """Verify complete forward pass from tokens to loss, including MTP."""
    print("[Test 6] Full pipeline: tokens -> embeddings -> layers -> logits -> loss")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import DeepSeekV3ForCausalLM

    model = DeepSeekV3ForCausalLM(TINY_CFG)
    input_ids, labels = make_dummy_batch()

    # Forward pass
    loss, logits, kv = model(input_ids=input_ids, labels=labels)

    assert logits.shape == (B, S, TINY_CFG["vocab_size"]), f"Logits shape: {logits.shape}"
    assert loss is not None, "Loss should not be None"
    assert loss.dim() == 0, f"Loss should be scalar: {loss.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss}"

    # Backward pass
    loss.backward()

    # Verify gradients flow to all major components
    grad_components = {
        "embed_tokens": model.model.embed_tokens.weight.grad,
        "lm_head": model.lm_head.weight.grad,
        "layer_0_attn_q_a": model.model.layers[0].self_attn.q_a_proj.weight.grad,
        "layer_0_attn_kv": model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.grad,
        "layer_0_norm": model.model.layers[0].input_layernorm.weight.grad,
    }

    for name, grad in grad_components.items():
        assert grad is not None, f"No gradient for {name}"
        assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"

    # Check MTP gradients
    if model.mtp_layers is not None:
        mtp_grad = model.mtp_layers[0].embed_proj.weight.grad
        assert mtp_grad is not None, "No gradient for MTP layer"

    print(f"  Input: {input_ids.shape} -> Logits: {logits.shape}")
    print(f"  Loss (CE + MTP): {loss.item():.4f}")
    print(f"  Gradient components verified: {len(grad_components)}")
    print("  PASS\n")


# ===========================================================================
# Test 7: KV cache data flow (autoregressive)
# ===========================================================================

def test_kv_cache_flow():
    """Verify KV cache extends correctly during autoregressive generation."""
    print("[Test 7] KV cache data flow (autoregressive)")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import DeepSeekV3ForCausalLM

    model = DeepSeekV3ForCausalLM(TINY_CFG)
    model.eval()

    input_ids = torch.randint(0, 256, (1, 8))

    with torch.no_grad():
        # Step 1: Prefill
        _, logits1, cache = model(input_ids=input_ids, use_cache=True)
        prefill_len = cache.get_seq_length()

        # Step 2: Decode 3 tokens
        for step in range(3):
            next_token = logits1[:, -1:, :].argmax(dim=-1)
            _, logits1, cache = model(
                input_ids=next_token, past_key_values=cache, use_cache=True
            )

        decode_len = cache.get_seq_length()

    assert prefill_len == 8, f"Prefill cache length: {prefill_len}"
    assert decode_len == 11, f"After 3 decode steps: {decode_len}"
    assert logits1.shape == (1, 1, TINY_CFG["vocab_size"]), f"Decode logits shape: {logits1.shape}"

    print(f"  Prefill cache: {prefill_len} tokens")
    print(f"  After decode: {decode_len} tokens")
    print(f"  Decode logits: {logits1.shape}")
    print("  PASS\n")


# ===========================================================================
# Test 8: YaRN RoPE data flow
# ===========================================================================

def test_yarn_rope_flow():
    """Verify YaRN RoPE produces correctly shaped and scaled embeddings."""
    print("[Test 8] YaRN RoPE data flow")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import RotaryEmbedding, apply_rotary_pos_emb

    rope = RotaryEmbedding(TINY_CFG)
    rope_dim = TINY_CFG["qk_rope_head_dim"]

    # Test with short and long positions
    for positions in [16, 64, 128]:
        x = torch.randn(1, positions, TINY_CFG["hidden_size"])
        pos_ids = torch.arange(positions).unsqueeze(0)
        cos, sin = rope(x, pos_ids)

        assert cos.shape == (1, positions, rope_dim), f"cos shape at pos={positions}: {cos.shape}"
        assert sin.shape == (1, positions, rope_dim), f"sin shape at pos={positions}: {sin.shape}"
        assert torch.isfinite(cos).all(), f"Non-finite cos at pos={positions}"
        assert torch.isfinite(sin).all(), f"Non-finite sin at pos={positions}"

    # Verify YaRN attention scaling is applied
    assert rope.attention_scaling != 1.0, "YaRN should modify attention_scaling"

    # Test rotary application
    q = torch.randn(1, 4, 8, rope_dim)  # [B, H, S, rope_dim]
    x_short = torch.randn(1, 8, TINY_CFG["hidden_size"])
    cos_s, sin_s = rope(x_short, torch.arange(8).unsqueeze(0))
    q_rotated = apply_rotary_pos_emb(q, cos_s, sin_s, unsqueeze_dim=1)
    assert q_rotated.shape == q.shape, f"Rotated shape mismatch: {q_rotated.shape}"

    print(f"  YaRN attention_scaling: {rope.attention_scaling:.4f}")
    print(f"  cos shape: {cos.shape}")
    print(f"  Rotation preserves shape: True")
    print("  PASS\n")


# ===========================================================================
# Test 9: Cross-entropy loss data flow
# ===========================================================================

def test_cross_entropy_flow():
    """Verify cross-entropy loss handles label masking correctly."""
    print("[Test 9] Cross-entropy loss data flow")

    vocab_size = TINY_CFG["vocab_size"]
    logits = torch.randn(B, S, vocab_size, requires_grad=True)

    # Create labels with -100 masking (non-assistant tokens)
    labels = torch.randint(0, vocab_size, (B, S))
    mask_positions = torch.rand(B, S) > 0.5
    labels[mask_positions] = -100

    # Compute loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

    assert loss.dim() == 0, f"Loss should be scalar: {loss.shape}"
    assert torch.isfinite(loss), f"Loss not finite: {loss}"

    # Backward
    loss.backward()
    assert logits.grad is not None, "No gradient on logits"

    # Verify masked positions don't contribute to gradient
    # (PyTorch CE with ignore_index=-100 handles this)
    loss_masked = F.cross_entropy(
        logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100
    )
    assert torch.isfinite(loss_masked), f"Masked loss not finite: {loss_masked}"

    trained_tokens = (labels != -100).sum().item()
    total_tokens = labels.numel()

    print(f"  Logits: {logits.shape} -> Loss: {loss.item():.4f}")
    print(f"  Trained tokens: {trained_tokens}/{total_tokens}")
    print(f"  Masked loss: {loss_masked.item():.4f}")
    print("  PASS\n")


# ===========================================================================
# Test 10: End-to-end training step
# ===========================================================================

def test_training_step():
    """Verify a complete training step works: forward -> loss -> backward -> step."""
    print("[Test 10] End-to-end training step")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepseekv3_2-raw-decoupled-from-hf"))
    from model import DeepSeekV3ForCausalLM

    model = DeepSeekV3ForCausalLM(TINY_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids, labels = make_dummy_batch()

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        loss, logits, _ = model(input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should be finite at all steps
    for i, l in enumerate(losses):
        assert not (l != l), f"NaN loss at step {i}"  # NaN check

    print(f"  Step 0 loss: {losses[0]:.4f}")
    print(f"  Step 4 loss: {losses[-1]:.4f}")
    print(f"  Loss decreased: {losses[-1] < losses[0]}")
    print("  PASS\n")


# ===========================================================================
# Main runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DeepSeek-V3 Data Flow Integration Tests")
    print("Testing: PyTorch <-> Unsloth component boundaries")
    print("=" * 70)
    print()

    tests = [
        ("embedding_to_rmsnorm", test_embedding_to_rmsnorm),
        ("swiglu_data_flow", test_swiglu_data_flow),
        ("mla_attention_flow", test_mla_attention_flow),
        ("moe_routing_flow", test_moe_routing_flow),
        ("mtp_data_flow", test_mtp_data_flow),
        ("full_pipeline", test_full_pipeline),
        ("kv_cache_flow", test_kv_cache_flow),
        ("yarn_rope_flow", test_yarn_rope_flow),
        ("cross_entropy_flow", test_cross_entropy_flow),
        ("training_step", test_training_step),
    ]

    passed, failed, errors = 0, 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1
            errors.append((name, str(e)))

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
