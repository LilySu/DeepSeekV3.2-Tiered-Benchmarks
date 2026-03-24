"""Tests for standalone DeepSeek-V3 model."""

import sys
import torch

sys.path.insert(0, "/home/lily/wsl_git/deepseekv3_2")

from model import (
    RMSNorm, RotaryEmbedding, MLAttention,
    FeedForward, TopkRouter, MoeExperts, MoE, MTPLayer,
    DecoderLayer, DeepSeekV3Model, DeepSeekV3ForCausalLM,
    rotate_half, apply_rotary_pos_emb, repeat_kv, make_causal_mask,
)
from cache import KVCache


# Small config for testing (matches HF structure but tiny dimensions)
SMALL_CFG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 8,
    "kv_lora_rank": 16,
    "v_head_dim": 16,
    "qk_nope_head_dim": 16,
    "qk_head_dim": 24,  # 16 + 8
    "attention_bias": False,
    "attention_dropout": 0.0,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5,
    "n_group": 2,
    "topk_group": 1,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "ep_size": 1,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
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
    "num_nextn_predict_layers": 1,
    "pad_token_id": None,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    # First layer dense, rest sparse
    "mlp_layer_types": ["dense", "sparse", "sparse"],
}


def test_smoke_instantiate():
    """Model instantiates and has non-zero parameter count."""
    model = DeepSeekV3ForCausalLM(SMALL_CFG)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "Model has no parameters"
    print(f"  params: {n_params:,}")


def test_forward_shapes():
    """Forward pass produces correct logit shapes."""
    model = DeepSeekV3ForCausalLM(SMALL_CFG)
    ids = torch.randint(0, 256, (2, 8))

    loss, logits, cache = model(input_ids=ids)

    assert logits.shape == (2, 8, 256), f"Expected (2, 8, 256), got {logits.shape}"
    assert loss is None, "Loss should be None when no labels provided"
    assert cache is None, "Cache should be None when use_cache not set"
    print(f"  logits: {logits.shape}")


def test_forward_with_labels():
    """Forward pass with labels produces scalar loss (includes MTP loss)."""
    model = DeepSeekV3ForCausalLM(SMALL_CFG)
    ids = torch.randint(0, 256, (2, 8))
    labels = torch.randint(0, 256, (2, 8))

    loss, logits, cache = model(input_ids=ids, labels=labels)

    assert loss is not None, "Loss should not be None with labels"
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"  loss: {loss.item():.4f}")


def test_kv_cache():
    """Two-step generation correctly uses and extends KV cache."""
    model = DeepSeekV3ForCausalLM(SMALL_CFG)
    model.eval()

    ids = torch.randint(0, 256, (1, 6))

    with torch.no_grad():
        _, logits1, cache = model(input_ids=ids, use_cache=True)

    assert cache is not None, "Cache should be returned when use_cache=True"
    assert cache.get_seq_length() == 6, f"Cache should have 6 tokens, got {cache.get_seq_length()}"

    next_token = logits1[:, -1:, :].argmax(dim=-1)
    with torch.no_grad():
        _, logits2, cache = model(input_ids=next_token, past_key_values=cache, use_cache=True)

    assert cache.get_seq_length() == 7, f"Cache should have 7 tokens, got {cache.get_seq_length()}"
    assert logits2.shape == (1, 1, 256), f"Expected (1, 1, 256), got {logits2.shape}"
    print(f"  cache seq_length after 2 steps: {cache.get_seq_length()}")


def test_gradient_flow():
    """loss.backward() runs without error and produces gradients."""
    model = DeepSeekV3ForCausalLM(SMALL_CFG)
    ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))

    loss, logits, _ = model(input_ids=ids, labels=labels)
    loss.backward()

    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert has_grad > 0, "No parameters received gradients"
    print(f"  {has_grad}/{total} parameters have non-zero gradients")


def test_causal_mask():
    """Causal mask has correct shape and values."""
    mask = make_causal_mask(seq_len=4, past_len=0, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 4, 4), f"Expected (1,1,4,4), got {mask.shape}"

    for i in range(4):
        for j in range(4):
            val = mask[0, 0, i, j].item()
            if j <= i:
                assert val == 0.0, f"mask[{i},{j}] should be 0, got {val}"
            else:
                assert val < -1e30, f"mask[{i},{j}] should be -inf, got {val}"

    mask2 = make_causal_mask(seq_len=2, past_len=3, dtype=torch.float32, device="cpu")
    assert mask2.shape == (1, 1, 2, 5), f"Expected (1,1,2,5), got {mask2.shape}"
    assert mask2[0, 0, 0, 3].item() == 0.0
    assert mask2[0, 0, 0, 4].item() < -1e30
    print("  causal mask OK")


def test_parameter_names_match_hf():
    """Parameter names match HF checkpoint convention for weight loading."""
    model = DeepSeekV3ForCausalLM(SMALL_CFG)
    names = set(model.state_dict().keys())

    expected_prefixes = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        # Layer 0 (dense) attention
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        # Norms
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        # Dense MLP (layer 0)
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        # Sparse MoE (layer 1)
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        # MTP layer
        "mtp_layers.0.embed_proj.weight",
        "mtp_layers.0.norm.weight",
        "mtp_layers.0.head.weight",
    ]

    missing = []
    for key in expected_prefixes:
        if key not in names:
            missing.append(key)

    if missing:
        print(f"  WARNING: missing keys: {missing}")
        for n in sorted(names):
            print(f"    {n}")
        assert False, f"Missing {len(missing)} expected keys"
    else:
        print(f"  all {len(expected_prefixes)} expected keys present")


def test_mtp_layer():
    """MTP layer produces logits and loss."""
    mtp = MTPLayer(SMALL_CFG)
    from model import DeepSeekV3Model
    base = DeepSeekV3Model(SMALL_CFG)

    ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))

    with torch.no_grad():
        hidden, _ = base(input_ids=ids)

    mtp_logits, mtp_loss = mtp(hidden, base.embed_tokens, input_ids=ids, labels=labels)
    assert mtp_logits is not None, "MTP should produce logits"
    assert mtp_loss is not None, "MTP should produce loss with labels"
    assert mtp_loss.dim() == 0, f"MTP loss should be scalar, got {mtp_loss.shape}"
    print(f"  MTP logits: {mtp_logits.shape}, loss: {mtp_loss.item():.4f}")


def test_yarn_rope():
    """YaRN RoPE produces correctly shaped outputs."""
    rope = RotaryEmbedding(SMALL_CFG)
    x = torch.randn(2, 8, 64)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos, sin = rope(x, pos)
    rope_dim = SMALL_CFG["qk_rope_head_dim"]
    assert cos.shape == (2, 8, rope_dim), f"Expected cos shape (2, 8, {rope_dim}), got {cos.shape}"
    assert sin.shape == (2, 8, rope_dim), f"Expected sin shape (2, 8, {rope_dim}), got {sin.shape}"
    print(f"  YaRN RoPE cos: {cos.shape}, attention_scaling: {rope.attention_scaling:.4f}")


def test_group_routing():
    """MoE group-based routing works with n_group=2, topk_group=1."""
    moe = MoE(SMALL_CFG)
    x = torch.randn(2, 4, 64)
    out = moe(x)
    assert out.shape == x.shape, f"MoE output shape mismatch: {out.shape} vs {x.shape}"
    print(f"  Group routing OK, output: {out.shape}")


def test_components():
    """Individual components work correctly."""
    norm = RMSNorm(64, eps=1e-6)
    x = torch.randn(2, 4, 64)
    out = norm(x)
    assert out.shape == x.shape
    print("  RMSNorm OK")

    rope = RotaryEmbedding(SMALL_CFG)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = rope(x[:, :1, :], pos)
    rope_dim = SMALL_CFG["qk_rope_head_dim"]
    assert cos.shape == (1, 8, rope_dim)
    print("  RotaryEmbedding OK")

    kv = torch.randn(1, 2, 4, 16)
    expanded = repeat_kv(kv, 3)
    assert expanded.shape == (1, 6, 4, 16)
    print("  repeat_kv OK")

    cache = KVCache(2)
    k, v = torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)
    k_out, v_out = cache.update(k, v, 0)
    assert cache.get_seq_length(0) == 3
    k2, v2 = torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16)
    k_out, v_out = cache.update(k2, v2, 0)
    assert cache.get_seq_length(0) == 4
    print("  KVCache OK")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("smoke_instantiate", test_smoke_instantiate),
        ("forward_shapes", test_forward_shapes),
        ("forward_with_labels", test_forward_with_labels),
        ("kv_cache", test_kv_cache),
        ("gradient_flow", test_gradient_flow),
        ("causal_mask", test_causal_mask),
        ("parameter_names_match_hf", test_parameter_names_match_hf),
        ("mtp_layer", test_mtp_layer),
        ("yarn_rope", test_yarn_rope),
        ("group_routing", test_group_routing),
        ("components", test_components),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
