"""DeepSeek-V3 component verification — minimal, no test framework."""

import torch
from config import DEEPSEEK_V3_CONFIG
from cache import KVCache
from model import (
    RMSNorm,
    RotaryEmbedding,
    MLAttention,
    FeedForward,
    TopkRouter,
    MoeExperts,
    MoE,
    MTPLayer,
    DecoderLayer,
    DeepSeekV3Model,
    DeepSeekV3ForCausalLM,
)

# Small config for quick verification
SMALL_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "n_shared_experts": 1,
    "n_routed_experts": 8,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 64,
    "q_lora_rank": 128,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "qk_nope_head_dim": 32,
    "qk_head_dim": 48,  # 32 + 16
    "n_group": 2,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "first_k_dense_replace": 2,
    "moe_layer_freq": 1,
    "ep_size": 1,
    "hidden_act": "silu",
    "max_position_embeddings": 512,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
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
    "dtype": "bfloat16",
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse"],
    "pad_token_id": None,
    "tie_word_embeddings": False,
}

cfg = SMALL_CONFIG
B, S = 2, 16

print("=" * 60)
print("DeepSeek-V3 Component Verification")
print("=" * 60)

# 1. RMSNorm
print("\n[1] RMSNorm")
norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
x = torch.randn(B, S, cfg["hidden_size"])
out = norm(x)
print(f"    Input: {x.shape} -> Output: {out.shape}")
print(f"    Params: {sum(p.numel() for p in norm.parameters()):,}")

# 2. RotaryEmbedding (YaRN)
print("\n[2] RotaryEmbedding (YaRN)")
rope = RotaryEmbedding(cfg)
pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
cos, sin = rope(x, pos_ids)
print(f"    cos: {cos.shape}, sin: {sin.shape}")
print(f"    attention_scaling: {rope.attention_scaling:.4f}")

# 3. MLAttention (no DSA — standard causal)
print("\n[3] MLAttention (no DSA)")
attn = MLAttention(cfg, layer_idx=0)
hidden = torch.randn(B, S, cfg["hidden_size"])
attn_out, _ = attn(hidden, (cos, sin), attention_mask=None)
print(f"    Input: {hidden.shape} -> Output: {attn_out.shape}")
print(f"    Params: {sum(p.numel() for p in attn.parameters()):,}")

# 4. FeedForward
print("\n[4] FeedForward")
ff = FeedForward(cfg)
ff_out = ff(hidden)
print(f"    Input: {hidden.shape} -> Output: {ff_out.shape}")
print(f"    Params: {sum(p.numel() for p in ff.parameters()):,}")

# 5. TopkRouter
print("\n[5] TopkRouter")
router = TopkRouter(cfg)
router_logits = router(hidden)
print(f"    Input: {hidden.shape} -> Logits: {router_logits.shape}")
print(f"    Params: {sum(p.numel() for p in router.parameters()):,}")

# 6. MoeExperts
print("\n[6] MoeExperts")
experts = MoeExperts(cfg)
top_k_index = torch.randint(0, cfg["n_routed_experts"], (B * S, cfg["num_experts_per_tok"]))
top_k_weights = torch.softmax(torch.randn(B * S, cfg["num_experts_per_tok"]), dim=-1)
expert_out = experts(hidden.view(-1, cfg["hidden_size"]), top_k_index, top_k_weights)
print(f"    Output: {expert_out.shape}")
print(f"    Params: {sum(p.numel() for p in experts.parameters()):,}")

# 7. MoE (group-based routing: n_group=2, topk_group=1)
print("\n[7] MoE (group routing)")
moe = MoE(cfg)
moe_out = moe(hidden)
print(f"    Input: {hidden.shape} -> Output: {moe_out.shape}")
print(f"    Params: {sum(p.numel() for p in moe.parameters()):,}")

# 8. MTPLayer
print("\n[8] MTPLayer")
mtp = MTPLayer(cfg)
import torch.nn as nn
embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
fake_ids = torch.randint(0, cfg["vocab_size"], (B, S))
fake_labels = torch.randint(0, cfg["vocab_size"], (B, S))
mtp_logits, mtp_loss = mtp(hidden, embed, input_ids=fake_ids, labels=fake_labels)
print(f"    MTP logits: {mtp_logits.shape if mtp_logits is not None else 'None'}")
print(f"    MTP loss: {mtp_loss.item():.4f}" if mtp_loss is not None else "    MTP loss: None")
print(f"    Params: {sum(p.numel() for p in mtp.parameters()):,}")

# 9. DecoderLayer (dense, layer 0)
print("\n[9] DecoderLayer (dense)")
layer0 = DecoderLayer(cfg, layer_idx=0)
layer_out = layer0(hidden, position_embeddings=(cos, sin))
print(f"    Input: {hidden.shape} -> Output: {layer_out.shape}")
print(f"    Params: {sum(p.numel() for p in layer0.parameters()):,}")

# 10. DecoderLayer (sparse, layer 2)
print("\n[10] DecoderLayer (sparse/MoE)")
layer2 = DecoderLayer(cfg, layer_idx=2)
layer_out = layer2(hidden, position_embeddings=(cos, sin))
print(f"    Input: {hidden.shape} -> Output: {layer_out.shape}")
print(f"    Params: {sum(p.numel() for p in layer2.parameters()):,}")

# 11. DeepSeekV3Model
print("\n[11] DeepSeekV3Model")
base_model = DeepSeekV3Model(cfg)
input_ids = torch.randint(0, cfg["vocab_size"], (B, S))
hidden_states, _ = base_model(input_ids)
print(f"    input_ids: {input_ids.shape} -> hidden_states: {hidden_states.shape}")
print(f"    Params: {sum(p.numel() for p in base_model.parameters()):,}")

# 12. DeepSeekV3ForCausalLM (with MTP)
print("\n[12] DeepSeekV3ForCausalLM (with MTP)")
model = DeepSeekV3ForCausalLM(cfg)
loss, logits, _ = model(input_ids, labels=input_ids)
print(f"    input_ids: {input_ids.shape} -> logits: {logits.shape}")
print(f"    loss (CE + MTP): {loss.item():.4f}")
print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

# Quick forward pass
print("\n" + "=" * 60)
print("Quick forward pass")
print("=" * 60)
out = model(torch.tensor([[1, 2, 3]]))
print(f"model(torch.tensor([[1, 2, 3]])) -> logits shape: {out[1].shape}")

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
print("\nAll components verified successfully!")
