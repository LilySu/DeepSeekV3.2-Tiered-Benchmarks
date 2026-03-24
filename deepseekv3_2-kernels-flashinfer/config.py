# DeepSeek-V3 model configuration.
# Default values match the DeepSeek-V3 671B (37B active) architecture
# from the paper (arXiv 2412.19437).
#
# This is a plain dict -- no HuggingFace PretrainedConfig dependency.
#
# Key differences from GLM-5:
#   - No DSA (Dynamic Sparse Attention) -- DeepSeek-V3 uses standard causal
#   - YaRN RoPE scaling (factor=40) for 128K context
#   - n_group=8, topk_group=4 (hierarchical group-based expert routing)
#   - topk_method="noaux_tc" (no auxiliary loss, top-k with correction)
#   - qk_nope_head_dim=128 (not 192 like GLM-5)
#   - v_head_dim=128 (not 256 like GLM-5)
#   - hidden_size=7168, num_attention_heads=128
#   - 61 layers total: first 3 dense, remaining 58 MoE
#   - MTP (Multi-Token Prediction) with 1 additional prediction layer

import json
import os

DEEPSEEK_V3_CONFIG = {
    # --- Vocabulary & embeddings ---
    "vocab_size": 129280,            # DeepSeek-V3 token vocabulary size
    "hidden_size": 7168,             # Transformer hidden dimension
    "tie_word_embeddings": False,    # Whether lm_head shares embed_tokens weights

    # --- Layers ---
    "num_hidden_layers": 61,         # Total decoder layers (3 dense + 58 MoE)
    "intermediate_size": 18432,      # Dense MLP intermediate size (for layers 0-2)

    # --- Attention ---
    "num_attention_heads": 128,      # Number of query heads
    "num_key_value_heads": 128,      # MLA uses same count (no GQA -- MLA compresses differently)
    "attention_bias": False,         # No bias on Q/K/V/O projections
    "attention_dropout": 0.0,        # Attention dropout rate

    # --- MLA (Multi-head Latent Attention) ---
    "q_lora_rank": 1536,             # Query compression bottleneck (7168 -> 1536 -> 128*192)
    "kv_lora_rank": 512,             # KV compression bottleneck (7168 -> 512 -> 128*(128+128))
    "qk_rope_head_dim": 64,         # RoPE-applied portion of each head
    "qk_nope_head_dim": 128,        # Non-RoPE portion of each head
    "qk_head_dim": 192,             # Total = qk_nope_head_dim + qk_rope_head_dim
    "v_head_dim": 128,              # Value head dimension

    # --- MoE (Mixture of Experts) ---
    "n_routed_experts": 256,         # Total expert count
    "n_shared_experts": 1,           # Always-active shared expert count
    "num_experts_per_tok": 8,        # Top-k experts selected per token
    "moe_intermediate_size": 2048,   # Per-expert intermediate dimension
    "routed_scaling_factor": 2.5,    # Scale factor applied after expert weighted sum
    "n_group": 8,                    # Number of expert groups for group-based selection
    "topk_group": 4,                 # Number of groups selected before within-group topk
    "norm_topk_prob": True,          # Normalize routing weights after selection
    "topk_method": "noaux_tc",       # No auxiliary loss, top-k with correction bias

    # --- Activation & normalization ---
    "hidden_act": "silu",            # SwiGLU uses SiLU (sigmoid linear unit)
    "rms_norm_eps": 1e-6,            # RMSNorm epsilon (DeepSeek-V3 uses 1e-6, not 1e-5)

    # --- Positional encoding (YaRN RoPE) ---
    "max_position_embeddings": 163840,  # Maximum context length (128K + margin)
    "rope_theta": 10000.0,              # RoPE base frequency
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,                    # YaRN scaling factor for 128K context
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.707,
    },

    # --- Multi-Token Prediction (MTP) ---
    "num_nextn_predict_layers": 1,   # Number of MTP prediction layers (HF key name)

    # --- Weight initialization ---
    "initializer_range": 0.006,      # Std dev for weight initialization (DeepSeek-V3 uses smaller)

    # --- Special tokens ---
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,

    # --- Cache ---
    "use_cache": True,

    # --- MLP layer types: first 3 dense, remaining 58 sparse (MoE) ---
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 58,

    # --- FP8 quantization config ---
    "fp8_format": "e4m3",            # FP8 format for KV cache and activations
    "fp8_block_size": 128,           # Block size for per-block FP8 scaling
    "fp8_activation_block_size": [128, 128],  # 128x128 block for activation quantization
}


def load_config_from_hf(checkpoint_dir: str) -> dict:
    """Read config.json from a HuggingFace checkpoint directory.

    Maps HF config keys to our standalone config dict format.
    Falls back to DEEPSEEK_V3_CONFIG defaults for any missing keys.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    config = dict(DEEPSEEK_V3_CONFIG)

    # Keys that match directly between HF and standalone configs
    direct_keys = [
        "vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "attention_bias", "attention_dropout",
        "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "moe_intermediate_size", "routed_scaling_factor", "n_group", "topk_group",
        "norm_topk_prob", "hidden_act", "rms_norm_eps", "max_position_embeddings",
        "initializer_range", "pad_token_id", "bos_token_id", "eos_token_id",
        "tie_word_embeddings", "use_cache",
        "topk_method", "mtp_num_layers",
    ]
    for key in direct_keys:
        if key in hf_config:
            config[key] = hf_config[key]

    # Computed: total QK head dim = nope + rope
    config["qk_head_dim"] = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]

    # Extract rope_theta from nested rope_parameters or rope_scaling if present
    rope_params = hf_config.get("rope_scaling")
    if isinstance(rope_params, dict):
        config["rope_scaling"] = rope_params
        if "rope_theta" in rope_params:
            config["rope_theta"] = rope_params["rope_theta"]
    if "rope_theta" in hf_config:
        config["rope_theta"] = hf_config["rope_theta"]

    # MLP layer types: first N are dense, rest are sparse (MoE)
    if "mlp_layer_types" in hf_config and hf_config["mlp_layer_types"] is not None:
        config["mlp_layer_types"] = hf_config["mlp_layer_types"]
    else:
        n = config["num_hidden_layers"]
        n_dense = hf_config.get("first_k_dense_replace", 3)
        config["mlp_layer_types"] = ["dense"] * min(n_dense, n) + ["sparse"] * max(0, n - n_dense)

    # FP8 config
    if "quantization_config" in hf_config:
        qc = hf_config["quantization_config"]
        if "activation_scheme" in qc:
            config["fp8_format"] = qc.get("fmt", "e4m3")
        if "weight_block_size" in qc:
            config["fp8_activation_block_size"] = qc["weight_block_size"]

    return config
