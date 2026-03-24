# DeepSeek-V3 model configuration.
# Default values match the DeepSeek-V3 671B architecture from the paper (arXiv 2412.19437).
# Config values from: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
#
# This is a plain dict -- no HuggingFace PretrainedConfig dependency.
#
# DeepSeek-V3 671B: 671B total parameters, ~37B active per token.
# 61 transformer layers (3 dense + 58 MoE), MLA attention, 256 routed experts,
# auxiliary-loss-free load balancing, multi-token prediction (1 MTP layer).
#
# KEY DIFFERENCES FROM GLM5 (arXiv 2602.15763v2):
# - No DSA (Dynamic Sparse Attention) -- this is GLM5-specific
# - YaRN RoPE scaling (factor=40, from 4096 to 163840)
# - Different MoE routing: n_group=8, topk_group=4, topk_method="noaux_tc"
# - 1 MTP layer (GLM5 has 3 shared MTP layers)
# - Different dimensions: hidden=7168, heads=128, layers=61

import json
import os


DEEPSEEK_V3_CONFIG = {
    # --- Vocabulary & embeddings ---
    "vocab_size": 129280,        # DeepSeek-V3 token vocabulary size
    "hidden_size": 7168,         # Transformer hidden dimension
    "tie_word_embeddings": False, # Whether lm_head shares embed_tokens weights

    # --- Layers ---
    "num_hidden_layers": 61,     # Total decoder layers (3 dense + 58 MoE)
    "intermediate_size": 18432,  # Dense MLP intermediate size (for layers 0-2)

    # --- Attention ---
    "num_attention_heads": 128,  # Number of query heads
    "num_key_value_heads": 128,  # MLA uses same count (MLA compresses differently from GQA)
    "attention_bias": False,     # No bias on Q/K/V/O projections
    "attention_dropout": 0.0,    # Attention dropout rate

    # --- MLA (Multi-head Latent Attention) dimensions ---
    # Paper ref: arXiv 2412.19437, Section 2.1 -- "Multi-head Latent Attention"
    "q_lora_rank": 1536,         # Query compression bottleneck (7168 -> 1536 -> 128*192)
    "kv_lora_rank": 512,         # KV compression bottleneck (7168 -> 512 -> 128*(128+128))
    "qk_rope_head_dim": 64,     # RoPE-applied portion of each head
    "qk_nope_head_dim": 128,    # Non-RoPE portion of each head
    "qk_head_dim": 192,         # Total = qk_nope_head_dim + qk_rope_head_dim = 128 + 64
    "v_head_dim": 128,           # Value head dimension

    # --- MoE (Mixture of Experts) ---
    # Paper ref: arXiv 2412.19437, Section 2.2 -- "DeepSeekMoE"
    "n_routed_experts": 256,     # Total expert count
    "n_shared_experts": 1,       # Always-active shared expert count
    "num_experts_per_tok": 8,    # Top-k experts selected per token
    "moe_intermediate_size": 2048, # Per-expert intermediate dimension
    "routed_scaling_factor": 2.5,  # Scale factor applied after expert weighted sum
    "n_group": 8,                # Number of expert groups for group-based selection
    "topk_group": 4,             # Number of groups selected before within-group topk
    "norm_topk_prob": True,      # Normalize routing weights after selection
    "scoring_func": "sigmoid",   # Sigmoid activation for routing (not softmax)
    "topk_method": "noaux_tc",   # Auxiliary-loss-free load balancing with bias correction
    "first_k_dense_replace": 3,  # First 3 layers use dense MLP instead of MoE
    "moe_layer_freq": 1,         # MoE layer frequency (every layer after first_k_dense_replace)
    "ep_size": 1,                # Expert parallelism size

    # --- Activation & normalization ---
    "hidden_act": "silu",       # SwiGLU uses SiLU (sigmoid linear unit)
    "rms_norm_eps": 1e-6,       # RMSNorm epsilon (NOTE: 1e-6 for DeepSeek-V3, 1e-5 for GLM5)

    # --- Positional encoding (YaRN RoPE) ---
    # Paper ref: arXiv 2412.19437, Section 3.5 -- "Long Context Extension"
    # DeepSeek-V3 extends context from 4096 to 163840 using YaRN.
    "max_position_embeddings": 163840,  # Maximum context length
    "rope_theta": 10000.0,              # RoPE base frequency
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.707,
    },

    # --- Multi-Token Prediction ---
    # Paper ref: arXiv 2412.19437, Section 2.3 -- "Multi-Token Prediction"
    # DeepSeek-V3 uses 1 MTP layer. GLM5 uses 3 shared MTP layers.
    "num_nextn_predict_layers": 1,

    # --- Weight initialization ---
    "initializer_range": 0.02,  # Std dev for weight initialization

    # --- Special tokens ---
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,

    # --- Cache ---
    "use_cache": True,

    # --- MLP layer types: first 3 dense, remaining 58 sparse (MoE) ---
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 58,
}


def load_config_from_hf(checkpoint_dir: str) -> dict:
    """Read config.json from a HuggingFace checkpoint directory.

    Maps HF config keys to our standalone config dict format.
    Falls back to DEEPSEEK_V3_CONFIG defaults for any missing keys.

    Args:
        checkpoint_dir: Path to directory containing config.json

    Returns:
        config: Standalone config dict ready for model instantiation.
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
        "scoring_func", "topk_method", "first_k_dense_replace", "moe_layer_freq",
        "ep_size", "num_nextn_predict_layers",
    ]
    for key in direct_keys:
        if key in hf_config:
            config[key] = hf_config[key]

    # Computed: total QK head dim = nope + rope
    config["qk_head_dim"] = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]

    # RoPE theta
    if "rope_theta" in hf_config:
        config["rope_theta"] = hf_config["rope_theta"]

    # YaRN RoPE scaling
    if "rope_scaling" in hf_config and hf_config["rope_scaling"] is not None:
        config["rope_scaling"] = hf_config["rope_scaling"]

    # MLP layer types from first_k_dense_replace + moe_layer_freq
    first_k = config.get("first_k_dense_replace", 3)
    n = config["num_hidden_layers"]
    config["mlp_layer_types"] = ["dense"] * first_k + ["sparse"] * (n - first_k)

    return config
