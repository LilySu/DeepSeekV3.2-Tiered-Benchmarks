"""DeepSeek-V3 model configuration as a plain Python dict.

Architecture reference: DeepSeek-V3 Technical Report (arXiv 2412.19437)
Config values from: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json

This is the LARGEST variant: 671B total parameters, ~37B active per token.
61 transformer layers (3 dense + 58 MoE), MLA attention, 256 routed experts,
auxiliary-loss-free load balancing, multi-token prediction (1 MTP layer).
"""

import json
import os


DEEPSEEK_V3_CONFIG = {
    # Vocabulary & embeddings
    "vocab_size": 129280,
    "hidden_size": 7168,
    "tie_word_embeddings": False,
    # Layers
    "num_hidden_layers": 61,
    "intermediate_size": 18432,
    # Attention
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "attention_bias": False,
    "attention_dropout": 0.0,
    # MLA (Multi-head Latent Attention) dimensions
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "qk_head_dim": 192,  # qk_nope_head_dim + qk_rope_head_dim
    "v_head_dim": 128,
    # MoE (Mixture of Experts)
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.5,
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
    "ep_size": 1,
    # Activation & normalization
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    # Positional encoding (YaRN RoPE)
    "max_position_embeddings": 163840,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 0.707,
    },
    # Multi-Token Prediction
    "num_nextn_predict_layers": 1,
    # Weight initialization
    "initializer_range": 0.02,
    # Special tokens
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,
    # Cache
    "use_cache": True,
    # MLP layer pattern: first 3 dense, rest sparse (MoE)
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 58,
}


def load_config_from_hf(checkpoint_dir: str) -> dict:
    """Read config.json from a HuggingFace checkpoint and return a standalone config dict."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    config = dict(DEEPSEEK_V3_CONFIG)

    # Direct mappings (keys that match between HF and standalone)
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

    # Computed fields
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
