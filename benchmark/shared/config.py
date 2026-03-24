"""
DeepSeek-V3 671B configuration constants and benchmark dataclasses.

All model dimensions sourced from the DeepSeek-V3 technical report (arXiv: 2412.19437).
Hardware specs reference NVIDIA H100 SXM5 (80 GB HBM3).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import time


# ---------------------------------------------------------------------------
# DeepSeek-V3 671B model configuration
# ---------------------------------------------------------------------------

DEEPSEEK_V3_CONFIG: Dict[str, Any] = {
    # --- Transformer backbone ---
    "hidden_size": 7168,
    "num_layers": 61,           # 3 dense layers + 58 MoE layers
    "num_dense_layers": 3,
    "num_moe_layers": 58,
    "vocab_size": 129280,

    # --- Multi-head Latent Attention (MLA) ---
    "num_heads": 128,
    "kv_lora_rank": 512,        # compressed KV dimension
    "qk_rope_head_dim": 64,     # RoPE-applied portion of Q/K
    "qk_nope_head_dim": 128,    # non-RoPE portion of Q/K
    "v_head_dim": 128,          # value head dimension
    "q_lora_rank": 1536,        # query compression rank (optional, used in some layers)

    # --- Mixture of Experts (MoE) ---
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,   # top-K experts per token
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,      # shared expert that always activates
    "n_group": 8,               # expert groups for grouped routing
    "topk_group": 4,            # top groups selected before expert selection
    "routed_scaling_factor": 2.5,  # auxiliary-loss-free load balancing factor
    "norm_topk_prob": True,     # normalize expert probabilities

    # --- Multi-Token Prediction (MTP) ---
    "mtp_layers": 1,            # number of MTP prediction heads

    # --- FP8 Quantization ---
    "fp8_format": "e4m3",
    "fp8_block_size": (128, 128),  # block-wise quantization granularity

    # --- Positional encoding ---
    "rope_theta": 10000.0,
    "max_position_embeddings": 163840,  # 160K with YaRN
    "rope_scaling_type": "yarn",
    "rope_scaling_factor": 40,
    "rope_scaling_original_max_position_embeddings": 4096,
    "rope_scaling_beta_fast": 32,
    "rope_scaling_beta_slow": 1,
    "rope_scaling_mscale": 0.707,
    "rope_scaling_mscale_all_dim": 0.707,

    # --- NOT applicable (no DSA) ---
    # DeepSeek-V3 does not use Differential State Attention.
    # DSA indexer parameters are omitted entirely.
}


# ---------------------------------------------------------------------------
# Hardware specifications -- NVIDIA H100 SXM5
# ---------------------------------------------------------------------------

H100_SPECS: Dict[str, Any] = {
    "name": "NVIDIA H100 SXM5 80GB",
    "fp16_tflops": 989.5,
    "bf16_tflops": 989.5,
    "fp8_tflops": 1979.0,       # with sparsity: 3958
    "int8_tops": 1979.0,
    "hbm_bandwidth_tb_s": 3.35, # TB/s
    "hbm_capacity_gb": 80,
    "l2_cache_mb": 50,
    "nvlink_bandwidth_gb_s": 900,
    "tdp_watts": 700,
    "sm_count": 132,
    "cuda_cores": 16896,
    "tensor_cores": 528,        # 4th gen
}


# ---------------------------------------------------------------------------
# Benchmark configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    """Configuration for a single benchmark run."""

    # Identification
    name: str = "unnamed"
    description: str = ""

    # Workload
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])

    # Model overrides (defaults to full DeepSeek-V3)
    hidden_size: int = 7168
    num_heads: int = 128
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    n_group: int = 8
    topk_group: int = 4
    num_layers: int = 61
    vocab_size: int = 129280

    # Precision
    dtype: str = "bfloat16"
    use_fp8: bool = False
    fp8_block_size: tuple = (128, 128)

    # Measurement
    warmup_iters: int = 10
    bench_iters: int = 100
    num_bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Hardware target
    target_device: str = "H100"

    # Output
    output_dir: str = "results"
    save_traces: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "BenchConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Result of a single benchmark measurement."""

    # Identification
    config_name: str = ""
    component: str = ""         # e.g. "mla", "moe", "mtp", "e2e"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    # Workload parameters
    batch_size: int = 0
    seq_len: int = 0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Timing (milliseconds)
    mean_ms: float = 0.0
    std_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    ci_lower_ms: float = 0.0
    ci_upper_ms: float = 0.0

    # Throughput
    tokens_per_sec: float = 0.0
    tflops_achieved: float = 0.0
    mfu: float = 0.0            # model FLOPs utilization
    bandwidth_utilization: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary_line(self) -> str:
        return (
            f"[{self.component:>6s}] bs={self.batch_size:<4d} seq={self.seq_len:<5d} "
            f"mean={self.mean_ms:8.3f}ms  MFU={self.mfu:5.1f}%  "
            f"toks/s={self.tokens_per_sec:12,.0f}  peak_mem={self.peak_memory_mb:8.1f}MB"
        )
