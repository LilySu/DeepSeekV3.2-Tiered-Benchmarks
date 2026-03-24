"""
Pre-computed tuning configurations for DeepSeek-V3 MoE shapes.

These are the optimal kernel configurations for the specific matrix shapes
that appear in DeepSeek-V3's MoE layers on H100 GPUs.  They can be used
directly without running the auto-tuner at startup.

Key shapes for DeepSeek-V3 (per expert):
  - Gate/Up:  [T_e, 7168] x [7168, 2048]  (hidden -> intermediate)
  - Down:     [T_e, 2048] x [2048, 7168]  (intermediate -> hidden)

where T_e = tokens assigned to expert e (varies, avg ~= batch*seq*8/256).
"""

from __future__ import annotations

from typing import Dict

from .autotuning import TuningConfig, ProblemShape


# ---------------------------------------------------------------------------
# Pre-tuned configurations for H100 (SM90)
# ---------------------------------------------------------------------------

# These configurations were determined by offline auto-tuning on H100 SXM5
# for representative batch sizes.

_H100_CONFIGS: Dict[str, TuningConfig] = {
    # Gate/Up projection: small M (decode), K=7168, N=2048
    "gate_up_small": TuningConfig(
        block_m=32, block_n=64, block_k=64,
        num_warps=4, num_stages=4,
        tflops=0.0, time_ms=0.0,
    ),
    # Gate/Up projection: medium M (short prompt), K=7168, N=2048
    "gate_up_medium": TuningConfig(
        block_m=64, block_n=64, block_k=64,
        num_warps=4, num_stages=3,
        tflops=0.0, time_ms=0.0,
    ),
    # Gate/Up projection: large M (long prompt), K=7168, N=2048
    "gate_up_large": TuningConfig(
        block_m=128, block_n=64, block_k=64,
        num_warps=8, num_stages=3,
        tflops=0.0, time_ms=0.0,
    ),
    # Down projection: small M, K=2048, N=7168
    "down_small": TuningConfig(
        block_m=32, block_n=128, block_k=32,
        num_warps=4, num_stages=4,
        tflops=0.0, time_ms=0.0,
    ),
    # Down projection: medium M, K=2048, N=7168
    "down_medium": TuningConfig(
        block_m=64, block_n=128, block_k=32,
        num_warps=4, num_stages=3,
        tflops=0.0, time_ms=0.0,
    ),
    # Down projection: large M, K=2048, N=7168
    "down_large": TuningConfig(
        block_m=128, block_n=128, block_k=32,
        num_warps=8, num_stages=3,
        tflops=0.0, time_ms=0.0,
    ),
}

# DeepGEMM FP8 specific configs (these are used when DeepGEMM is available)
_DEEPGEMM_FP8_CONFIGS: Dict[str, TuningConfig] = {
    # DeepGEMM uses its own internal tuning, so these are hints only
    "fp8_gate_up": TuningConfig(
        block_m=128, block_n=128, block_k=64,
        num_warps=8, num_stages=4,
    ),
    "fp8_down": TuningConfig(
        block_m=128, block_n=128, block_k=64,
        num_warps=8, num_stages=4,
    ),
}


# ---------------------------------------------------------------------------
# Token-count thresholds for selecting configs
# ---------------------------------------------------------------------------

# Average tokens per expert at different batch configurations:
#   B=1, S=1 (decode): T_e ~= 8/256 ~= 0.03 -> effectively 1 token
#   B=1, S=2048: T_e ~= 2048*8/256 = 64
#   B=8, S=2048: T_e ~= 8*2048*8/256 = 512
#   B=32, S=2048: T_e ~= 32*2048*8/256 = 2048

SMALL_THRESHOLD = 32    # tokens per expert
LARGE_THRESHOLD = 256   # tokens per expert


def select_config(
    tokens_per_expert: int,
    projection_type: str = "gate_up",
    use_fp8: bool = True,
) -> TuningConfig:
    """Select the best pre-tuned config for a given workload.

    Parameters
    ----------
    tokens_per_expert : int
        Average number of tokens per expert.
    projection_type : str
        Either 'gate_up' or 'down'.
    use_fp8 : bool
        Whether FP8 (DeepGEMM) is being used.

    Returns
    -------
    config : TuningConfig
    """
    if use_fp8:
        key = f"fp8_{projection_type}"
        if key in _DEEPGEMM_FP8_CONFIGS:
            return _DEEPGEMM_FP8_CONFIGS[key]

    if tokens_per_expert < SMALL_THRESHOLD:
        size = "small"
    elif tokens_per_expert < LARGE_THRESHOLD:
        size = "medium"
    else:
        size = "large"

    key = f"{projection_type}_{size}"
    return _H100_CONFIGS.get(key, TuningConfig())


# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------

DEEPSEEK_V3_TUNING_CONFIGS = {
    "h100_configs": _H100_CONFIGS,
    "deepgemm_fp8_configs": _DEEPGEMM_FP8_CONFIGS,
}

DEEPSEEK_V3_PROBLEM_SHAPES = {
    "gate_up": ProblemShape(M=64, K=7168, N=2048, num_groups=256),
    "down": ProblemShape(M=64, K=2048, N=7168, num_groups=256),
}
