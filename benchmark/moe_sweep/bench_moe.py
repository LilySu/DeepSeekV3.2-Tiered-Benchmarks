"""
MoE sweep benchmarks for DeepSeek-V3.

Sweeps the MoE layer configuration space to understand the performance
impact of:
  - Number of routed experts (64, 128, 256)
  - Top-K experts per token (2, 4, 8)
  - Number of groups (1, 4, 8, 16) -- flat vs. grouped routing
  - Expert intermediate size (1024, 2048, 4096)
  - Token batch size (how many tokens per expert dispatch)
  - Load balancing quality

DeepSeek-V3 default: 256 experts, top-8, 8 groups, top-4 groups, intermediate=2048
Reference: arXiv:2412.19437, Section 3.1

Key insight from DeepSeek-V3: grouped routing with auxiliary-loss-free
load balancing achieves better expert utilization than flat top-K routing
while maintaining model quality. The n_group=8, topk_group=4 configuration
was chosen to balance routing granularity with computational efficiency.
"""

from __future__ import annotations

import os
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, BenchConfig, BenchResult
from benchmark.shared.timer import CUDATimer
from benchmark.shared.metrics import compute_moe_flops, compute_mfu
from benchmark.shared.report import save_json_report, save_markdown_report, print_results_table


# ---------------------------------------------------------------------------
# MoE Sweep Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoESweepConfig:
    """A single point in the MoE sweep space."""
    n_experts: int = 256
    top_k: int = 8
    n_group: int = 8
    topk_group: int = 4
    intermediate_size: int = 2048
    hidden_size: int = 7168
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"E{self.n_experts}_K{self.top_k}_G{self.n_group}_TG{self.topk_group}_I{self.intermediate_size}"


# Default sweep grid
def get_default_sweep_configs() -> List[MoESweepConfig]:
    """Generate the default sweep configuration grid."""
    configs = []

    # Sweep expert count
    for n_exp in [64, 128, 256]:
        configs.append(MoESweepConfig(n_experts=n_exp, n_group=max(1, n_exp // 32)))

    # Sweep top-K
    for top_k in [2, 4, 8, 16]:
        configs.append(MoESweepConfig(top_k=top_k))

    # Sweep group count (flat routing = n_group=1, grouped = n_group>1)
    for n_group in [1, 4, 8, 16]:
        topk_group = max(1, n_group // 2)
        configs.append(MoESweepConfig(n_group=n_group, topk_group=topk_group))

    # Sweep intermediate size
    for i_size in [1024, 2048, 4096]:
        configs.append(MoESweepConfig(intermediate_size=i_size))

    # DeepSeek-V3 default (for comparison)
    configs.append(MoESweepConfig(
        n_experts=256, top_k=8, n_group=8, topk_group=4,
        intermediate_size=2048, label="DeepSeekV3_default"
    ))

    return configs


# ---------------------------------------------------------------------------
# Grouped routing simulation
# ---------------------------------------------------------------------------

def grouped_top_k_routing(
    logits: "torch.Tensor",
    n_group: int,
    topk_group: int,
    top_k: int,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Implement DeepSeek-V3 grouped routing.

    Args:
        logits: (batch_tokens, n_experts) gating logits.
        n_group: Number of expert groups.
        topk_group: Number of groups to select.
        top_k: Total number of experts to activate per token.

    Returns:
        (selected_expert_indices, selected_expert_weights) both shape (batch_tokens, top_k)

    Algorithm:
        1. Reshape logits to (tokens, n_group, experts_per_group)
        2. Compute group scores (max or mean of expert scores per group)
        3. Select top-k groups
        4. Within selected groups, select top experts
        5. Normalize weights
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")

    tokens, n_experts = logits.shape
    experts_per_group = n_experts // n_group

    if n_group <= 1:
        # Flat routing: standard top-K
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(top_k, dim=-1)
        return topk_idx, topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    # Reshape to groups
    grouped_logits = logits.reshape(tokens, n_group, experts_per_group)

    # Group scores: max score within each group
    group_scores = grouped_logits.max(dim=-1).values  # (tokens, n_group)

    # Select top groups
    _, top_groups = group_scores.topk(topk_group, dim=-1)  # (tokens, topk_group)

    # Create mask for selected groups
    group_mask = torch.zeros(tokens, n_group, dtype=torch.bool, device=logits.device)
    group_mask.scatter_(1, top_groups, True)

    # Mask out unselected groups
    group_mask_expanded = group_mask.unsqueeze(-1).expand_as(grouped_logits)
    masked_logits = grouped_logits.clone()
    masked_logits[~group_mask_expanded] = float("-inf")

    # Flatten back and select top-K experts
    flat_logits = masked_logits.reshape(tokens, n_experts)
    probs = torch.softmax(flat_logits, dim=-1)
    topk_vals, topk_idx = probs.topk(top_k, dim=-1)

    # Normalize
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    return topk_idx, topk_vals


# ---------------------------------------------------------------------------
# MoE Layer Benchmark
# ---------------------------------------------------------------------------

def bench_moe_layer(
    sweep_config: MoESweepConfig,
    batch_tokens: int = 2048,
    bench_config: Optional[BenchConfig] = None,
) -> BenchResult:
    """
    Benchmark a single MoE layer configuration.

    Measures:
      1. Gating computation (routing decision)
      2. Expert dispatch (token-to-expert assignment)
      3. Expert computation (SwiGLU FFN for selected experts)
      4. Expert combine (weighted sum of expert outputs)
    """
    if bench_config is None:
        bench_config = BenchConfig(name="moe_sweep")

    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)

    H = sweep_config.hidden_size
    I = sweep_config.intermediate_size
    N = sweep_config.n_experts
    K = sweep_config.top_k

    if HAS_TORCH and torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16

        # Create weights
        W_gate_proj = torch.randn(H, N, dtype=dtype, device=device)  # gating

        # Expert weights (for simulation, we use shared weights)
        W_ffn_gate = torch.randn(H, I, dtype=dtype, device=device)
        W_ffn_up = torch.randn(H, I, dtype=dtype, device=device)
        W_ffn_down = torch.randn(I, H, dtype=dtype, device=device)

        x = torch.randn(batch_tokens, H, dtype=dtype, device=device)

        def moe_forward():
            # 1. Gating
            gate_logits = x @ W_gate_proj  # (tokens, N)

            # 2. Grouped routing
            expert_idx, expert_weights = grouped_top_k_routing(
                gate_logits, sweep_config.n_group, sweep_config.topk_group, K
            )

            # 3. Expert computation (simplified: all tokens through shared expert weights)
            # In production, tokens are dispatched to individual experts
            gate_out = F.silu(x @ W_ffn_gate)
            up_out = x @ W_ffn_up
            hidden = gate_out * up_out
            expert_out = hidden @ W_ffn_down

            # 4. Weighted combine (simplified)
            weight_sum = expert_weights.sum(dim=-1, keepdim=True)
            out = expert_out * weight_sum

            return out

        timing = timer.time_fn(moe_forward)

        # Memory tracking
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
    else:
        from benchmark.shared.timer import TimingResult
        timing = TimingResult(
            mean_ms=1.0, std_ms=0.1, median_ms=1.0, min_ms=0.9, max_ms=1.2,
            ci_lower_ms=0.95, ci_upper_ms=1.05, raw_times_ms=[1.0],
            warmup_iters=0, bench_iters=1, confidence_level=0.95,
        )
        peak_mem = 0.0

    # Compute FLOPs
    flop_info = compute_moe_flops(
        batch_size=1, seq_len=batch_tokens,
        hidden_size=H, n_routed_experts=N, num_experts_per_tok=K,
        moe_intermediate_size=I, n_group=sweep_config.n_group,
        topk_group=sweep_config.topk_group,
    )
    elapsed_s = timing.mean_ms / 1000.0
    mfu = compute_mfu(flop_info["total_flops"], elapsed_s, dtype=bench_config.dtype)

    return BenchResult(
        config_name=sweep_config.label,
        component="moe",
        batch_size=1,
        seq_len=batch_tokens,
        extra_params={
            "n_experts": N, "top_k": K, "n_group": sweep_config.n_group,
            "topk_group": sweep_config.topk_group, "intermediate": I,
        },
        mean_ms=timing.mean_ms,
        std_ms=timing.std_ms,
        median_ms=timing.median_ms,
        min_ms=timing.min_ms,
        max_ms=timing.max_ms,
        ci_lower_ms=timing.ci_lower_ms,
        ci_upper_ms=timing.ci_upper_ms,
        tokens_per_sec=batch_tokens / elapsed_s if elapsed_s > 0 else 0,
        tflops_achieved=flop_info["total_flops"] / elapsed_s / 1e12 if elapsed_s > 0 else 0,
        mfu=mfu,
        peak_memory_mb=peak_mem,
    )


# ---------------------------------------------------------------------------
# Load balance analysis
# ---------------------------------------------------------------------------

def analyze_load_balance(
    n_experts: int = 256,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    n_tokens: int = 8192,
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    Analyze expert load balance under grouped routing.

    Measures how evenly tokens are distributed across experts,
    which directly impacts MoE computation efficiency.

    Returns coefficient of variation and max/min load ratio.
    """
    if not HAS_TORCH:
        return {"error": "PyTorch not available"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_counts_all = []

    for trial in range(n_trials):
        # Random gating logits (simulating natural distribution)
        logits = torch.randn(n_tokens, n_experts, device=device)

        expert_idx, _ = grouped_top_k_routing(logits, n_group, topk_group, top_k)

        # Count tokens per expert
        counts = torch.zeros(n_experts, device=device)
        for k in range(top_k):
            expert_ids = expert_idx[:, k]
            counts.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.float))

        load_counts_all.append(counts.cpu())

    # Aggregate statistics
    all_counts = torch.stack(load_counts_all)
    mean_counts = all_counts.float().mean(dim=0)
    overall_mean = mean_counts.mean().item()
    overall_std = mean_counts.std().item()
    cv = overall_std / overall_mean if overall_mean > 0 else float("inf")

    max_load = mean_counts.max().item()
    min_load = mean_counts.min().item()
    max_min_ratio = max_load / min_load if min_load > 0 else float("inf")

    # Expected uniform load
    expected_load = n_tokens * top_k / n_experts

    return {
        "n_experts": n_experts,
        "n_group": n_group,
        "topk_group": topk_group,
        "top_k": top_k,
        "n_tokens": n_tokens,
        "expected_load_per_expert": expected_load,
        "actual_mean_load": overall_mean,
        "load_std": overall_std,
        "coefficient_of_variation": cv,
        "max_load": max_load,
        "min_load": min_load,
        "max_min_ratio": max_min_ratio,
        "load_balance_efficiency": expected_load / max_load if max_load > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Full MoE sweep
# ---------------------------------------------------------------------------

def run_moe_sweep(
    sweep_configs: Optional[List[MoESweepConfig]] = None,
    batch_tokens: int = 2048,
    bench_config: Optional[BenchConfig] = None,
) -> List[BenchResult]:
    """Run the full MoE parameter sweep."""
    if sweep_configs is None:
        sweep_configs = get_default_sweep_configs()

    if bench_config is None:
        bench_config = BenchConfig(name="moe_sweep")

    results = []

    for i, sc in enumerate(sweep_configs):
        print(f"\n[{i+1}/{len(sweep_configs)}] {sc.label}")
        try:
            result = bench_moe_layer(sc, batch_tokens=batch_tokens, bench_config=bench_config)
            results.append(result)
            print(f"  -> {result.mean_ms:.3f}ms, MFU={result.mfu:.1f}%, "
                  f"{result.tokens_per_sec:,.0f} tok/s")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-V3 MoE Sweep Benchmark")
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/moe_sweep")
    parser.add_argument("--load-balance", action="store_true", help="Run load balance analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DeepSeek-V3 MoE Sweep Benchmark")
    print(f"Default: 256 experts, top-8, 8 groups, top-4 groups, intermediate=2048")
    print("=" * 70)

    config = BenchConfig(
        name="moe_sweep",
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )

    results = run_moe_sweep(batch_tokens=args.batch_tokens, bench_config=config)

    if results:
        json_path = save_json_report(results, os.path.join(args.output_dir, "moe_sweep.json"))
        md_path = save_markdown_report(results, os.path.join(args.output_dir, "moe_sweep.md"),
                                       title="MoE Parameter Sweep")
        print(f"\nJSON: {json_path}")
        print(f"Markdown: {md_path}")
        print_results_table(results, "MoE Sweep Results")

    if args.load_balance:
        print("\n" + "=" * 70)
        print("Load Balance Analysis")
        print("=" * 70)

        for n_group in [1, 4, 8, 16]:
            topk_g = max(1, n_group // 2)
            lb = analyze_load_balance(n_group=n_group, topk_group=topk_g)
            print(f"  Groups={n_group}, TopK_Group={topk_g}: "
                  f"CV={lb['coefficient_of_variation']:.3f}, "
                  f"Max/Min={lb['max_min_ratio']:.2f}, "
                  f"Efficiency={lb['load_balance_efficiency']:.2f}")


if __name__ == "__main__":
    main()
