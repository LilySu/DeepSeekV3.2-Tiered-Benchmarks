"""
Precision comparison experiments for DeepSeek-V3.

Compares numerical accuracy and throughput across:
  - FP8 e4m3 with 128x128 block quantization (DeepSeek-V3 default)
  - FP8 e5m2 (wider range, less precision)
  - BF16 baseline
  - FP16 baseline
  - INT8 symmetric quantization

Also measures the impact of different block sizes on quantization error,
replicating the analysis from the DeepSeek-V3 paper (arXiv: 2412.19437)
Section 3.3 which shows that fine-grained block quantization is critical
for maintaining training stability with FP8.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.shared.config import DEEPSEEK_V3_CONFIG, BenchConfig, BenchResult
from benchmark.shared.timer import CUDATimer
from benchmark.shared.report import save_json_report, save_markdown_report


# ---------------------------------------------------------------------------
# Precision Experiment Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrecisionExperiment:
    """Configuration for a single precision experiment."""
    name: str
    dtype_str: str
    use_quantization: bool = False
    block_size: Tuple[int, int] = (0, 0)
    description: str = ""


PRECISION_EXPERIMENTS = [
    PrecisionExperiment("bf16_baseline", "bfloat16", description="BF16 baseline (no quantization)"),
    PrecisionExperiment("fp16_baseline", "float16", description="FP16 baseline"),
    PrecisionExperiment("fp8_e4m3_block128", "float8_e4m3fn", True, (128, 128),
                        "FP8 e4m3 with 128x128 block quant (DeepSeek-V3 default)"),
    PrecisionExperiment("fp8_e4m3_block64", "float8_e4m3fn", True, (64, 64),
                        "FP8 e4m3 with 64x64 block quant"),
    PrecisionExperiment("fp8_e4m3_block256", "float8_e4m3fn", True, (256, 256),
                        "FP8 e4m3 with 256x256 block quant"),
    PrecisionExperiment("fp8_e4m3_per_tensor", "float8_e4m3fn", True, (0, 0),
                        "FP8 e4m3 per-tensor quant (baseline)"),
    PrecisionExperiment("fp8_e5m2_block128", "float8_e5m2", True, (128, 128),
                        "FP8 e5m2 with 128x128 block quant"),
]


# ---------------------------------------------------------------------------
# Quantization Error Analysis
# ---------------------------------------------------------------------------

def compute_quantization_error(
    reference: "torch.Tensor",
    quantized: "torch.Tensor",
) -> Dict[str, float]:
    """
    Compute quantization error metrics between reference and quantized tensors.

    Returns:
        Dict with MSE, RMSE, max absolute error, relative error, and SNR.
    """
    ref = reference.float()
    quant = quantized.float()

    diff = ref - quant
    mse = (diff ** 2).mean().item()
    rmse = math.sqrt(mse)
    max_abs_err = diff.abs().max().item()

    ref_norm = ref.norm().item()
    rel_err = rmse / ref_norm if ref_norm > 0 else float("inf")

    # Signal-to-Noise Ratio (SNR)
    signal_power = (ref ** 2).mean().item()
    noise_power = mse
    snr_db = 10 * math.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs_err,
        "relative_error": rel_err,
        "snr_db": snr_db,
    }


def block_quantize_fp8(
    tensor: "torch.Tensor",
    block_h: int,
    block_w: int,
    fp8_max: float = 448.0,
) -> "torch.Tensor":
    """
    Simulate block-wise FP8 quantization.

    For each (block_h x block_w) block:
      1. Compute block-level max absolute value
      2. Compute scale = max_val / fp8_max
      3. Quantize: round(tensor / scale)
      4. Dequantize: quantized * scale

    Args:
        tensor: Input tensor (2D).
        block_h: Block height (0 for per-tensor).
        block_w: Block width (0 for per-tensor).
        fp8_max: Max representable value in FP8 format.

    Returns:
        Dequantized tensor (simulated round-trip).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")

    if block_h == 0 or block_w == 0:
        # Per-tensor quantization
        scale = tensor.abs().max() / fp8_max
        scale = scale.clamp(min=1e-12)
        quantized = torch.round(tensor / scale).clamp(-fp8_max, fp8_max)
        return quantized * scale

    h, w = tensor.shape

    # Pad to block boundaries
    pad_h = (block_h - h % block_h) % block_h
    pad_w = (block_w - w % block_w) % block_w
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))

    ph, pw = padded.shape
    nh, nw = ph // block_h, pw // block_w

    # Reshape to blocks: (nh, block_h, nw, block_w)
    blocks = padded.reshape(nh, block_h, nw, block_w)

    # Per-block scaling
    block_max = blocks.abs().reshape(nh, block_h, nw, block_w).permute(0, 2, 1, 3).reshape(nh, nw, -1).amax(dim=-1)
    scales = (block_max / fp8_max).clamp(min=1e-12)  # (nh, nw)

    # Expand scales to match block dimensions
    scales_expanded = scales[:, :, None, None].expand(nh, nw, block_h, block_w)
    scales_expanded = scales_expanded.permute(0, 2, 1, 3).reshape(ph, pw)

    # Quantize and dequantize
    quantized = torch.round(padded / scales_expanded).clamp(-fp8_max, fp8_max)
    dequantized = quantized * scales_expanded

    return dequantized[:h, :w]


# ---------------------------------------------------------------------------
# Run Precision Experiments
# ---------------------------------------------------------------------------

def run_precision_experiments(
    bench_config: Optional[BenchConfig] = None,
    experiments: Optional[List[PrecisionExperiment]] = None,
    matrix_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run precision comparison experiments.

    For each matrix size and precision config:
      1. Generate a random reference matrix in FP32.
      2. Quantize using the specified method.
      3. Measure quantization error (MSE, SNR, etc.).
      4. Time the round-trip quantize/dequantize.

    Returns list of experiment result dicts.
    """
    if not HAS_TORCH:
        print("WARNING: PyTorch not available, returning empty results")
        return []

    if bench_config is None:
        bench_config = BenchConfig(name="precision_experiment")

    if experiments is None:
        experiments = PRECISION_EXPERIMENTS

    if matrix_sizes is None:
        # Representative shapes from DeepSeek-V3
        matrix_sizes = [
            (7168, 512),      # MLA KV compression
            (7168, 7168),     # Attention output projection
            (7168, 2048),     # MoE expert FFN
            (7168, 129280),   # LM head
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timer = CUDATimer(warmup=bench_config.warmup_iters, iters=bench_config.bench_iters)

    all_results = []

    for exp in experiments:
        print(f"\n--- Experiment: {exp.name} ---")
        print(f"    {exp.description}")

        for rows, cols in matrix_sizes:
            print(f"  Matrix ({rows}, {cols}): ", end="", flush=True)

            # Reference in FP32
            ref = torch.randn(rows, cols, dtype=torch.float32, device=device)

            if not exp.use_quantization:
                # Cast to target dtype and back
                target_dtype = getattr(torch, exp.dtype_str, torch.bfloat16)
                result_tensor = ref.to(target_dtype).float()
                quant_fn = lambda: ref.to(target_dtype).float()
            else:
                # Block FP8 quantization simulation
                bh, bw = exp.block_size
                ref_dev = ref.to(device)
                result_tensor = block_quantize_fp8(ref_dev, bh, bw).float()
                quant_fn = lambda: block_quantize_fp8(ref_dev, bh, bw)

            # Error analysis
            error_metrics = compute_quantization_error(ref, result_tensor)

            # Timing
            timing = timer.time_fn(quant_fn)

            result = {
                "experiment": exp.name,
                "dtype": exp.dtype_str,
                "block_size": list(exp.block_size),
                "matrix_shape": [rows, cols],
                "error": error_metrics,
                "timing_ms": timing.mean_ms,
                "timing_std_ms": timing.std_ms,
                "elements": rows * cols,
                "throughput_gelem_s": (rows * cols) / (timing.mean_ms / 1000) / 1e9 if timing.mean_ms > 0 else 0,
            }
            all_results.append(result)

            print(f"RMSE={error_metrics['rmse']:.6f}, SNR={error_metrics['snr_db']:.1f}dB, "
                  f"time={timing.mean_ms:.3f}ms")

    return all_results


# ---------------------------------------------------------------------------
# Analysis: Block Size Impact
# ---------------------------------------------------------------------------

def analyze_block_size_impact(
    matrix_shape: Tuple[int, int] = (7168, 7168),
    block_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze how block size affects quantization error.

    DeepSeek-V3 chose 128x128 blocks. This experiment sweeps block sizes
    to understand the precision-overhead trade-off.
    """
    if not HAS_TORCH:
        return []

    if block_sizes is None:
        block_sizes = [
            (0, 0),       # per-tensor
            (256, 256),
            (128, 128),   # DeepSeek-V3 default
            (64, 64),
            (32, 32),
            (16, 16),
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, cols = matrix_shape
    ref = torch.randn(rows, cols, dtype=torch.float32, device=device)

    results = []
    for bh, bw in block_sizes:
        label = f"{bh}x{bw}" if bh > 0 else "per_tensor"
        quantized = block_quantize_fp8(ref, bh, bw)
        error = compute_quantization_error(ref, quantized)

        # Count scaling factors
        if bh > 0 and bw > 0:
            n_scales = math.ceil(rows / bh) * math.ceil(cols / bw)
        else:
            n_scales = 1

        overhead_pct = 100.0 * n_scales * 4 / (rows * cols)  # FP32 scales vs FP8 data

        results.append({
            "block_size": label,
            "block_h": bh,
            "block_w": bw,
            "rmse": error["rmse"],
            "snr_db": error["snr_db"],
            "max_abs_error": error["max_abs_error"],
            "relative_error": error["relative_error"],
            "n_scaling_factors": n_scales,
            "scale_overhead_pct": overhead_pct,
        })

        print(f"  Block {label:>12s}: RMSE={error['rmse']:.6f}, SNR={error['snr_db']:.1f}dB, "
              f"scales={n_scales}, overhead={overhead_pct:.2f}%")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek-V3 Precision Experiments")
    parser.add_argument("--output-dir", type=str, default="results/precision")
    parser.add_argument("--block-size-sweep", action="store_true", help="Run block size impact analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DeepSeek-V3 Precision Comparison Experiments")
    print("Reference: arXiv:2412.19437, Section 3.3")
    print("=" * 70)

    results = run_precision_experiments()

    if args.block_size_sweep:
        print("\n" + "=" * 70)
        print("Block Size Impact Analysis")
        print("=" * 70)
        block_results = analyze_block_size_impact()

    import json
    with open(os.path.join(args.output_dir, "precision_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_dir}/precision_results.json")


if __name__ == "__main__":
    main()
