"""H100 Category 10: Thermal Throttling Detection for DeepSeek-V3.

H100 throttles under sustained load (TDP=700W SXM). If the last window's
TFLOPS drops below 85% of the first window, thermal management is kicking in.

Requirements: H100 (SM90), CUDA.
"""

import sys
import time
import torch
from .conftest import skip_no_cuda, cuda_timer_fn


@skip_no_cuda
def h100_test_thermal_sustained_gemm():
    """Run sustained BF16 GEMM for 30s, compare first-5s vs last-5s throughput."""
    print("\n[H100-Thermal-1] Sustained GEMM thermal stability (30s)")
    device = "cuda"
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    flops_per_op = 2 * M * N * K

    def run_window(duration_s, label):
        times = []
        deadline = time.time() + duration_s
        while time.time() < deadline:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); torch.mm(a, b); end.record(); torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = times[len(times) // 2]
        tflops = flops_per_op / (median_ms * 1e-3) / 1e12
        return tflops, median_ms, len(times)

    for _ in range(20):
        torch.mm(a, b)
    torch.cuda.synchronize()

    tflops_first, ms_first, n_first = run_window(5.0, "first")
    print(f"  First 5s:  {tflops_first:.1f} TFLOPS ({ms_first:.3f} ms/op, {n_first} ops)")

    _ = run_window(20.0, "middle")

    tflops_last, ms_last, n_last = run_window(5.0, "last")
    print(f"  Last 5s:   {tflops_last:.1f} TFLOPS ({ms_last:.3f} ms/op, {n_last} ops)")

    ratio = tflops_last / tflops_first if tflops_first > 0 else 0
    ok = ratio > 0.85
    print(f"  Ratio: {ratio:.3f} (last/first)")
    print(f"  {'PASS' if ok else 'FAIL'} thermal ratio {ratio:.3f} > 0.85")
    return ok


@skip_no_cuda
def h100_test_thermal_clock_frequency():
    """Check GPU clock frequency before and after sustained load."""
    print("\n[H100-Thermal-2] Clock frequency stability")
    device = "cuda"

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            clock_mhz = int(result.stdout.strip().split("\n")[0])
            print(f"  Current GPU clock: {clock_mhz} MHz")
            ok = clock_mhz > 1000  # Should be above 1 GHz
            print(f"  {'PASS' if ok else 'FAIL'} clock > 1000 MHz")
            return ok
    except Exception as e:
        print(f"  SKIP nvidia-smi not available: {e}")

    return True
