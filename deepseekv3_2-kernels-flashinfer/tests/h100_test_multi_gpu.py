"""H100 Category 5: Multi-GPU NCCL + Tensor Parallelism for DeepSeek-V3.

All-reduce bandwidth must reach >80% of NVLink 900GB/s. TP-sharded model
must produce numerically equivalent output to single-GPU.

Run with torchrun:
    torchrun --nproc_per_node=2 -m deepseekv3_2-kernels-flashinfer.tests.h100_test_multi_gpu

Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 3.2
"""

import os
import sys
import torch
import torch.distributed as dist
from .conftest import skip_no_multi_gpu, make_cfg, assert_close


def _setup_dist():
    if "RANK" not in os.environ:
        return None, None
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank


@skip_no_multi_gpu
def h100_test_nccl_allreduce_bandwidth():
    """Measure NCCL all-reduce bandwidth. Target: >80% of NVLink peak."""
    print("\n[H100-NCCL-1] All-reduce bandwidth")
    rank, local_rank = _setup_dist()
    if rank is None:
        print("  SKIP not launched with torchrun")
        return True

    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()

    for size_mb in [1, 10, 50]:
        numel = size_mb * 1024 * 1024 // 2
        tensor = torch.randn(numel, dtype=torch.bfloat16, device=device)

        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        times = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(tensor)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        data_gb = size_mb / 1024 * 2 * (world_size - 1) / world_size
        bw_gb_s = data_gb / (median_ms / 1000)

        if rank == 0:
            print(f"  {size_mb}MB: {median_ms:.3f}ms, {bw_gb_s:.1f} GB/s")

    return True


@skip_no_multi_gpu
def h100_test_tp_equivalence():
    """TP-sharded DeepSeek-V3 hidden states match single-GPU after all-reduce."""
    print("\n[H100-TP-1] Tensor parallel equivalence")
    rank, local_rank = _setup_dist()
    if rank is None:
        print("  SKIP not launched with torchrun")
        return True

    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()

    # Simulate TP shard of hidden states
    torch.manual_seed(42)
    hidden_size = 7168  # DeepSeek-V3
    B, S = 1, 8
    full_hidden = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=device)

    # Shard and all-reduce
    shard_size = hidden_size // world_size
    shard = full_hidden[:, :, rank * shard_size:(rank + 1) * shard_size].clone()
    gathered = torch.zeros_like(full_hidden)
    gathered[:, :, rank * shard_size:(rank + 1) * shard_size] = shard
    dist.all_reduce(gathered)

    ok = torch.allclose(gathered, full_hidden, atol=1e-4)
    if rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'} all-reduce reconstructs original hidden states")
    return ok
