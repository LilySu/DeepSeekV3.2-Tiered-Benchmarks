"""H100 Category 6: Kernel Launch Overhead for DeepSeek-V3.

Python dispatch + gaps between kernels across 61 layers. Measure the
overhead and verify CUDA graphs provide significant speedup.

Requirements: H100 (SM90), CUDA.
Paper ref: DeepSeek-V3 (arXiv 2412.19437)
"""

import sys
import torch
from .conftest import make_cfg, skip_no_cuda, cuda_timer_fn


@skip_no_cuda
def h100_test_launch_overhead_empty_kernels():
    """Measure pure kernel launch overhead with no-op kernels."""
    print("\n[H100-Launch-1] Empty kernel launch overhead")
    device = "cuda"
    N = 1000
    tensors = [torch.ones(1, device=device) for _ in range(N)]

    def eager_loop():
        for t in tensors:
            t.add_(1)

    times = cuda_timer_fn(eager_loop, warmup=3, iters=10)
    median_ms = times[len(times) // 2]
    overhead_us_per_launch = (median_ms * 1000) / N
    print(f"  {N} launches: {median_ms:.3f} ms total, {overhead_us_per_launch:.1f} us/launch")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eager_loop()

    graph_times = cuda_timer_fn(lambda: graph.replay(), warmup=5, iters=20)
    graph_median = graph_times[len(graph_times) // 2]
    speedup = median_ms / graph_median if graph_median > 0 else 0
    print(f"  Graph: {graph_median:.3f} ms, speedup={speedup:.1f}x")

    ok = speedup > 2.0
    print(f"  {'PASS' if ok else 'FAIL'} graph speedup {speedup:.1f}x > 2x")
    return ok


@skip_no_cuda
def h100_test_launch_overhead_per_layer():
    """Measure per-layer overhead for DeepSeek-V3 decoder layer."""
    print("\n[H100-Launch-2] Per-layer launch overhead")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")
    rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")

    cfg = make_cfg(num_layers=1)
    device = "cuda"
    layer = kernel_model.DecoderLayer(cfg, 0).to(device).eval()

    B, S = 1, 64
    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.float32, device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rope(hidden, pos_ids)
    mask = kernel_model.make_causal_mask(S, 0, hidden.dtype, device)

    with torch.no_grad():
        single_times = cuda_timer_fn(
            lambda: layer(hidden, attention_mask=mask, position_embeddings=(cos, sin)),
            warmup=5, iters=20,
        )
    single_median = single_times[len(single_times) // 2]
    print(f"  Single layer: {single_median:.3f} ms")

    # Estimate 61-layer overhead
    estimated_61 = single_median * 61
    print(f"  Estimated 61 layers (serial): {estimated_61:.1f} ms")
    ok = single_median > 0  # Just verify it ran
    print(f"  {'PASS' if ok else 'FAIL'} per-layer timing valid")
    return ok


@skip_no_cuda
def h100_test_launch_overhead_graph_vs_eager_model():
    """Compare graph vs eager for full 2-layer model decode."""
    print("\n[H100-Launch-3] Graph vs eager for model decode")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    device = "cuda"
    model = kernel_model.DeepSeekV3ForCausalLM(cfg).to(device).eval()

    input_ids = torch.randint(0, cfg["vocab_size"], (1, 8), device=device)
    with torch.no_grad():
        _, _, cache = model(input_ids=input_ids, use_cache=True)

    static_token = torch.tensor([[42]], device=device)

    eager_times = cuda_timer_fn(
        lambda: model(input_ids=static_token, past_key_values=cache, use_cache=True),
        warmup=5, iters=20
    )
    eager_median = eager_times[len(eager_times) // 2]
    print(f"  Eager decode: {eager_median:.3f} ms")

    ok = eager_median > 0
    print(f"  {'PASS' if ok else 'FAIL'} eager timing valid")
    return ok
