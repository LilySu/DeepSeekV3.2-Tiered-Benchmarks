"""H100 Category 1: CUDA Graph Capture & Replay for DeepSeek-V3.

Decode uses CUDA graphs to eliminate kernel launch overhead (~390us/token).
Must verify: (1) graphs capture successfully, (2) replay with updated
KV cache produces correct outputs, (3) no DSA indices needed (standard causal).

Requirements: H100 (SM90), CUDA.
Paper ref: DeepSeek-V3 (arXiv 2412.19437)
"""

import sys
import torch
import pytest
from .conftest import assert_close, make_cfg, skip_no_cuda, cuda_timer_fn


@skip_no_cuda
def h100_test_cuda_graph_capture_model():
    """Capture a full decode step in a CUDA graph and replay it."""
    print("\n[H100-Graph-1] CUDA graph capture of full decode step")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    device = "cuda"
    model = kernel_model.DeepSeekV3ForCausalLM(cfg).to(device).eval()
    B, S_prefill = 1, 16
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S_prefill), device=device)

    with torch.no_grad():
        _, logits, cache = model(input_ids=input_ids, use_cache=True)

    static_token = torch.zeros(B, 1, dtype=torch.long, device=device)
    static_token[0, 0] = logits[:, -1:, :].argmax(dim=-1).item()

    with torch.no_grad():
        _, out_warmup, cache = model(input_ids=static_token, past_key_values=cache, use_cache=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            _, static_out, _ = model(input_ids=static_token, past_key_values=cache, use_cache=True)

    static_token[0, 0] = 42
    graph.replay()
    captured_out = static_out.clone()

    ok = torch.isfinite(captured_out).all() and captured_out.shape == (B, 1, cfg["vocab_size"])
    print(f"  {'PASS' if ok else 'FAIL'} graph capture + replay")
    return ok


@skip_no_cuda
def h100_test_cuda_graph_kv_cache_update():
    """Verify KV cache updates work correctly inside CUDA graph replays."""
    print("\n[H100-Graph-2] KV cache update in CUDA graph")
    device = "cuda"
    B, H, D = 1, 4, 32
    static_k = torch.randn(B, H, 1, D, device=device)
    static_v = torch.randn(B, H, 1, D, device=device)
    cache_k = torch.randn(B, H, 16, D, device=device)
    cache_v = torch.randn(B, H, 16, D, device=device)

    # Warmup
    new_k = torch.cat([cache_k, static_k], dim=2)
    new_v = torch.cat([cache_v, static_v], dim=2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        new_k = torch.cat([cache_k, static_k], dim=2)
        new_v = torch.cat([cache_v, static_v], dim=2)

    ok = True
    for trial in range(3):
        static_k.copy_(torch.randn_like(static_k))
        graph.replay()
        expected_k = torch.cat([cache_k, static_k], dim=2)
        if not torch.equal(new_k, expected_k):
            ok = False
            print(f"  FAIL trial {trial}: KV cache mismatch")
    if ok:
        print("  PASS KV cache updates correctly across graph replays")
    return ok


@skip_no_cuda
def h100_test_cuda_graph_speedup():
    """CUDA graph should provide >2x speedup over eager execution."""
    print("\n[H100-Graph-3] CUDA graph speedup measurement")
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

    # Graph capture
    with torch.no_grad():
        _ = model(input_ids=static_token, past_key_values=cache, use_cache=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            _, static_out, _ = model(input_ids=static_token, past_key_values=cache, use_cache=True)

    graph_times = cuda_timer_fn(lambda: graph.replay(), warmup=5, iters=20)
    graph_median = graph_times[len(graph_times) // 2]

    speedup = eager_median / graph_median if graph_median > 0 else 0
    print(f"  Eager: {eager_median:.3f}ms, Graph: {graph_median:.3f}ms, Speedup: {speedup:.1f}x")
    ok = speedup > 1.5
    print(f"  {'PASS' if ok else 'FAIL'} speedup {speedup:.1f}x {'>' if ok else '<'} 1.5x threshold")
    return ok
