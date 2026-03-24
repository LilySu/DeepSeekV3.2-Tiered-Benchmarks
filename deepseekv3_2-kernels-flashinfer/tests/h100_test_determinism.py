"""H100 Category 7: Deterministic Execution for DeepSeek-V3.

Non-deterministic CUDA topk or MoE routing can cause performance degradation
during RL training. Must guarantee bit-identical decode outputs across runs.

Requirements: CUDA GPU (any).
Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 3.1
"""

import sys
import torch
from .conftest import make_cfg, skip_no_cuda


@skip_no_cuda
def h100_test_deterministic_topk():
    """torch.topk must produce identical results across 10 calls."""
    print("\n[H100-Det-1] Deterministic topk")
    device = "cuda"
    torch.manual_seed(42)
    scores = torch.randn(32, 256, device=device)

    results = []
    for _ in range(10):
        idx = scores.topk(8, dim=-1).indices
        results.append(idx.clone())

    ok = all(torch.equal(results[0], r) for r in results[1:])
    if ok:
        print("  PASS topk is bit-identical across 10 calls")
    else:
        mismatches = sum(not torch.equal(results[0], r) for r in results[1:])
        print(f"  FAIL topk produced {mismatches}/9 different results")
    return ok


@skip_no_cuda
def h100_test_deterministic_full_decode():
    """Full decode: 10 tokens from same seed must be bit-identical across 3 runs."""
    print("\n[H100-Det-2] Deterministic full decode (3 runs x 10 tokens)")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    device = "cuda"

    def run_decode():
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = kernel_model.DeepSeekV3ForCausalLM(cfg).to(device).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
        tokens = []
        with torch.no_grad():
            _, logits, cache = model(input_ids=input_ids, use_cache=True)
            for _ in range(10):
                next_token = logits[:, -1:, :].argmax(dim=-1)
                tokens.append(next_token.item())
                _, logits, cache = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        return tokens

    runs = [run_decode() for _ in range(3)]
    ok = runs[0] == runs[1] == runs[2]
    if ok:
        print(f"  PASS 3 runs produced identical tokens: {runs[0]}")
    else:
        print(f"  FAIL runs differ:")
        for i, r in enumerate(runs):
            print(f"    run {i}: {r}")
    return ok


@skip_no_cuda
def h100_test_deterministic_moe_routing():
    """MoE group-based routing must be deterministic across calls."""
    print("\n[H100-Det-3] Deterministic MoE group routing")
    from importlib import import_module
    router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")

    device = "cuda"
    torch.manual_seed(42)
    logits = torch.randn(64, 256, device=device)
    bias = torch.randn(256, device=device)

    results = []
    for _ in range(10):
        idx, wts = router.sigmoid_topk_route(
            logits.clone(), bias, top_k=8, n_group=8, topk_group=4,
        )
        results.append((idx.clone(), wts.clone()))

    ok = all(
        torch.equal(results[0][0], r[0]) and torch.equal(results[0][1], r[1])
        for r in results[1:]
    )
    if ok:
        print("  PASS MoE routing is bit-identical across 10 calls")
    else:
        print("  FAIL MoE routing produced different results")
    return ok
