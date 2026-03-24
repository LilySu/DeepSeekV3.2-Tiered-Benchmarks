"""CPU tests for individual DeepSeek-V3 components: KV cache, MoE dispatch,
FP8 layout, autoregressive decode, group routing, gradients, edge cases.

Adapted from GLM-5 test_components.py with DeepSeek-V3 specific changes:
  - No DSA mask tests (DSA not applicable)
  - Group routing with n_group=4, topk_group=2 (tiny config)
  - YaRN RoPE verification
  - MTP layer tests

Run: python3 -m deepseekv3_2-kernels-flashinfer.tests.test_components
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_cfg, jaccard


# -- Test 1: KV cache -------------------------------------------------------

def test_kvcache_multistep():
    print("\n[Test 1a] KVCache multi-step")
    from importlib import import_module
    cache_mod = import_module("deepseekv3_2-kernels-flashinfer.cache")
    cache = cache_mod.KVCache(num_layers=2)
    all_k, all_v = [], []
    for step in range(5):
        k = torch.randn(2, 4, 1, 32) + step
        v = torch.randn(2, 4, 1, 32) + step
        all_k.append(k); all_v.append(v)
        full_k, full_v = cache.update(k, v, 0)
    ok = assert_close("keys", full_k, torch.cat(all_k, dim=2), atol=0)
    ok = assert_close("vals", full_v, torch.cat(all_v, dim=2), atol=0) and ok
    ok = (cache.get_seq_length(0) == 5) and ok
    print(f"  {'PASS' if ok else 'FAIL'} seq_length=5")
    return ok

def test_kvcache_reset():
    print("\n[Test 1b] KVCache reset")
    from importlib import import_module
    cache_mod = import_module("deepseekv3_2-kernels-flashinfer.cache")
    cache = cache_mod.KVCache(2)
    cache.update(torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32), 0)
    cache.reset()
    ok = cache.get_seq_length(0) == 0
    print(f"  {'PASS' if ok else 'FAIL'} reset")
    return ok

def test_paged_kvcache():
    print("\n[Test 1c] FlashInferPagedKVCache allocation")
    from importlib import import_module
    cache_mod = import_module("deepseekv3_2-kernels-flashinfer.cache")
    cache = cache_mod.FlashInferPagedKVCache(1, num_pages=8, page_size=4, device="cpu")
    pages = [cache.allocate_page() for _ in range(5)]
    ok = len(set(pages)) == 5
    cache.free_page(pages[0])
    try:
        for _ in range(10): cache.allocate_page()
        ok = False; print("  FAIL should exhaust")
    except RuntimeError:
        print("  PASS exhaustion raises")
    # Check separate ckv/kpe pools with DeepSeek-V3 dims
    ckv = cache.get_ckv_cache(0)
    kpe = cache.get_kpe_cache(0)
    ok = ok and ckv.shape[-1] == 512 and kpe.shape[-1] == 64
    print(f"  {'PASS' if ok else 'FAIL'} ckv shape {ckv.shape[-1]}, kpe shape {kpe.shape[-1]}")
    return ok


# -- Test 2: MoE expert dispatch --------------------------------------------

def test_expert_dispatch_single():
    print("\n[Test 2a] Expert dispatch: single expert")
    from importlib import import_module
    moe_gemm = import_module("deepseekv3_2-kernels-flashinfer.moe_grouped_gemm")
    torch.manual_seed(42)
    N, D, I, E = 4, 32, 16, 4
    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2*I, D)
    down = torch.randn(E, D, I)
    indices = torch.zeros(N, 1, dtype=torch.long)
    weights = torch.ones(N, 1)
    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)
    gate, up = F.linear(hidden, gate_up[0]).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, down[0])
    return assert_close("single_expert", output, expected, atol=1e-5)

def test_expert_dispatch_sparse():
    print("\n[Test 2b] Expert dispatch: sparse routing")
    from importlib import import_module
    moe_gemm = import_module("deepseekv3_2-kernels-flashinfer.moe_grouped_gemm")
    torch.manual_seed(42)
    N, D, I, E, K = 8, 32, 16, 8, 2
    hidden = torch.randn(N, D)
    gate_up = torch.randn(E, 2*I, D)
    down = torch.randn(E, D, I)
    indices = torch.stack([torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)], dim=1)
    weights = torch.full((N, K), 0.5)
    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, E)
    expected = torch.zeros_like(hidden)
    for eidx in [0, 1]:
        g, u = F.linear(hidden, gate_up[eidx]).chunk(2, dim=-1)
        expected += F.linear(F.silu(g) * u, down[eidx]) * 0.5
    return assert_close("sparse_routing", output, expected, atol=1e-5)


# -- Test 3: FlashInfer FP8 KV layout ---------------------------------------

def test_flashinfer_kv_roundtrip():
    print("\n[Test 3] FlashInfer FP8 KV roundtrip")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")
    torch.manual_seed(42)
    ckv = torch.randn(1, 4, 512, dtype=torch.bfloat16)
    kpe = torch.randn(1, 4, 64, dtype=torch.bfloat16)
    kv_fp8, scale = fp8.quantize_kv_flashinfer(ckv, kpe)
    ckv_rt, kpe_rt = fp8.dequantize_kv_flashinfer(kv_fp8, scale)
    rel_err = ((ckv.float() - ckv_rt.float()).abs() / (ckv.float().abs() + 1e-6)).max().item()
    ok = rel_err < 0.07 and kv_fp8.shape == (1, 4, 576)
    print(f"  {'PASS' if ok else 'FAIL'} roundtrip rel_err={rel_err:.4f}, shape={kv_fp8.shape}")
    return ok


# -- Test 4: Group routing --------------------------------------------------

def test_group_routing_filters():
    print("\n[Test 4a] Group routing filters groups (DeepSeek-V3 style)")
    from importlib import import_module
    router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")
    torch.manual_seed(42)
    logits = torch.randn(8, 16)
    # Boost first 2 groups (indices 0-7), suppress last 2 (indices 8-15)
    logits[:, :4] += 5.0; logits[:, 4:8] += 5.0
    logits[:, 8:12] -= 5.0; logits[:, 12:16] -= 5.0
    indices, _ = router.sigmoid_topk_route(logits, torch.zeros(16), top_k=4, n_group=4, topk_group=2)
    ok = all(idx.item() < 8 for row in indices for idx in row)
    print(f"  {'PASS' if ok else 'FAIL'} only surviving groups selected")
    return ok

def test_group_routing_full_selection():
    print("\n[Test 4b] n_group=N, topk_group=N == flat topk")
    from importlib import import_module
    router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")
    torch.manual_seed(42)
    logits = torch.randn(16, 8)
    bias = torch.randn(8)
    idx1, w1 = router.sigmoid_topk_route(logits.clone(), bias, top_k=2, n_group=1, topk_group=1)
    idx8, w8 = router.sigmoid_topk_route(logits.clone(), bias, top_k=2, n_group=8, topk_group=8)
    ok = all(set(idx1[i].tolist()) == set(idx8[i].tolist()) for i in range(16))
    if ok:
        s1, s8 = idx1.sort(dim=-1), idx8.sort(dim=-1)
        ok = assert_close("weights", w1.gather(1, s1.indices), w8.gather(1, s8.indices), atol=1e-5)
    return ok


# -- Test 5: Autoregressive decode ------------------------------------------

def test_autoregressive_decode():
    print("\n[Test 5] Autoregressive decode (prefill=4, decode=3)")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    model = kernel_model.DeepSeekV3ForCausalLM(cfg)
    model.eval()
    prefill_ids = torch.randint(0, cfg["vocab_size"], (1, 4))
    ok = True
    with torch.no_grad():
        _, logits, cache = model(input_ids=prefill_ids, use_cache=True)
        for step in range(3):
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            _, logits, cache = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
            ok = logits.shape == (1, 1, cfg["vocab_size"]) and ok
            ok = torch.isfinite(logits).all() and ok
    print(f"  {'PASS' if ok else 'FAIL'} decode shape/finite check")
    return ok


# -- Test 6: Gradient flow ---------------------------------------------------

def test_gradient_flow():
    print("\n[Test 6] Gradient flow")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    model = kernel_model.DeepSeekV3ForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    loss, _, _ = model(input_ids=ids, labels=ids)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    ok = len(no_grad) == 0
    if ok:
        grad_count = sum(1 for _, p in model.named_parameters() if p.grad is not None and p.grad.abs().max() > 0)
        print(f"  PASS gradient flow: {grad_count} params have non-zero grads")
    else:
        print(f"  FAIL {len(no_grad)} params with no grad: {no_grad[:5]}")
    return ok


# -- Test 7: Edge cases ------------------------------------------------------

def test_single_token():
    print("\n[Test 7a] Single token forward (B=1, S=1)")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=1)
    torch.manual_seed(42)
    model = kernel_model.DeepSeekV3ForCausalLM(cfg)
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(input_ids=torch.tensor([[42]]), use_cache=True)
    ok = logits.shape == (1, 1, cfg["vocab_size"]) and torch.isfinite(logits).all()
    print(f"  {'PASS' if ok else 'FAIL'} shape={logits.shape}")
    return ok

def test_shared_expert():
    print("\n[Test 7b] MoE shared expert contribution")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")
    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    moe = kernel_model.MoE(cfg); moe.eval()
    with torch.no_grad():
        out = moe(torch.randn(1, 4, cfg["hidden_size"]))
    ok = out.shape == (1, 4, cfg["hidden_size"]) and out.abs().max() > 1e-6
    print(f"  {'PASS' if ok else 'FAIL'} non-zero output")
    return ok

def test_empty_expert():
    print("\n[Test 7c] Empty expert assignment")
    from importlib import import_module
    moe_gemm = import_module("deepseekv3_2-kernels-flashinfer.moe_grouped_gemm")
    torch.manual_seed(42)
    hidden = torch.randn(2, 32)
    gate_up = torch.randn(8, 32, 32)
    down = torch.randn(8, 32, 16)
    indices = torch.tensor([[0, 1], [6, 7]])
    weights = torch.ones(2, 2) * 0.5
    output = moe_gemm.moe_grouped_gemm_forward(hidden, gate_up, down, indices, weights, 8)
    ok = output.shape == (2, 32) and torch.isfinite(output).all()
    print(f"  {'PASS' if ok else 'FAIL'} empty experts handled")
    return ok


# -- Test 8: YaRN RoPE ------------------------------------------------------

def test_yarn_rope_scaling():
    print("\n[Test 8] YaRN RoPE produces different freqs than standard RoPE")
    from importlib import import_module
    rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")
    cfg_yarn = make_cfg()
    cfg_std = dict(cfg_yarn)
    cfg_std["rope_scaling"] = None  # Disable YaRN

    rope_yarn = rope_mod.RotaryEmbedding(cfg_yarn)
    rope_std = rope_mod.RotaryEmbedding(cfg_std)

    x = torch.randn(1, 8, 128)
    pos = torch.arange(8).unsqueeze(0)
    cos_y, sin_y = rope_yarn(x, pos)
    cos_s, sin_s = rope_std(x, pos)

    # YaRN should produce different frequencies at longer positions
    diff = (cos_y - cos_s).abs().max().item()
    ok = diff > 1e-6  # Should be different
    print(f"  {'PASS' if ok else 'FAIL'} YaRN vs standard diff={diff:.6f}")
    return ok


# -- Test 9: DSA Stub -------------------------------------------------------

def test_dsa_stub():
    print("\n[Test 9] DSA stub returns all-position indices")
    from importlib import import_module
    dsa = import_module("deepseekv3_2-kernels-flashinfer.dsa_indexer")
    indexer = dsa.DSAIndexerStub()
    hidden = torch.randn(2, 8, 128)
    indices = indexer(hidden)
    ok = indices.shape == (2, 8, 8)
    ok = ok and (indices >= 0).all()
    print(f"  {'PASS' if ok else 'FAIL'} stub indices shape={indices.shape}")
    return ok


# -- Main --------------------------------------------------------------------

ALL_TESTS = [
    test_kvcache_multistep, test_kvcache_reset, test_paged_kvcache,
    test_expert_dispatch_single, test_expert_dispatch_sparse,
    test_flashinfer_kv_roundtrip,
    test_group_routing_filters, test_group_routing_full_selection,
    test_autoregressive_decode,
    test_gradient_flow,
    test_single_token, test_shared_expert, test_empty_expert,
    test_yarn_rope_scaling,
    test_dsa_stub,
]

if __name__ == "__main__":
    results = {}
    for fn in ALL_TESTS:
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            results[fn.__name__] = False
    print(f"\n{'='*60}\n{sum(results.values())}/{len(results)} passed\n{'='*60}")
    for n, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)
