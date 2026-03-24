"""H100 Category 3: Memory Allocation & Peak Usage for DeepSeek-V3.

671B model on 8xH100 is tight. KV cache at 128K context with 576 bytes/token
per layer across 61 layers. These tests track peak memory and check for leaks.

Requirements: H100 (SM90), CUDA.
Paper ref: DeepSeek-V3 (arXiv 2412.19437)
"""

import sys
import gc
import torch
from .conftest import make_cfg, make_full_cfg, skip_no_cuda


@skip_no_cuda
def h100_test_memory_peak_single_layer():
    """Measure peak GPU memory for a single MoE layer at full DeepSeek-V3 dims."""
    print("\n[H100-Mem-1] Peak memory for single MoE layer (full dims)")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    device = "cuda"
    cfg = make_full_cfg()
    cfg["num_hidden_layers"] = 1
    cfg["mlp_layer_types"] = ["sparse"]

    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1e9

    layer = kernel_model.DecoderLayer(cfg, layer_idx=0).to(device).half().eval()
    for p in layer.parameters():
        p.data = p.data.to(torch.bfloat16)

    mem_params = torch.cuda.memory_allocated() / 1e9 - mem_before

    rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    B, S = 1, 128
    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rope(hidden, pos_ids)
    mask = kernel_model.make_causal_mask(S, 0, hidden.dtype, device)

    with torch.no_grad():
        _ = layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))

    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Params: {mem_params:.2f} GB, Peak: {mem_peak:.2f} GB (B={B}, S={S})")

    ok = mem_peak < 30
    print(f"  {'PASS' if ok else 'FAIL'} peak memory {mem_peak:.2f} GB < 30 GB limit")
    return ok


@skip_no_cuda
def h100_test_memory_kv_cache_scaling():
    """KV cache memory should scale linearly with sequence length."""
    print("\n[H100-Mem-2] KV cache memory scaling")
    from importlib import import_module
    cache_mod = import_module("deepseekv3_2-kernels-flashinfer.cache")

    device = "cuda"
    gc.collect(); torch.cuda.empty_cache()

    sizes = []
    for num_pages in [64, 128, 256]:
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        cache = cache_mod.FlashInferPagedKVCache(
            num_layers=1, num_pages=num_pages, page_size=64,
            head_dim_ckv=512, head_dim_kpe=64, device=device,
        )
        mem_after = torch.cuda.memory_allocated()
        delta = (mem_after - mem_before) / 1e6
        sizes.append((num_pages, delta))
        del cache
        gc.collect(); torch.cuda.empty_cache()

    # Check linear scaling
    ok = True
    for i in range(1, len(sizes)):
        ratio = sizes[i][1] / sizes[0][1]
        expected = sizes[i][0] / sizes[0][0]
        if abs(ratio - expected) / expected > 0.15:
            ok = False
            print(f"  FAIL non-linear scaling: {sizes[i][0]} pages = {ratio:.2f}x (expected {expected:.1f}x)")

    if ok:
        print(f"  PASS KV cache scales linearly: {sizes}")
    return ok


@skip_no_cuda
def h100_test_memory_no_leak_decode():
    """Memory should not grow during decode loop (no leaks)."""
    print("\n[H100-Mem-3] No memory leak during decode")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    device = "cuda"
    model = kernel_model.DeepSeekV3ForCausalLM(cfg).to(device).eval()

    input_ids = torch.randint(0, cfg["vocab_size"], (1, 8), device=device)
    with torch.no_grad():
        _, logits, cache = model(input_ids=input_ids, use_cache=True)

    gc.collect(); torch.cuda.empty_cache()
    mem_start = torch.cuda.memory_allocated()

    with torch.no_grad():
        for _ in range(20):
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            _, logits, cache = model(input_ids=next_tok, past_key_values=cache, use_cache=True)

    mem_end = torch.cuda.memory_allocated()
    growth = (mem_end - mem_start) / 1e6

    # Some growth is expected (KV cache grows), but should be bounded
    per_step = growth / 20
    ok = per_step < 5.0  # Less than 5 MB per step for tiny model
    print(f"  Memory growth: {growth:.1f} MB over 20 steps ({per_step:.2f} MB/step)")
    print(f"  {'PASS' if ok else 'FAIL'} growth per step < 5 MB")
    return ok
