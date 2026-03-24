"""Test that deepseekv3_2-kernels-flashinfer components produce correct outputs.

Tests MoE router, FP8 utilities, RMSNorm, MLA attention, and full model forward
against expected behavior. Unlike GLM-5, there is no DSA indexer to test.

Run: python3 -m deepseekv3_2-kernels-flashinfer.tests.test_equivalence
"""

import sys
import torch
import torch.nn.functional as F
from .conftest import assert_close, make_cfg


def test_moe_router():
    print("\n[Test 1] MoE Sigmoid Router with Group Selection")
    from importlib import import_module
    kernel_router = import_module("deepseekv3_2-kernels-flashinfer.moe_router")

    cfg = make_cfg()
    torch.manual_seed(42)
    num_tokens, n_experts = 16, cfg["n_routed_experts"]
    router_logits = torch.randn(num_tokens, n_experts)
    bias = torch.randn(n_experts)

    kern_indices, kern_weights = kernel_router.sigmoid_topk_route(
        router_logits.clone(), bias, top_k=cfg["num_experts_per_tok"],
        n_group=cfg["n_group"], topk_group=cfg["topk_group"],
        norm_topk_prob=cfg["norm_topk_prob"], routed_scaling_factor=cfg["routed_scaling_factor"],
    )

    ok = True
    # Basic shape checks
    if kern_indices.shape != (num_tokens, cfg["num_experts_per_tok"]):
        print(f"  FAIL indices shape {kern_indices.shape}")
        ok = False
    if kern_weights.shape != (num_tokens, cfg["num_experts_per_tok"]):
        print(f"  FAIL weights shape {kern_weights.shape}")
        ok = False
    # All indices should be valid expert IDs
    if not (kern_indices >= 0).all() or not (kern_indices < n_experts).all():
        print(f"  FAIL invalid expert indices")
        ok = False
    # Weights should be positive (sigmoid scores)
    if not (kern_weights > 0).all():
        print(f"  FAIL non-positive weights")
        ok = False
    if ok:
        print(f"  PASS router output shapes and values correct")
    return ok


def test_fp8_utils():
    print("\n[Test 2] FP8 Quantization Utilities")
    from importlib import import_module
    fp8 = import_module("deepseekv3_2-kernels-flashinfer.fp8_utils")

    ok = True
    torch.manual_seed(42)
    x = torch.randn(4, 512, dtype=torch.float32)
    x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
    x_rt = fp8.dequantize_fp8(x_fp8, scales, block_size=128)
    max_rel_err = ((x - x_rt).abs() / (x.abs() + 1e-6)).max().item()
    if max_rel_err < 0.07:
        print(f"  PASS DeepGEMM roundtrip (max_rel_err={max_rel_err:.4f})")
    else:
        print(f"  FAIL DeepGEMM roundtrip (max_rel_err={max_rel_err:.4f})")
        ok = False

    # FlashInfer KV format: contiguous [pages, page_size, 576]
    ckv = torch.randn(2, 64, 512, dtype=torch.bfloat16)
    kpe = torch.randn(2, 64, 64, dtype=torch.bfloat16)
    kv_fp8, scale = fp8.quantize_kv_flashinfer(ckv, kpe)
    if kv_fp8.shape == (2, 64, 576) and kv_fp8.dtype == torch.float8_e4m3fn:
        print(f"  PASS FlashInfer KV shape {kv_fp8.shape}, scale={scale:.6f}")
    else:
        print(f"  FAIL FlashInfer KV shape {kv_fp8.shape} or dtype {kv_fp8.dtype}")
        ok = False

    # Roundtrip
    ckv_rt, kpe_rt = fp8.dequantize_kv_flashinfer(kv_fp8, scale, head_dim_ckv=512)
    if ckv_rt.shape == (2, 64, 512) and kpe_rt.shape == (2, 64, 64):
        print(f"  PASS FlashInfer KV dequant shapes correct")
    else:
        print(f"  FAIL FlashInfer KV dequant shapes: ckv={ckv_rt.shape}, kpe={kpe_rt.shape}")
        ok = False

    return ok


def test_rmsnorm():
    print("\n[Test 3] RMSNorm (should match PyTorch reference)")
    from importlib import import_module
    kernel_mla = import_module("deepseekv3_2-kernels-flashinfer.mla_attention")

    torch.manual_seed(42)
    norm = kernel_mla.RMSNorm(128, eps=1e-6)
    x = torch.randn(2, 8, 128)
    y = norm(x)

    # Manual reference
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    y_ref = norm.weight * (x_f * torch.rsqrt(var + 1e-6)).to(x.dtype)

    return assert_close("RMSNorm", y, y_ref, atol=1e-6)


def test_mla_attention():
    print("\n[Test 4] MLA Attention (full forward, eager fallback)")
    from importlib import import_module
    kernel_mla = import_module("deepseekv3_2-kernels-flashinfer.mla_attention")
    kernel_rope = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=1)
    torch.manual_seed(42)
    attn = kernel_mla.MLAttention(cfg, layer_idx=0)

    B, S, D = 1, 8, cfg["hidden_size"]
    hidden = torch.randn(B, S, D)
    rope_emb = kernel_rope.RotaryEmbedding(cfg)
    cos, sin = rope_emb(hidden, torch.arange(S).unsqueeze(0))
    mask = kernel_model.make_causal_mask(S, 0, hidden.dtype, hidden.device)

    out, _ = attn(hidden, (cos, sin), attention_mask=mask)
    ok = out.shape == (B, S, D)
    ok = ok and torch.isfinite(out).all()
    print(f"  {'PASS' if ok else 'FAIL'} MLA output shape={out.shape}, finite={torch.isfinite(out).all()}")
    return ok


def test_full_model():
    print("\n[Test 5] Full Model Forward (2 layers)")
    from importlib import import_module
    kernel_model = import_module("deepseekv3_2-kernels-flashinfer.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    model = kernel_model.DeepSeekV3ForCausalLM(cfg)
    model.eval()

    B, S = 1, 8
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S))
    with torch.no_grad():
        loss, logits, _ = model(input_ids=input_ids, labels=input_ids)

    ok = logits.shape == (B, S, cfg["vocab_size"])
    ok = ok and torch.isfinite(logits).all()
    ok = ok and loss is not None and torch.isfinite(loss)
    print(f"  {'PASS' if ok else 'FAIL'} logits shape={logits.shape}, loss={loss.item():.4f}")
    return ok


def test_yarn_rope():
    print("\n[Test 6] YaRN RoPE Embedding")
    from importlib import import_module
    rope_mod = import_module("deepseekv3_2-kernels-flashinfer.rope_partial")

    cfg = make_cfg()
    rope = rope_mod.RotaryEmbedding(cfg)

    x = torch.randn(1, 16, cfg["hidden_size"])
    pos_ids = torch.arange(16).unsqueeze(0)
    cos, sin = rope(x, pos_ids)

    ok = True
    if cos.shape != (1, 16, cfg["qk_rope_head_dim"]):
        print(f"  FAIL cos shape {cos.shape}")
        ok = False
    if sin.shape != (1, 16, cfg["qk_rope_head_dim"]):
        print(f"  FAIL sin shape {sin.shape}")
        ok = False
    if not torch.isfinite(cos).all() or not torch.isfinite(sin).all():
        print(f"  FAIL non-finite values in cos/sin")
        ok = False
    # cos^2 + sin^2 should be close to attention_scaling^2
    scale_sq = rope.attention_scaling ** 2
    norm_check = (cos[:, 0, :2].pow(2) + sin[:, 0, :2].pow(2))
    if not torch.allclose(norm_check, torch.full_like(norm_check, scale_sq), atol=1e-5):
        print(f"  FAIL cos^2+sin^2 != attention_scaling^2")
        ok = False
    if ok:
        print(f"  PASS YaRN RoPE shapes correct, scaling={rope.attention_scaling:.4f}")
    return ok


if __name__ == "__main__":
    results = {}
    for name, fn in [("moe_router", test_moe_router), ("fp8_utils", test_fp8_utils),
                     ("rmsnorm", test_rmsnorm), ("mla_attention", test_mla_attention),
                     ("full_model", test_full_model), ("yarn_rope", test_yarn_rope)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = False
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for n, r in results.items():
        print(f"  {'PASS' if r else 'FAIL'}  {n}")
    sys.exit(0 if all(results.values()) else 1)
