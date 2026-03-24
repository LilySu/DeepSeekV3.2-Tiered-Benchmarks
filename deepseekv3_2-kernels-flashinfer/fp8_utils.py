# FP8 quantization utilities for DeepSeek-V3 with FlashInfer and DeepGEMM.
#
# DeepSeek-V3 uses FP8 (E4M3) quantization extensively:
#   - KV cache compression: 576 bytes/token (512 ckv + 64 kpe)
#   - Activation quantization: 128x128 block-wise scaling (dynamic)
#   - Weight quantization: per-block FP8 with 128x128 blocks
#
# FlashInfer FP8 format:
#   [num_pages, page_size, 576] contiguous FP8 (ckv + kpe concatenated)
#   Scale factors are EXTERNAL (bmm1_scale, bmm2_scale), not inline.
#
# DeepGEMM FP8 format:
#   (tensor, scales) pair with per-block scaling (block_size=128)
#
# Key difference from GLM-5:
#   - Same KV cache dimensions (kv_lora_rank=512, qk_rope_head_dim=64)
#   - DeepSeek-V3 specifies 128x128 block size for activation FP8
#   - Dynamic activation quantization (not static calibration)
#
# Paper ref: DeepSeek-V3 (arXiv 2412.19437), Section 3.3 -- FP8 Training

import torch


# ---------------------------------------------------------------------------
# FlashInfer FP8 KV cache format
# ---------------------------------------------------------------------------

def quantize_kv_flashinfer(
    ckv: torch.Tensor,  # [num_pages, page_size, head_dim_ckv] BF16 (512)
    kpe: torch.Tensor,  # [num_pages, page_size, head_dim_kpe] BF16 (64)
) -> tuple[torch.Tensor, float]:
    """Quantize KV cache into FlashInfer's contiguous FP8 format.

    Layout per token: [ckv (512) | kpe (64)] = 576 bytes as FP8 E4M3.
    Scale factor is computed globally and returned separately.

    Args:
        ckv: [num_pages, page_size, 512] compressed KV in BF16
        kpe: [num_pages, page_size, 64] RoPE keys in BF16

    Returns:
        kv_fp8: [num_pages, page_size, 576] in FP8 E4M3
        scale: float -- global scale factor (amax / 448)
    """
    # Concatenate ckv and kpe
    kv = torch.cat([ckv, kpe], dim=-1)  # [num_pages, page_size, 576]

    # Global scale for the entire cache
    amax = kv.abs().float().max().clamp(min=1e-4)
    scale = (amax / 448.0).item()

    kv_fp8 = (kv.float() / scale).to(torch.float8_e4m3fn)
    return kv_fp8, scale


def dequantize_kv_flashinfer(
    kv_fp8: torch.Tensor,  # [num_pages, page_size, 576] FP8
    scale: float,
    head_dim_ckv: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize FlashInfer FP8 KV cache back to BF16.

    Returns:
        ckv: [num_pages, page_size, head_dim_ckv] BF16
        kpe: [num_pages, page_size, head_dim_kpe] BF16
    """
    kv = kv_fp8.float() * scale
    ckv = kv[..., :head_dim_ckv].to(torch.bfloat16)
    kpe = kv[..., head_dim_ckv:].to(torch.bfloat16)
    return ckv, kpe


# ---------------------------------------------------------------------------
# DeepGEMM FP8 activation format (128x128 block-wise scaling)
# ---------------------------------------------------------------------------

def quantize_activations_deepgemm(
    x: torch.Tensor,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 E4M3 with per-block scaling for DeepGEMM.

    DeepSeek-V3 uses 128x128 block-wise quantization for activations.
    Each block of `block_size` elements gets its own scale factor.

    Args:
        x: input tensor in BF16/FP32
        block_size: elements per scaling group (128 for DeepSeek-V3)

    Returns:
        x_fp8: quantized tensor in FP8 E4M3
        scales: per-block scale factors
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    flat = x.reshape(-1, d).float()
    m = flat.shape[0]

    num_blocks_per_row = (d + block_size - 1) // block_size
    if d % block_size != 0:
        pad = block_size - (d % block_size)
        flat = torch.nn.functional.pad(flat, (0, pad))

    flat_blocked = flat.reshape(m, num_blocks_per_row, block_size)
    amax = flat_blocked.abs().amax(dim=-1).clamp(min=1e-4)
    scales = amax / 448.0

    x_scaled = flat_blocked / scales.unsqueeze(-1)
    x_fp8 = x_scaled.reshape(m, -1)[:, :d].to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(orig_shape)

    return x_fp8, scales


def dequantize_fp8(x_fp8: torch.Tensor, scales: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Dequantize FP8 tensor back to BF16.

    Args:
        x_fp8: quantized tensor in FP8 E4M3
        scales: per-block scale factors
        block_size: elements per scaling group

    Returns:
        x: dequantized tensor in BF16
    """
    orig_shape = x_fp8.shape
    d = orig_shape[-1]
    flat = x_fp8.reshape(-1, d).float()
    m = flat.shape[0]

    num_blocks = scales.shape[-1]
    if d % block_size != 0:
        pad = block_size - (d % block_size)
        flat = torch.nn.functional.pad(flat, (0, pad))

    flat_blocked = flat.reshape(m, num_blocks, block_size)
    result = flat_blocked * scales.unsqueeze(-1)
    return result.reshape(m, -1)[:, :d].reshape(orig_shape).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# FP8 weight quantization (128x128 blocks, static)
# ---------------------------------------------------------------------------

def quantize_weights_fp8(
    weight: torch.Tensor,
    block_size_row: int = 128,
    block_size_col: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize model weights to FP8 E4M3 with 2D block-wise scaling.

    DeepSeek-V3 uses 128x128 block quantization for weight matrices.

    Args:
        weight: [out_features, in_features] weight matrix
        block_size_row: rows per block (128)
        block_size_col: cols per block (128)

    Returns:
        w_fp8: quantized weight in FP8 E4M3
        scales: [num_row_blocks, num_col_blocks] scale factors
    """
    out_f, in_f = weight.shape
    nr = (out_f + block_size_row - 1) // block_size_row
    nc = (in_f + block_size_col - 1) // block_size_col

    # Pad if necessary
    pad_r = nr * block_size_row - out_f
    pad_c = nc * block_size_col - in_f
    if pad_r > 0 or pad_c > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_c, 0, pad_r))

    w_blocked = weight.float().reshape(nr, block_size_row, nc, block_size_col)
    w_blocked = w_blocked.permute(0, 2, 1, 3)  # [nr, nc, br, bc]
    amax = w_blocked.abs().amax(dim=(-1, -2)).clamp(min=1e-4)  # [nr, nc]
    scales = amax / 448.0

    w_scaled = w_blocked / scales[:, :, None, None]
    w_scaled = w_scaled.permute(0, 2, 1, 3).reshape(nr * block_size_row, nc * block_size_col)
    w_fp8 = w_scaled[:out_f, :in_f].to(torch.float8_e4m3fn)

    return w_fp8, scales


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def compute_fp8_error(original: torch.Tensor, roundtrip: torch.Tensor) -> dict:
    """Compute FP8 quantization error metrics.

    Returns dict with max_abs_error, mean_abs_error, max_rel_error, cosine_sim.
    """
    diff = (original.float() - roundtrip.float()).abs()
    orig_abs = original.float().abs()

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    mask = orig_abs > 1e-6
    max_rel = (diff[mask] / orig_abs[mask]).max().item() if mask.any() else 0.0

    a_flat = original.float().flatten()
    b_flat = roundtrip.float().flatten()
    cos_sim = (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-12)).item()

    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
        "cosine_sim": cos_sim,
    }
