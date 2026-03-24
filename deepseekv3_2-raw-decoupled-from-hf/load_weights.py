"""Load HuggingFace safetensors checkpoint into standalone DeepSeek-V3 model.

Supports both BF16 and FP8 (e4m3) checkpoints. The official DeepSeek-V3
checkpoint uses FP8 block quantization (128x128) with per-block scale factors
stored as 'weight_scale_inv' keys.
"""

import glob
import re
import torch
from safetensors.torch import load_file


# Skip layers beyond 61 (DeepSeek-V3 has 61 layers: 0-60).
# DO NOT skip MTP keys (may be stored as layer 61+ or under mtp_layers.*).
# DO NOT skip weight_scale_inv (needed for FP8 dequantization).
_SKIP_LAYER_RE = re.compile(r"model\.layers\.(6[2-9]|[7-9]\d|\d{3,})\.")


def assign(left, right, tensor_name="unknown"):
    """Copy right into left, checking shapes match."""
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in '{tensor_name}'. "
            f"Model: {left.shape}, Checkpoint: {right.shape}"
        )
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
    return left


def _dequantize_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize FP8 (e4m3) weight using per-block scale factors.

    DeepSeek-V3 uses 128x128 block quantization:
      weight_bf16[i,j] = weight_fp8[i,j] * scale_inv[i // 128, j // 128]

    Args:
        weight_fp8: FP8 tensor [M, N]
        scale_inv: Inverse scale factors [ceil(M/128), ceil(N/128)]
        block_size: Quantization block size (default 128)

    Returns:
        Dequantized BF16 tensor [M, N]
    """
    M, N = weight_fp8.shape
    weight_f32 = weight_fp8.to(torch.float32)

    # Expand scale_inv to match weight dimensions
    scale_expanded = scale_inv.repeat_interleave(block_size, dim=0)[:M, :]
    scale_expanded = scale_expanded.repeat_interleave(block_size, dim=1)[:, :N]

    return (weight_f32 * scale_expanded).to(torch.bfloat16)


def load_weights(model, checkpoint_dir, device="cpu", dtype=torch.bfloat16):
    """Load safetensors weights from checkpoint_dir into a DeepSeekV3ForCausalLM.

    Handles:
    - Multi-shard safetensors files
    - FP8 dequantization via weight_scale_inv keys
    - Weight tying (lm_head <-> embed_tokens)
    - MTP layer weights (stored as mtp_layers.* or model.layers.61.*)
    """
    shard_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")

    params = {}
    for shard in shard_files:
        params.update(load_file(shard, device=str(device)))

    # Collect FP8 scale factors separately
    scale_inv_map = {}
    for key in list(params.keys()):
        if "weight_scale_inv" in key:
            # Map "model.layers.3.mlp.experts.gate_up_proj.weight_scale_inv"
            # to  "model.layers.3.mlp.experts.gate_up_proj"
            weight_key = key.replace(".weight_scale_inv", "")
            scale_inv_map[weight_key] = params.pop(key)

    state_dict = model.state_dict()
    loaded_keys = set()

    for ckpt_key, ckpt_tensor in params.items():
        # Skip layers beyond model's layer count
        if _SKIP_LAYER_RE.search(ckpt_key):
            continue

        # Dequantize FP8 weights if scale factors are available
        if ckpt_key in scale_inv_map and ckpt_tensor.dtype == torch.float8_e4m3fn:
            ckpt_tensor = _dequantize_fp8_block(ckpt_tensor, scale_inv_map[ckpt_key])

        if ckpt_key in state_dict:
            assign(state_dict[ckpt_key], ckpt_tensor.to(state_dict[ckpt_key].dtype), ckpt_key)
            loaded_keys.add(ckpt_key)
        else:
            # Try buffer/parameter traversal for keys not in state_dict
            parts = ckpt_key.split(".")
            obj = model
            try:
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                attr_name = parts[-1]
                target = getattr(obj, attr_name, None)
                if target is not None and isinstance(target, torch.Tensor):
                    assign(target, ckpt_tensor.to(target.dtype), ckpt_key)
                    loaded_keys.add(ckpt_key)
            except (AttributeError, IndexError):
                pass

    # Weight tying
    if "lm_head.weight" not in loaded_keys:
        if "model.embed_tokens.weight" in loaded_keys:
            model.lm_head.weight = model.model.embed_tokens.weight
            print("Weight tying: lm_head.weight <- model.embed_tokens.weight")

    model.to(dtype=dtype, device=device)

    # Report
    model_keys = set(state_dict.keys())
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys
    if missing:
        print(f"Missing keys ({len(missing)}): {sorted(missing)[:10]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:10]}...")
    n_fp8 = len(scale_inv_map)
    print(f"Loaded {len(loaded_keys)} / {len(model_keys)} keys from {len(shard_files)} shard(s)")
    if n_fp8 > 0:
        print(f"Dequantized {n_fp8} FP8 tensors (128x128 block quantization)")

    return model
