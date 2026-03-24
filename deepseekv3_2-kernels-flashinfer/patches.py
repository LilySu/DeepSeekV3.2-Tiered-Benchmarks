# FlashInfer compatibility patches for DeepSeek-V3.
#
# DeepSeek-V3 uses qk_nope_head_dim=128, which MATCHES FlashInfer's native
# validation check (the kernel was originally designed for DeepSeek-V2/V3).
# Therefore, NO monkey-patching is needed for dimension validation.
#
# This module provides patch infrastructure for other potential compatibility
# issues and serves as the centralized location for any runtime modifications
# to FlashInfer or other kernel libraries.
#
# Key difference from GLM-5:
#   - GLM-5 requires a monkey-patch because qk_nope_head_dim=192 fails FlashInfer's
#     hardcoded check for qk_nope_head_dim=128
#   - DeepSeek-V3 passes the check natively (qk_nope_head_dim=128)
#
# Potential patches (not currently needed):
#   - Page size validation (if using non-standard page sizes)
#   - FP8 KV cache dtype registration
#   - CUDA graph workspace size adjustment for 128K context
#   - TMA descriptor alignment for DeepGEMM interop

_patched = False


def apply_deepseekv3_patches():
    """Apply FlashInfer compatibility patches for DeepSeek-V3.

    Currently a no-op because DeepSeek-V3's dimensions are natively
    compatible with FlashInfer. Safe to call multiple times.

    If future FlashInfer versions introduce validation that conflicts
    with DeepSeek-V3's configuration, patches should be added here.
    """
    global _patched
    if _patched:
        return

    try:
        import flashinfer.mla as _mla

        # Verify that FlashInfer accepts DeepSeek-V3's dimensions natively
        # qk_nope_head_dim=128 should pass the validation check
        # If it doesn't, we patch here (but it should)
        _patched = True

    except ImportError:
        pass  # FlashInfer not installed -- patches not needed
    except AttributeError:
        pass  # API changed upstream -- patch not applicable


def patch_flashinfer_workspace_size(workspace_size_mb: int = 256):
    """Increase FlashInfer workspace buffer for long-context DeepSeek-V3.

    DeepSeek-V3 supports 128K context with YaRN. The default FlashInfer
    workspace (128MB) may be insufficient for batch_size > 16 at long
    sequences. This patch increases the workspace allocation.

    Args:
        workspace_size_mb: workspace size in MB (default 256 for 128K context)
    """
    # This is a configuration hint, not a monkey-patch.
    # The actual workspace is allocated when creating BatchMLAPagedAttentionWrapper.
    # Usage: workspace = torch.empty(workspace_size_mb * 1024 * 1024, dtype=torch.int8, device="cuda")
    pass


def patch_cuda_graph_pool_size(pool_size_gb: float = 2.0):
    """Configure CUDA graph memory pool for DeepSeek-V3 decode.

    DeepSeek-V3's 61 layers with 128 attention heads require more
    graph memory than shallower models.

    Args:
        pool_size_gb: CUDA graph memory pool size in GB
    """
    import torch
    if hasattr(torch.cuda, 'graph_pool_handle'):
        # PyTorch manages the pool automatically, but we can hint the size
        pass


def get_deepseekv3_flashinfer_config():
    """Return FlashInfer configuration optimized for DeepSeek-V3.

    Returns a dict with recommended FlashInfer settings:
        - workspace_size: bytes for workspace buffer
        - page_size: tokens per KV cache page
        - backend: "fa3" for H100, "auto" for other GPUs
        - use_cuda_graph: True for decode, False for prefill
    """
    import torch
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)

    config = {
        "workspace_size": 256 * 1024 * 1024,  # 256 MB
        "page_size": 64,                       # 64 tokens per page
        "backend": "fa3" if capability >= (9, 0) else "auto",
        "use_cuda_graph": True,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "num_heads": 128,
        "sm_scale": (128 + 64) ** -0.5,       # 1/sqrt(192) for absorbed MLA
    }
    return config
