"""
H100 CUDA graph tests.

Tests CUDA graph capture and replay for DeepSeek-V3 inference:
  - Single-token decode step capture
  - Graph replay correctness
  - Memory footprint of captured graphs
  - Multi-iteration graph replay stability

Requires: H100 GPU.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DeepSeekV3Config, MoEConfig

requires_hopper = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_properties(0).major >= 9),
    reason="H100/Hopper GPU required",
)


def _make_small_model_on_gpu():
    """Create a small model on GPU for graph testing."""
    from model import DeepSeekV3Model
    config = DeepSeekV3Config(
        hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4,
        q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        vocab_size=256,
        moe=MoEConfig(
            num_experts=8, num_experts_per_tok=2,
            n_group=2, topk_group=1,
            expert_intermediate_size=128,
            shared_expert_intermediate_size=128,
            first_k_dense_replace=1,
        ),
        use_flashmla=False, use_deepgemm=False,
    )
    model = DeepSeekV3Model(config).cuda().eval().to(torch.bfloat16)
    return model, config


@requires_hopper
class TestH100CUDAGraph:
    """Test CUDA graph capture and replay."""

    def test_graph_capture_decode(self):
        """Single decode step should be capturable."""
        model, config = _make_small_model_on_gpu()
        B = 1
        device = torch.device("cuda")

        input_ids = torch.randint(0, config.vocab_size, (B, 1), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)

        # Capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = model(input_ids)

        # Replay
        g.replay()
        torch.cuda.synchronize()

        assert out["logits"].shape == (B, 1, config.vocab_size)

    def test_graph_replay_deterministic(self):
        """Graph replay should produce consistent results."""
        model, config = _make_small_model_on_gpu()
        device = torch.device("cuda")
        B = 1

        static_input = torch.randint(0, config.vocab_size, (B, 1), device=device)

        with torch.no_grad():
            model(static_input)  # warmup

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(static_input)

        g.replay()
        logits1 = out["logits"].clone()

        g.replay()
        logits2 = out["logits"].clone()

        torch.testing.assert_close(logits1, logits2)

    def test_graph_replay_multiple_iterations(self):
        """Graph should be stable across many replays."""
        model, config = _make_small_model_on_gpu()
        device = torch.device("cuda")
        B = 1

        static_input = torch.randint(0, config.vocab_size, (B, 1), device=device)

        with torch.no_grad():
            model(static_input)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(static_input)

        first_logits = None
        for i in range(100):
            g.replay()
            if first_logits is None:
                first_logits = out["logits"].clone()

        torch.testing.assert_close(out["logits"], first_logits)

    def test_graph_memory_footprint(self):
        """CUDA graph should not leak memory."""
        model, config = _make_small_model_on_gpu()
        device = torch.device("cuda")
        B = 1

        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        static_input = torch.randint(0, config.vocab_size, (B, 1), device=device)

        with torch.no_grad():
            model(static_input)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(static_input)

        after_capture = torch.cuda.memory_allocated()

        for _ in range(50):
            g.replay()

        after_replay = torch.cuda.memory_allocated()

        print(f"\nGraph memory: before={before/1024**2:.1f}MB, "
              f"captured={after_capture/1024**2:.1f}MB, "
              f"after_replay={after_replay/1024**2:.1f}MB")

        # Memory should not grow significantly after replay
        assert after_replay <= after_capture * 1.1
