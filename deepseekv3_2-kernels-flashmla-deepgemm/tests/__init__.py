"""
Test suite for DeepSeek-V3 FlashMLA + DeepGEMM kernels.

Tests are organized into:
  - CPU-runnable tests (no GPU required): test_*.py
  - H100-specific tests (require Hopper GPU): h100_*.py

Run CPU tests:
    pytest tests/ -k "not h100"

Run H100 tests:
    pytest tests/ -k "h100"

Run all tests:
    pytest tests/
"""
