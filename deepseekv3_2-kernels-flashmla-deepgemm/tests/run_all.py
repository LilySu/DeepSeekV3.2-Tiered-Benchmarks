#!/usr/bin/env python3
"""
Test runner for DeepSeek-V3 FlashMLA + DeepGEMM kernel tests.

Usage:
    # Run all CPU tests
    python tests/run_all.py

    # Run specific test file
    python tests/run_all.py test_equivalence

    # Run with verbose output
    python tests/run_all.py -v

    # Run H100 tests (requires Hopper GPU)
    python tests/run_all.py --h100

    # Run all tests including H100
    python tests/run_all.py --all
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 FlashMLA+DeepGEMM test runner"
    )
    parser.add_argument(
        "tests",
        nargs="*",
        default=[],
        help="Specific test files or patterns to run",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--h100",
        action="store_true",
        help="Run H100-specific tests only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run ALL tests (CPU + H100)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=True,
        help="Run CPU-only tests (default)",
    )
    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Stop on first failure",
    )
    parser.add_argument(
        "-k", "--keyword",
        default=None,
        help="pytest -k expression",
    )

    args = parser.parse_args()

    # Build pytest arguments
    pytest_args = []

    test_dir = os.path.dirname(os.path.abspath(__file__))

    if args.tests:
        for t in args.tests:
            if os.path.exists(os.path.join(test_dir, t)):
                pytest_args.append(os.path.join(test_dir, t))
            elif os.path.exists(os.path.join(test_dir, f"test_{t}.py")):
                pytest_args.append(os.path.join(test_dir, f"test_{t}.py"))
            else:
                pytest_args.append(t)
    else:
        pytest_args.append(test_dir)

    # Filter by test type
    if args.h100:
        pytest_args.extend(["-k", "h100"])
    elif not args.all:
        pytest_args.extend(["-k", "not h100"])

    if args.verbose:
        pytest_args.append("-v")

    if args.exitfirst:
        pytest_args.append("-x")

    if args.keyword:
        # Override any existing -k flag
        if "-k" in pytest_args:
            idx = pytest_args.index("-k")
            pytest_args[idx + 1] = args.keyword
        else:
            pytest_args.extend(["-k", args.keyword])

    # Add useful pytest options
    pytest_args.extend([
        "--tb=short",
        "-q" if not args.verbose else "",
    ])
    pytest_args = [a for a in pytest_args if a]  # remove empty strings

    print(f"Running: pytest {' '.join(pytest_args)}")
    print(f"Test directory: {test_dir}")
    print()

    try:
        import pytest
        return pytest.main(pytest_args)
    except ImportError:
        print("ERROR: pytest is not installed. Install it with:")
        print("  pip install pytest")
        return 1


if __name__ == "__main__":
    sys.exit(main())
