"""Run all tests for deepseekv3_2-kernels-flashinfer.

Usage:
    # CPU tests only:
    python3 -m deepseekv3_2-kernels-flashinfer.tests.run_all

    # Include H100 kernel tests:
    python3 -m deepseekv3_2-kernels-flashinfer.tests.run_all --h100
"""

import sys
import importlib
import traceback

CPU_TESTS = [
    ("test_equivalence", [
        "test_moe_router", "test_fp8_utils",
        "test_rmsnorm", "test_mla_attention", "test_full_model",
        "test_yarn_rope",
    ]),
    ("test_components", [fn.__name__ for fn in
        importlib.import_module("deepseekv3_2-kernels-flashinfer.tests.test_components").ALL_TESTS
    ]),
]

H100_TESTS = [
    ("h100_test_flashinfer_kernels", [
        "h100_test_flashinfer_fa3_dense_decode",
        "h100_test_flashinfer_mla_paged_attention",
        "h100_test_flashinfer_cuda_graph_dense",
    ]),
    ("h100_test_deepgemm_kernels", [
        "h100_test_deepgemm_grouped_gemm_contiguous",
        "h100_test_deepgemm_grouped_gemm_masked",
        "h100_test_deepgemm_fp8_activation_quantize",
    ]),
    ("h100_test_cuda_graph", [
        "h100_test_cuda_graph_capture_model",
        "h100_test_cuda_graph_kv_cache_update",
        "h100_test_cuda_graph_speedup",
    ]),
    ("h100_test_tma", [
        "h100_test_tma_bandwidth_flashinfer",
        "h100_test_tma_bandwidth_deepgemm",
    ]),
    ("h100_test_memory", [
        "h100_test_memory_peak_single_layer",
        "h100_test_memory_kv_cache_scaling",
        "h100_test_memory_no_leak_decode",
    ]),
    ("h100_test_fp8_edge_cases", [
        "h100_test_fp8_overflow_detection",
        "h100_test_fp8_zero_handling",
        "h100_test_fp8_subnormal_precision",
        "h100_test_fp8_flashinfer_kv_scale_correctness",
    ]),
    ("h100_test_launch_overhead", [
        "h100_test_launch_overhead_empty_kernels",
        "h100_test_launch_overhead_per_layer",
        "h100_test_launch_overhead_graph_vs_eager_model",
    ]),
    ("h100_test_determinism", [
        "h100_test_deterministic_topk",
        "h100_test_deterministic_full_decode",
        "h100_test_deterministic_moe_routing",
    ]),
    ("h100_test_sparse_patterns", [
        "h100_test_moe_expert_utilization",
        "h100_test_moe_group_balance",
        "h100_test_moe_routing_stability",
    ]),
    ("h100_test_precision_chain", [
        "h100_test_precision_chain_roundtrips",
        "h100_test_precision_full_pipeline",
    ]),
    ("h100_test_thermal", [
        "h100_test_thermal_sustained_gemm",
        "h100_test_thermal_clock_frequency",
    ]),
]


def run_test_list(test_modules, label):
    all_results = {}
    total = passed = failed = errors = 0
    for module_name, test_fns in test_modules:
        mod = importlib.import_module(f".{module_name}", package="deepseekv3_2-kernels-flashinfer.tests")
        for fn_name in test_fns:
            total += 1
            full_name = f"{module_name}.{fn_name}"
            try:
                result = getattr(mod, fn_name)()
                all_results[full_name] = result
                if result: passed += 1
                else: failed += 1
            except Exception as e:
                print(f"  ERROR {full_name}: {e}")
                traceback.print_exc()
                all_results[full_name] = False
                errors += 1
    print(f"\n{'='*70}")
    print(f"{label}: {passed}/{total} passed, {failed} failed, {errors} errors")
    print("=" * 70)
    for name, result in all_results.items():
        print(f"  {'PASS' if result else 'FAIL'}  {name}")
    return failed == 0 and errors == 0


def main():
    include_h100 = "--h100" in sys.argv
    print("=" * 70)
    print("DeepSeek-V3 FlashInfer Kernel Test Suite")
    print("=" * 70)
    ok = run_test_list(CPU_TESTS, "CPU TESTS")
    if include_h100:
        import torch
        if not torch.cuda.is_available():
            print("\nWARN: --h100 requested but no CUDA. Skipping.")
        else:
            h100_ok = run_test_list(H100_TESTS, "H100 KERNEL TESTS")
            ok = ok and h100_ok
    else:
        print("\nNote: Run with --h100 to include FlashInfer/DeepGEMM kernel tests on H100.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
