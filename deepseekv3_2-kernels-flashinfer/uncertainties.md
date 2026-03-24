# FlashInfer Integration: Uncertainties and Open Questions

## 1. FA3 Backend Maturity

### Question
FlashInfer's FA3 backend (FlashAttention-3 for Hopper) is relatively new. How stable is the API for MLA-specific operations?

### Current Understanding
- The `BatchMLAPagedAttentionWrapper` API was introduced for DeepSeek-V2 compatibility
- FA3 uses warp-specialization and TMA on Hopper, providing significant speedup over FA2
- The MLA path is less tested than the standard MHA/GQA paths

### Risk Level: Medium

### Open Items
- [ ] Verify FA3 MLA path handles `d_ckv=512` correctly (larger than typical head dims)
- [ ] Confirm that qk_nope=128 passes FlashInfer's internal validation without patching
- [ ] Test with mixed page sizes (page_size=1 for decode, page_size=64 for prefill)
- [ ] Verify correctness with FP8 KV cache + BF16 queries (mixed-precision path)

## 2. FP8 KV Cache Scale Handling

### Question
How should FP8 scales be stored and managed for the paged KV cache?

### Current Understanding
FlashInfer supports FP8 KV cache with an external scale tensor. However, the exact format for per-page vs per-entry scaling is unclear from the documentation.

### Options Under Consideration

**Option A: Per-page global scale**
- Single FP32 scale per page
- Pros: Simple, minimal memory overhead
- Cons: If a page contains both large and small values, precision suffers
- Memory: 4 bytes per page (negligible)

**Option B: Per-head per-page scale**
- One FP32 scale per attention head per page
- Pros: Better precision for varying magnitudes across heads
- Cons: 128 * 4 = 512 bytes per page extra memory
- Memory: significant for large page counts

**Option C: Per-token scale**
- One FP32 scale per token (like DeepSeek-V3's training format)
- Pros: Best precision match with training
- Cons: Memory overhead, may not be supported by FlashInfer natively

### Risk Level: Medium

### Open Items
- [ ] Benchmark precision difference between Option A and Option B
- [ ] Check if FlashInfer's FP8 MLA path supports per-page scale format
- [ ] Measure cosine similarity degradation at 128K context with each option
- [ ] Determine if DeepGEMM's FP8 format is compatible with FlashInfer's expectations

## 3. Absorbed vs Non-Absorbed MLA

### Question
Should we use absorbed (weight-absorbed) MLA or non-absorbed (explicit up-projection) MLA with FlashInfer?

### Current Understanding
DeepSeek-V3's paper describes absorbed MLA where the up-projection weights are absorbed into the Q/K/V projections, eliminating the need for explicit up-projection during inference. This reduces the attention head dimension to the latent dimension (512+64=576).

However, FlashInfer's MLA wrapper operates on the latent representations directly (`d_ckv=512`, `d_kpe=64`), which IS the absorbed form.

### Risk: Low (FlashInfer natively supports absorbed form)

### Open Items
- [ ] Verify that absorbed MLA gives identical results to non-absorbed on tiny configs
- [ ] Confirm the output space: FlashInfer MLA output is in latent space (512 dims), needs up-projection afterward
- [ ] Check if the shared V up-projection (identical for all heads) can be fused after attention

## 4. CUDA Graph Compatibility with Dynamic Sequences

### Question
Can FlashInfer's FA3 MLA decode be captured in CUDA graphs when batch composition changes?

### Current Understanding
CUDA graphs require fixed tensor shapes. In production decode:
- Batch size can change (sequences finish at different times)
- KV lengths differ across sequences
- New sequences can be added mid-batch

FlashInfer's `plan()` function recomputes the internal schedule, which may break graph capture.

### Risk Level: High

### Workarounds Being Investigated
1. **Fixed batch with padding**: Pad to max batch size, mask completed sequences
2. **Re-capture on batch change**: Accept re-capture cost when batch composition changes
3. **Bucket-based graphs**: Pre-capture graphs for common batch sizes (1, 2, 4, 8, ...)
4. **FlashInfer's CUDAGraphBatchDecodeWithPagedKVCacheWrapper**: Purpose-built for this

### Open Items
- [ ] Benchmark CUDAGraphBatchDecodeWithPagedKVCacheWrapper with MLA
- [ ] Measure re-capture cost vs padding cost for various batch sizes
- [ ] Test with continous batching (sequences added/removed dynamically)

## 5. TMA Utilization Verification

### Question
How do we verify that TMA (Tensor Memory Accelerator) is active in FlashInfer's FA3 kernels?

### Current Understanding
Hopper's TMA provides asynchronous data loading that overlaps with computation. If TMA is not active (e.g., due to alignment issues), kernels fall back to standard loads with 20-30% performance loss.

Direct TMA verification requires NSight Compute profiling, which is not always available in cloud environments (RunPod, Nebius).

### Proxy Metrics
- **Bandwidth**: If FA3 MLA decode achieves >2TB/s on H100, TMA is likely active
- **Latency**: Compare against known TMA vs non-TMA baselines
- **NSight**: `ncu --metrics smsp__sass_inst_executed_op_global_ld` shows TMA vs global loads

### Risk Level: Low (FlashInfer handles this internally)

### Open Items
- [ ] Run h100_test_tma.py to verify bandwidth proxy
- [ ] Profile with ncu on at least one H100 run to confirm TMA activity
- [ ] Document expected bandwidth thresholds for TMA-active vs TMA-inactive

## 6. Multi-GPU Tensor Parallelism

### Question
How does FlashInfer interact with tensor-parallel MLA across multiple H100s?

### Current Understanding
In TP, the 128 attention heads are split across GPUs (e.g., 16 heads per GPU for 8-way TP). Each GPU computes its local attention output, then all-reduce combines them.

FlashInfer operates on local heads only and is unaware of TP. The concern is:
- KV cache is local (no cross-GPU KV sharing for MLA)
- The query projection is TP-sharded, so q_nope and q_pe are local
- Output must be all-reduced after attention + up-projection

### Risk Level: Low

### Open Items
- [ ] Verify that FlashInfer works correctly with non-standard head counts (e.g., 16 instead of 128)
- [ ] Test that TP all-reduce after FlashInfer output preserves numerical equivalence
- [ ] Measure NCCL all-reduce overhead vs FlashInfer compute time

## 7. Memory Management at 128K Context

### Question
Can we serve 128K context on 8xH100 (640GB total) with FP8 KV cache?

### Memory Budget Estimate

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (FP8) | ~335 GB | 671B params / 2 (FP8) |
| KV cache (FP8, 128K) | ~4.3 GB per user | 576 * 128K * 61 layers |
| Activations | ~2 GB peak | Per-layer, recomputed |
| CUDA overhead | ~5 GB | Contexts, streams |
| **Total (1 user)** | **~346 GB** | Fits in 8xH100 |
| **Max concurrent users** | **~65** | (640-346)/4.3 GB |

### Risk Level: Medium (tight at high concurrency)

### Open Items
- [ ] Verify actual memory consumption with FlashInfer paged cache at 128K
- [ ] Test page defragmentation under memory pressure
- [ ] Measure throughput degradation as memory usage approaches 90%

## 8. FlashInfer Version Pinning

### Question
Which FlashInfer version should we pin to?

### Considerations
- FlashInfer is under active development; API changes are expected
- The MLA wrapper API has changed between 0.1.x and 0.2.x
- Building from source requires CUDA 12.0+ and specific Triton versions

### Recommendation
- Pin to `flashinfer>=0.2.0,<0.3.0` in requirements
- Abstract FlashInfer behind a wrapper class that can handle API changes
- Maintain a fallback to PyTorch eager attention if FlashInfer is unavailable

### Risk Level: Medium

### Open Items
- [ ] Test with flashinfer 0.2.0, 0.2.1, and latest nightly
- [ ] Create compatibility wrapper that handles API differences
- [ ] Document build instructions for source compilation on H100

## Summary Priority Matrix

| Uncertainty | Risk | Effort to Resolve | Priority |
|-------------|------|-------------------|----------|
| FA3 Backend Maturity | Medium | Medium | P1 |
| FP8 Scale Handling | Medium | High | P1 |
| Absorbed vs Non-Absorbed | Low | Low | P3 |
| CUDA Graph Dynamic Batches | High | High | P0 |
| TMA Utilization | Low | Low | P3 |
| Multi-GPU TP | Low | Medium | P2 |
| Memory at 128K | Medium | Medium | P1 |
| Version Pinning | Medium | Low | P2 |
