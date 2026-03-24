# Huawei Ascend Architecture for DeepSeek-V3

## Ascend 910B Architecture Overview

The Huawei Ascend 910B is built on the Da Vinci architecture, which features a heterogeneous compute design with three types of processing units:

**Cube Units:** Matrix computation engines that perform dense matrix multiply-accumulate (MAC) operations. Each Cube Unit operates on 16x16 matrices in a single cycle, providing the bulk of TFLOPS for GEMM workloads. The 910B has 32 AI Cores, each containing a Cube Unit. At 1.5 GHz, this provides 320 TFLOPS FP16 peak throughput.

**Vector Units:** SIMD engines for element-wise operations (activation functions, normalization, softmax). Each AI Core includes a Vector Unit with 256-bit SIMD width. These handle the non-GEMM operations in DeepSeek-V3: SwiGLU activation, RMSNorm, softmax, and RoPE application.

**Scalar Units:** General-purpose processors for control flow and addressing. Handle the routing logic in MoE layers, including the grouped top-K selection.

## Memory Hierarchy

The Ascend 910B memory hierarchy is crucial for DeepSeek-V3 performance:

- **HBM2e:** 64 GB with 1.6 TB/s bandwidth. This stores model weights, KV cache, and activations. For DeepSeek-V3 at INT8, the total weight footprint is approximately 671B / 8 = 84 GB for the full model, requiring at least 2 chips. With MLA's compressed KV cache (512 dims per layer per token), the KV cache is dramatically smaller than standard MHA.

- **L2 Buffer:** 32 MB shared across all AI Cores. This is the primary staging area for GEMM operands. The 128x128 block quantization of FP8 maps well to L2 tiling strategies, as each 128x128 block in BF16 is 32 KB, fitting neatly into L2 partitions.

- **L1 Buffer:** 1 MB per AI Core. Used for intra-operator tiling. MLA's down-projection (7168 to 512) can tile the input along the token dimension while keeping the entire output vector in L1.

- **Unified Buffer (UB):** Shared memory within each AI Core for data exchange between Cube and Vector units. Critical for fusing MLA projections with subsequent operations.

## Mapping DeepSeek-V3 to Ascend

### MLA on Cube Units

The MLA projections map naturally to Cube Unit GEMM operations:

1. **KV Down-projection** (7168 -> 512): This is a tall-skinny GEMM well-suited for Cube Units. The small output dimension (512) means the result fits in L1, enabling fusion with the subsequent up-projection.

2. **K/V Up-projection** (512 -> 16384): This fan-out GEMM requires careful tiling. The 512-dim input is small enough for the Cube Unit's 16x16 operation to process efficiently, but the large output requires multiple passes through L2.

3. **Q Projection** (7168 -> 24576): The largest MLA GEMM. At full model size (128 heads x 192 dim), this requires significant Cube Unit utilization. Tiling along the head dimension is the natural strategy.

4. **Output Projection** (16384 -> 7168): Standard GEMM, well-handled by Cube Units.

### MoE on Ascend

The MoE layer presents challenges on Ascend:

**Gating Network:** The gating GEMM (7168 -> 256) is small and fast on Cube Units. The grouped routing logic (8 groups, top-4 selection) runs on Scalar/Vector Units.

**Expert Dispatch:** Moving tokens between experts requires a scatter/gather pattern. On Ascend, this maps to the DMA engine for efficient data movement between AI Cores. The key optimization is to pipeline expert computation with data movement, overlapping the dispatch of the next expert's tokens with computation of the current expert.

**Expert FFN:** Each expert's SwiGLU FFN (7168 -> 2048 -> 7168) is a standard GEMM sequence. With 256 experts on a multi-chip setup, each chip handles a subset of experts. The 8-group structure aligns well with multi-chip partitioning.

### FP8 on Ascend

The Ascend 910B does not natively support FP8. Two adaptation strategies:

1. **INT8 Conversion:** Convert FP8 e4m3 weights to INT8 with calibrated scales. This introduces a small accuracy gap but leverages Ascend's INT8 Cube Unit support (640 TOPS). The 128x128 block scaling structure maps to INT8 per-channel quantization with per-block adjustments.

2. **BF16 Inference:** Use BF16 for computation, accepting the 2x memory overhead vs FP8. This is simpler but requires more HBM capacity (2 chips minimum for BF16 weights).

## CANN Kernel Development

Custom kernel development on Ascend uses the CANN toolkit:

```
// Pseudocode for MLA down-projection kernel on Ascend
// Uses AscendCL Cube API
aclStatus mla_kv_compress(
    aclTensor* input,      // (tokens, 7168) bf16
    aclTensor* W_down,     // (7168, 512) bf16/int8
    aclTensor* output      // (tokens, 512) bf16
) {
    // Tile along token dimension for L2 utilization
    // Each tile: (tile_tokens, 7168) x (7168, 512) -> (tile_tokens, 512)
    // Cube Unit handles the matmul
    // Result stays in L1 for subsequent up-projection fusion
}
```

The key optimization opportunities are:
- Fusing KV compression with up-projection via Unified Buffer
- Overlapping Cube (GEMM) and Vector (RMSNorm, SwiGLU) operations
- Using DMA for efficient KV cache management

## Multi-Chip Deployment

DeepSeek-V3 requires at least 8 Ascend 910B chips for INT8 inference (671B/8 bytes = 84 GB, plus KV cache and activations). The HCCS interconnect provides 56 GB/s bidirectional bandwidth between adjacent chips, compared to NVLink's 900 GB/s. This 16x bandwidth gap means that expert-parallel MoE communication is the primary bottleneck.

Mitigation strategies:
- Expert locality: assign expert groups to single chips to minimize cross-chip communication
- Token buffering: accumulate tokens before cross-chip dispatch to amortize communication overhead
- Computation-communication overlap: use Ascend's multi-stream capability to overlap GEMM with HCCS transfers
