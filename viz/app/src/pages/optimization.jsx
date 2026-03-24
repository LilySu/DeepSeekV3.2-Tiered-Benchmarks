import React, { useState } from 'react';

const card = {
  background: '#161b22',
  border: '1px solid #30363d',
  borderRadius: '8px',
  padding: '1.5rem',
  marginBottom: '1.5rem',
};

const badge = (color) => ({
  display: 'inline-block',
  background: color,
  color: 'white',
  padding: '0.15rem 0.5rem',
  borderRadius: '10px',
  fontSize: '0.75rem',
  marginLeft: '0.5rem',
});

const barStyle = (pct, color) => ({
  height: '24px',
  background: color,
  width: `${pct}%`,
  borderRadius: '4px',
  display: 'flex',
  alignItems: 'center',
  paddingLeft: '0.5rem',
  fontSize: '0.75rem',
  color: 'white',
  minWidth: '40px',
  transition: 'width 0.3s',
});

const barContainer = {
  background: '#21262d',
  borderRadius: '4px',
  overflow: 'hidden',
  marginBottom: '0.5rem',
};

export default function OptimizationPage() {
  const [dtype, setDtype] = useState('bf16');

  const peakTFLOPS = dtype === 'fp8' ? 1979 : 989.5;
  const hbmBW = 3.35; // TB/s

  // Component-level estimated MFU
  const components = [
    { name: 'MLA Projections', mfu: dtype === 'fp8' ? 42 : 68, bound: 'compute', color: '#238636' },
    { name: 'MLA Attention (prefill)', mfu: dtype === 'fp8' ? 38 : 62, bound: 'compute', color: '#238636' },
    { name: 'MLA Attention (decode)', mfu: 15, bound: 'memory', color: '#da3633' },
    { name: 'MoE Gating', mfu: dtype === 'fp8' ? 12 : 20, bound: 'memory', color: '#da3633' },
    { name: 'MoE Expert FFN', mfu: dtype === 'fp8' ? 35 : 55, bound: 'compute', color: '#238636' },
    { name: 'MoE Dispatch', mfu: 8, bound: 'memory', color: '#da3633' },
    { name: 'Dense FFN', mfu: dtype === 'fp8' ? 45 : 72, bound: 'compute', color: '#238636' },
    { name: 'RMSNorm', mfu: 5, bound: 'memory', color: '#da3633' },
    { name: 'LM Head', mfu: dtype === 'fp8' ? 40 : 65, bound: 'compute', color: '#238636' },
    { name: 'MTP Head', mfu: dtype === 'fp8' ? 38 : 60, bound: 'compute', color: '#238636' },
  ];

  const optimizations = [
    {
      name: 'FP8 Inference',
      impact: 'high',
      speedup: '1.5-2.0x',
      description: 'Use native FP8 compute on H100 for all GEMM operations. DeepSeek-V3 was trained in FP8, so no additional quantization error.',
      applicable: 'H100, H200',
    },
    {
      name: 'FlashAttention-3 for MLA',
      impact: 'high',
      speedup: '2-3x attention',
      description: 'Adapt FlashAttention-3 to handle MLA compressed KV cache. Eliminates O(S^2) memory for attention scores.',
      applicable: 'H100 (TMA/warp specialization)',
    },
    {
      name: 'Expert Parallelism (8-way)',
      impact: 'high',
      speedup: 'Near-linear scaling',
      description: 'Map 8 expert groups to 8 GPUs. Grouped routing naturally aligns with expert-parallel deployment.',
      applicable: 'Multi-GPU (NVLink)',
    },
    {
      name: 'Fused MLA Projections',
      impact: 'medium',
      speedup: '10-20% per layer',
      description: 'Fuse RMSNorm + down-projection + up-projection into single kernel, eliminating intermediate HBM round-trips.',
      applicable: 'All platforms',
    },
    {
      name: 'MoE Dispatch Optimization',
      impact: 'medium',
      speedup: '20-40% MoE layer',
      description: 'Use sorted token dispatch with padding-free expert batches. Radix sort + scatter/gather for efficient token routing.',
      applicable: 'CUDA (CUB library)',
    },
    {
      name: 'KV Cache INT8 Quantization',
      impact: 'medium',
      speedup: '2x cache capacity',
      description: 'Quantize the 512-dim KV latent to INT8 (1 byte/element). Doubles the context length for same memory budget.',
      applicable: 'All platforms',
    },
    {
      name: 'Speculative Decoding (MTP)',
      impact: 'medium',
      speedup: '1.3-1.8x decode',
      description: 'Use MTP head to predict speculative tokens, verify in parallel. Particularly effective for common sequences.',
      applicable: 'All platforms',
    },
    {
      name: 'Continuous Batching',
      impact: 'medium',
      speedup: '2-4x throughput',
      description: 'Batch multiple requests, filling decode slots as prefill completes. Requires expert-aware scheduling for MoE.',
      applicable: 'Serving (vLLM, SGLang)',
    },
  ];

  return (
    <div>
      <h2 style={{ color: '#58a6ff', marginBottom: '1rem' }}>
        Optimization Dashboard
      </h2>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Hardware Target: NVIDIA H100 SXM5
        </h3>
        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
          <div>
            <div style={{ color: '#8b949e', fontSize: '0.85rem' }}>Peak TFLOPS ({dtype.toUpperCase()})</div>
            <div style={{ color: '#58a6ff', fontSize: '1.5rem', fontFamily: 'monospace' }}>{peakTFLOPS}</div>
          </div>
          <div>
            <div style={{ color: '#8b949e', fontSize: '0.85rem' }}>HBM Bandwidth</div>
            <div style={{ color: '#58a6ff', fontSize: '1.5rem', fontFamily: 'monospace' }}>{hbmBW} TB/s</div>
          </div>
          <div>
            <div style={{ color: '#8b949e', fontSize: '0.85rem' }}>Ridge Point</div>
            <div style={{ color: '#58a6ff', fontSize: '1.5rem', fontFamily: 'monospace' }}>
              {(peakTFLOPS * 1000 / (hbmBW * 1000)).toFixed(0)} FLOP/B
            </div>
          </div>
          <div>
            <label style={{ color: '#8b949e', fontSize: '0.85rem' }}>
              Precision:
              <select
                value={dtype}
                onChange={e => setDtype(e.target.value)}
                style={{ marginLeft: '0.5rem', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', padding: '0.25rem', borderRadius: '4px' }}
              >
                <option value="bf16">BF16</option>
                <option value="fp8">FP8</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Estimated MFU by Component ({dtype.toUpperCase()})
        </h3>
        {components.map(comp => (
          <div key={comp.name} style={{ marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <span style={{ color: '#c9d1d9', fontSize: '0.85rem' }}>{comp.name}</span>
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>
                {comp.mfu}% MFU
                <span style={badge(comp.bound === 'compute' ? '#238636' : '#da3633')}>
                  {comp.bound}
                </span>
              </span>
            </div>
            <div style={barContainer}>
              <div style={barStyle(comp.mfu, comp.color)}>
                {comp.mfu > 15 ? `${comp.mfu}%` : ''}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Optimization Opportunities
        </h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Optimization</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Impact</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Speedup</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Platform</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {optimizations.map(opt => (
              <tr key={opt.name}>
                <td style={{ padding: '0.5rem', fontWeight: 600, color: '#c9d1d9' }}>{opt.name}</td>
                <td style={{ padding: '0.5rem' }}>
                  <span style={badge(
                    opt.impact === 'high' ? '#da3633' : opt.impact === 'medium' ? '#e3b341' : '#238636'
                  )}>
                    {opt.impact}
                  </span>
                </td>
                <td style={{ padding: '0.5rem', fontFamily: 'monospace', color: '#3fb950' }}>{opt.speedup}</td>
                <td style={{ padding: '0.5rem', color: '#8b949e', fontSize: '0.85rem' }}>{opt.applicable}</td>
                <td style={{ padding: '0.5rem', color: '#8b949e', fontSize: '0.85rem' }}>{opt.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Memory Budget (Single H100, 80 GB)</h3>
        <div style={{ ...barContainer, height: '36px', display: 'flex' }}>
          <div style={{ ...barStyle(52, '#da3633'), borderRadius: '4px 0 0 4px' }}>Weights (FP8): 42 GB</div>
          <div style={{ ...barStyle(10, '#e3b341'), borderRadius: 0 }}>KV (32K): 8 GB</div>
          <div style={{ ...barStyle(10, '#238636'), borderRadius: 0 }}>Activations: 8 GB</div>
          <div style={{ ...barStyle(28, '#21262d'), borderRadius: '0 4px 4px 0', color: '#8b949e' }}>Free: 22 GB</div>
        </div>
        <p style={{ color: '#8b949e', fontSize: '0.85rem', marginTop: '0.5rem' }}>
          Note: Full model requires 8+ H100s. Single-GPU shows weight shard only.
          MLA's compressed KV cache (512 dims) enables 32K context in just 8 GB.
        </p>
      </div>
    </div>
  );
}
