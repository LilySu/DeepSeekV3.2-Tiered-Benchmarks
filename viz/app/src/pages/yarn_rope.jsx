import React, { useState, useMemo } from 'react';

const card = {
  background: '#161b22',
  border: '1px solid #30363d',
  borderRadius: '8px',
  padding: '1.5rem',
  marginBottom: '1.5rem',
};

const statBox = {
  display: 'inline-block',
  background: '#21262d',
  borderRadius: '4px',
  padding: '0.25rem 0.75rem',
  margin: '0.25rem',
  fontFamily: 'monospace',
  color: '#79c0ff',
  fontSize: '0.85rem',
};

export default function YaRNRoPEPage() {
  const [maxSeqLen, setMaxSeqLen] = useState(163840);
  const [originalMaxLen, setOriginalMaxLen] = useState(4096);

  const config = {
    rope_theta: 10000,
    qk_rope_head_dim: 64,
    scaling_factor: 40,
    beta_fast: 32,
    beta_slow: 1,
    mscale: 0.707,
    mscale_all_dim: 0.707,
  };

  const extensionRatio = maxSeqLen / originalMaxLen;

  // Compute which frequency dimensions get scaled
  const freqAnalysis = useMemo(() => {
    const d = config.qk_rope_head_dim;
    const dims = [];
    for (let i = 0; i < d; i += 2) {
      const dimIdx = i / 2;
      const freq = 1.0 / Math.pow(config.rope_theta, i / d);
      const wavelength = 2 * Math.PI / freq;

      // YaRN classification
      let category;
      if (wavelength < originalMaxLen * 2 * Math.PI / config.beta_fast) {
        category = 'high_freq';
      } else if (wavelength > originalMaxLen * 2 * Math.PI / config.beta_slow) {
        category = 'low_freq';
      } else {
        category = 'interpolated';
      }

      dims.push({
        dimIdx,
        freq: freq.toExponential(3),
        wavelength: wavelength.toFixed(0),
        category,
      });
    }
    return dims;
  }, [originalMaxLen]);

  const categoryColors = {
    high_freq: '#3fb950',
    interpolated: '#e3b341',
    low_freq: '#da3633',
  };

  const categoryLabels = {
    high_freq: 'Unchanged (high freq)',
    interpolated: 'Interpolated (mid freq)',
    low_freq: 'Extrapolated (low freq)',
  };

  const categoryCounts = freqAnalysis.reduce((acc, d) => {
    acc[d.category] = (acc[d.category] || 0) + 1;
    return acc;
  }, {});

  return (
    <div>
      <h2 style={{ color: '#58a6ff', marginBottom: '1rem' }}>
        YaRN RoPE (Rotary Position Embedding)
      </h2>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Overview</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          DeepSeek-V3 uses YaRN (Yet another RoPE extensioN) to extend the context window
          from the pre-training length of {originalMaxLen.toLocaleString()} tokens to
          {' '}{maxSeqLen.toLocaleString()} tokens. Unlike DeepSeek-V3's nope+rope attention split,
          RoPE is only applied to the <strong style={{ color: '#ffa657' }}>qk_rope_head_dim = {config.qk_rope_head_dim}</strong> portion
          of each attention head. The remaining qk_nope_head_dim = 128 dimensions carry no positional information.
        </p>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7, marginTop: '0.75rem' }}>
          YaRN selectively scales different frequency components: high-frequency components (capturing
          local patterns) remain unchanged, while low-frequency components (capturing global position)
          are scaled to cover the extended range. Mid-frequency components are smoothly interpolated.
        </p>
        <div style={{ marginTop: '0.75rem' }}>
          <span style={statBox}>rope_theta = {config.rope_theta}</span>
          <span style={statBox}>d_rope = {config.qk_rope_head_dim}</span>
          <span style={statBox}>factor = {config.scaling_factor}</span>
          <span style={statBox}>beta_fast = {config.beta_fast}</span>
          <span style={statBox}>beta_slow = {config.beta_slow}</span>
          <span style={statBox}>mscale = {config.mscale}</span>
          <span style={statBox}>extension = {extensionRatio}x</span>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Configuration
        </h3>
        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
          <label style={{ color: '#8b949e', fontSize: '0.85rem' }}>
            Max Sequence Length:
            <select
              value={maxSeqLen}
              onChange={e => setMaxSeqLen(+e.target.value)}
              style={{ marginLeft: '0.5rem', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', padding: '0.25rem', borderRadius: '4px' }}
            >
              <option value={4096}>4,096 (no extension)</option>
              <option value={16384}>16,384</option>
              <option value={32768}>32,768</option>
              <option value={65536}>65,536</option>
              <option value={131072}>131,072</option>
              <option value={163840}>163,840 (DeepSeek-V3 default)</option>
            </select>
          </label>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Frequency Dimension Analysis
        </h3>
        <p style={{ color: '#8b949e', marginBottom: '0.75rem', fontSize: '0.85rem' }}>
          Each pair of dimensions in the {config.qk_rope_head_dim}-dim RoPE space has a characteristic frequency.
          YaRN categorizes them based on wavelength relative to the original context window:
        </p>

        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
          {Object.entries(categoryLabels).map(([cat, label]) => (
            <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '2px', background: categoryColors[cat] }} />
              <span style={{ color: '#c9d1d9', fontSize: '0.85rem' }}>
                {label}: {categoryCounts[cat] || 0} dims
              </span>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap', marginBottom: '1rem' }}>
          {freqAnalysis.map(d => (
            <div
              key={d.dimIdx}
              style={{
                width: '20px',
                height: '20px',
                borderRadius: '2px',
                background: categoryColors[d.category],
                opacity: 0.8,
                cursor: 'pointer',
              }}
              title={`Dim ${d.dimIdx}: freq=${d.freq}, wavelength=${d.wavelength}, ${d.category}`}
            />
          ))}
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Dim Pair</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Frequency</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Wavelength</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Category</th>
            </tr>
          </thead>
          <tbody>
            {freqAnalysis.filter((_, i) => i < 8 || i >= freqAnalysis.length - 4).map((d, i, arr) => (
              <React.Fragment key={d.dimIdx}>
                {i === 8 && arr.length > 12 && (
                  <tr><td colSpan={4} style={{ padding: '0.25rem', textAlign: 'center', color: '#484f58' }}>...</td></tr>
                )}
                <tr>
                  <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{d.dimIdx}</td>
                  <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{d.freq}</td>
                  <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{d.wavelength}</td>
                  <td style={{ padding: '0.5rem' }}>
                    <span style={{
                      color: categoryColors[d.category],
                      fontWeight: 600,
                      fontSize: '0.85rem',
                    }}>
                      {d.category.replace('_', ' ')}
                    </span>
                  </td>
                </tr>
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Key Differences from Standard RoPE</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Aspect</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Standard RoPE</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>DeepSeek-V3 YaRN + MLA</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ padding: '0.5rem', color: '#c9d1d9' }}>Applied dims</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>All head dims (e.g., 128)</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Only qk_rope = 64 dims</td>
            </tr>
            <tr>
              <td style={{ padding: '0.5rem', color: '#c9d1d9' }}>Non-positional dims</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>None</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>qk_nope = 128 dims</td>
            </tr>
            <tr>
              <td style={{ padding: '0.5rem', color: '#c9d1d9' }}>Context extension</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Linear interpolation</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>YaRN (frequency-dependent)</td>
            </tr>
            <tr>
              <td style={{ padding: '0.5rem', color: '#c9d1d9' }}>Max context</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Training length only</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{maxSeqLen.toLocaleString()} ({extensionRatio}x extension)</td>
            </tr>
            <tr>
              <td style={{ padding: '0.5rem', color: '#c9d1d9' }}>KV cache impact</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Stored in full KV cache</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Only rope key stored separately</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>MLA Nope/Rope Split</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          In DeepSeek-V3's MLA, each attention head's Q and K are split into two parts:
        </p>
        <div style={{ display: 'flex', gap: '1rem', margin: '1rem 0', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '200px', background: '#21262d', borderRadius: '6px', padding: '1rem', border: '1px solid #238636' }}>
            <h4 style={{ color: '#3fb950', marginBottom: '0.5rem' }}>Nope (128 dims)</h4>
            <p style={{ color: '#8b949e', fontSize: '0.85rem' }}>
              No positional encoding. Carries semantic/content information.
              Up-projected from the compressed KV latent (512 dims).
              Can be absorbed into the compressed attention computation.
            </p>
          </div>
          <div style={{ flex: 1, minWidth: '200px', background: '#21262d', borderRadius: '6px', padding: '1rem', border: '1px solid #58a6ff' }}>
            <h4 style={{ color: '#58a6ff', marginBottom: '0.5rem' }}>Rope (64 dims)</h4>
            <p style={{ color: '#8b949e', fontSize: '0.85rem' }}>
              RoPE with YaRN scaling applied. Carries positional information.
              Projected separately from the input (not from KV latent).
              Shared across all heads (broadcast). Cannot be absorbed.
            </p>
          </div>
        </div>
        <p style={{ color: '#8b949e', fontSize: '0.85rem' }}>
          This split allows MLA to capture position-independent semantics in the compressed latent
          while maintaining position awareness through the dedicated RoPE dimensions. The 128:64 ratio
          (2:1 nope:rope) reflects the observation that semantic content requires more dimensions than
          positional encoding.
        </p>
      </div>
    </div>
  );
}
