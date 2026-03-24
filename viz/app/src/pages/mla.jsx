import React, { useState } from 'react';

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

const flowRow = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.5rem',
  margin: '0.75rem 0',
  flexWrap: 'wrap',
};

const flowBox = (color = '#30363d') => ({
  background: '#21262d',
  border: `1px solid ${color}`,
  borderRadius: '4px',
  padding: '0.5rem 0.75rem',
  fontSize: '0.8rem',
  color: '#c9d1d9',
  textAlign: 'center',
  minWidth: '80px',
});

const arrow = { color: '#484f58', fontSize: '1.2rem' };

const table = { width: '100%', borderCollapse: 'collapse', margin: '1rem 0' };
const th = { padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' };
const td = { padding: '0.5rem', fontFamily: 'monospace', fontSize: '0.9rem', borderBottom: '1px solid #161b22' };

export default function MLAPage() {
  const [showAbsorbed, setShowAbsorbed] = useState(false);

  const standardKVPerToken = 128 * (128 + 64 + 128) * 2;  // num_heads * (nope+rope+v) * bf16
  const mlaKVPerToken = 512 * 2;  // kv_lora_rank * bf16
  const compressionRatio = (standardKVPerToken / mlaKVPerToken).toFixed(0);

  return (
    <div>
      <h2 style={{ color: '#58a6ff', marginBottom: '1rem' }}>
        Multi-head Latent Attention (MLA)
      </h2>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Key Insight</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          MLA compresses the KV cache from <strong style={{ color: '#ffa657' }}>{standardKVPerToken.toLocaleString()} bytes</strong> per
          token (standard MHA) to <strong style={{ color: '#3fb950' }}>{mlaKVPerToken.toLocaleString()} bytes</strong> per token,
          a <strong style={{ color: '#f0883e' }}>{compressionRatio}x</strong> reduction. This is achieved by
          projecting the hidden state into a low-rank latent space (d_c = 512) before storing in the KV cache.
          During attention, the latent is up-projected back to full K and V dimensions.
        </p>
        <div style={{ marginTop: '0.75rem' }}>
          <span style={statBox}>kv_lora_rank = 512</span>
          <span style={statBox}>num_heads = 128</span>
          <span style={statBox}>qk_nope = 128</span>
          <span style={statBox}>qk_rope = 64</span>
          <span style={statBox}>v_head = 128</span>
          <span style={statBox}>KV compression = {compressionRatio}x</span>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Data Flow
          <button
            onClick={() => setShowAbsorbed(!showAbsorbed)}
            style={{
              marginLeft: '1rem', padding: '0.25rem 0.75rem', borderRadius: '4px',
              background: showAbsorbed ? '#238636' : '#21262d', color: '#c9d1d9',
              border: '1px solid #30363d', cursor: 'pointer', fontSize: '0.8rem',
            }}
          >
            {showAbsorbed ? 'Absorbed Mode' : 'Standard Mode'}
          </button>
        </h3>

        {!showAbsorbed ? (
          <>
            <p style={{ color: '#8b949e', marginBottom: '0.5rem' }}>Standard MLA: explicit KV up-projection</p>
            <div style={flowRow}>
              <div style={flowBox('#ffa657')}>x<br/><small>7168</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#238636')}>W_dkv<br/><small>7168 &rarr; 512</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#da3633')}>c_kv (latent)<br/><small>512 dims</small><br/><small style={{color:'#ffa657'}}>KV Cache</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#8957e5')}>W_uk / W_uv<br/><small>512 &rarr; 16384</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox()}>K, V<br/><small>128 heads</small></div>
            </div>
            <div style={flowRow}>
              <div style={flowBox('#ffa657')}>x<br/><small>7168</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#238636')}>W_q<br/><small>7168 &rarr; 24576</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox()}>Q<br/><small>128h x (128+64)</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#da3633')}>Attention<br/><small>QK^T softmax V</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#238636')}>W_o<br/><small>16384 &rarr; 7168</small></div>
            </div>
          </>
        ) : (
          <>
            <p style={{ color: '#8b949e', marginBottom: '0.5rem' }}>Absorbed MLA: attention in compressed space (no explicit up-proj)</p>
            <div style={flowRow}>
              <div style={flowBox('#ffa657')}>x<br/><small>7168</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#238636')}>W_dkv<br/><small>7168 &rarr; 512</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#da3633')}>c_kv<br/><small>512 dims</small></div>
            </div>
            <div style={flowRow}>
              <div style={flowBox('#ffa657')}>Q</div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#8957e5')}>Absorb W_uk^T<br/><small>Q' = Q @ W_uk^T</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox('#da3633')}>Q' @ c_kv^T<br/><small>Compressed attn</small></div>
              <span style={arrow}>&rarr;</span>
              <div style={flowBox()}>Softmax @ c_kv<br/><small>then up-proj</small></div>
            </div>
            <p style={{ color: '#8b949e', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              Note: RoPE portion cannot be absorbed and is handled separately.
            </p>
          </>
        )}
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Projection Dimensions</h3>
        <table style={table}>
          <thead>
            <tr>
              <th style={th}>Projection</th>
              <th style={th}>Shape</th>
              <th style={th}>Parameters</th>
              <th style={th}>FLOPs/token</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style={td}>KV down (W_dkv)</td><td style={td}>7168 &rarr; 512</td><td style={td}>3.67M</td><td style={td}>7.34M</td></tr>
            <tr><td style={td}>K nope up (W_uk)</td><td style={td}>512 &rarr; 16384</td><td style={td}>8.39M</td><td style={td}>16.78M</td></tr>
            <tr><td style={td}>V up (W_uv)</td><td style={td}>512 &rarr; 16384</td><td style={td}>8.39M</td><td style={td}>16.78M</td></tr>
            <tr><td style={td}>RoPE key (W_rope_k)</td><td style={td}>7168 &rarr; 64</td><td style={td}>0.46M</td><td style={td}>0.92M</td></tr>
            <tr><td style={td}>Q (W_q)</td><td style={td}>7168 &rarr; 24576</td><td style={td}>176.16M</td><td style={td}>352.32M</td></tr>
            <tr><td style={td}>Output (W_o)</td><td style={td}>16384 &rarr; 7168</td><td style={td}>117.44M</td><td style={td}>234.88M</td></tr>
          </tbody>
        </table>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>KV Cache Comparison</h3>
        <table style={table}>
          <thead>
            <tr>
              <th style={th}>Method</th>
              <th style={th}>Bytes/Token/Layer</th>
              <th style={th}>128K Context (61 layers)</th>
              <th style={th}>Reduction</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={td}>Standard MHA (128 heads)</td>
              <td style={td}>{standardKVPerToken.toLocaleString()}</td>
              <td style={td}>{((standardKVPerToken * 128 * 1024 * 61) / 1e9).toFixed(1)} GB</td>
              <td style={td}>1x (baseline)</td>
            </tr>
            <tr>
              <td style={td}>GQA (8 KV heads)</td>
              <td style={td}>{(8 * (128+64+128) * 2).toLocaleString()}</td>
              <td style={td}>{((8 * 320 * 2 * 128 * 1024 * 61) / 1e9).toFixed(1)} GB</td>
              <td style={td}>{(standardKVPerToken / (8 * 320 * 2)).toFixed(0)}x</td>
            </tr>
            <tr style={{ background: '#1a2f1a' }}>
              <td style={td}>MLA (d_c=512)</td>
              <td style={td}>{mlaKVPerToken.toLocaleString()}</td>
              <td style={td}>{((mlaKVPerToken * 128 * 1024 * 61) / 1e9).toFixed(1)} GB</td>
              <td style={td}>{compressionRatio}x</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
