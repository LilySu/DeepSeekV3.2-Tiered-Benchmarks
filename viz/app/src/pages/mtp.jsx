import React from 'react';

const card = {
  background: '#161b22',
  border: '1px solid #30363d',
  borderRadius: '8px',
  padding: '1.5rem',
  marginBottom: '1.5rem',
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
});

const arrow = { color: '#484f58', fontSize: '1.2rem' };

export default function MTPPage() {
  const mtpProjFLOPs = 2 * 7168 * 7168;
  const lmHeadFLOPs = 2 * 7168 * 129280;
  const totalMTPFLOPs = mtpProjFLOPs + lmHeadFLOPs;

  return (
    <div>
      <h2 style={{ color: '#58a6ff', marginBottom: '1rem' }}>
        Multi-Token Prediction (MTP)
      </h2>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Overview</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          DeepSeek-V3 introduces Multi-Token Prediction (MTP) with 1 additional prediction layer
          that predicts the next-next token beyond standard next-token prediction. During training,
          MTP provides an additional training signal that improves representation quality.
          During inference, MTP enables speculative decoding by predicting additional tokens
          that can be verified in parallel.
        </p>
        <p style={{ color: '#8b949e', lineHeight: 1.7, marginTop: '0.75rem' }}>
          The MTP module shares the main model's embedding and LM head (output projection),
          adding only a single hidden-to-hidden projection per MTP layer. This makes the
          parameter and compute overhead minimal.
        </p>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Architecture</h3>

        <p style={{ color: '#8b949e', marginBottom: '0.5rem' }}>Standard next-token prediction:</p>
        <div style={flowRow}>
          <div style={flowBox('#ffa657')}>Input Tokens</div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#238636')}>61-Layer<br/>Transformer</div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#58a6ff')}>Hidden State<br/><small>7168 dims</small></div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#da3633')}>LM Head<br/><small>7168 &rarr; 129280</small></div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#3fb950')}>Token t+1<br/><small>prediction</small></div>
        </div>

        <p style={{ color: '#8b949e', marginBottom: '0.5rem', marginTop: '1rem' }}>MTP additional prediction (shared LM head):</p>
        <div style={flowRow}>
          <div style={flowBox('#58a6ff')}>Hidden State<br/><small>from layer 61</small></div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#8957e5')}>MTP Projection<br/><small>7168 &rarr; 7168</small></div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#58a6ff')}>MTP Hidden<br/><small>7168 dims</small></div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#da3633')}>Shared LM Head<br/><small>7168 &rarr; 129280</small></div>
          <span style={arrow}>&rarr;</span>
          <div style={flowBox('#e3b341')}>Token t+2<br/><small>prediction</small></div>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Compute Analysis</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Operation</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Shape</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>FLOPs/Token</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Parameters</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>MTP Projection</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>7168 &rarr; 7168</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{(mtpProjFLOPs / 1e6).toFixed(1)}M</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>51.4M</td>
            </tr>
            <tr>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>LM Head (shared)</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>7168 &rarr; 129280</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{(lmHeadFLOPs / 1e6).toFixed(0)}M</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>0 (shared)</td>
            </tr>
            <tr style={{ background: '#1a2f1a' }}>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace', fontWeight: 600 }}>Total MTP</td>
              <td style={{ padding: '0.5rem' }}></td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>{(totalMTPFLOPs / 1e6).toFixed(0)}M</td>
              <td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>51.4M (new)</td>
            </tr>
          </tbody>
        </table>
        <p style={{ color: '#8b949e', marginTop: '1rem', fontSize: '0.85rem' }}>
          MTP overhead: ~{((totalMTPFLOPs / (600e9)) * 100).toFixed(1)}% of estimated total model FLOPs per token.
          The 51.4M new parameters represent just 0.008% of the total 671B model.
        </p>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Speculative Decoding with MTP</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          During inference, MTP enables speculative decoding:
        </p>
        <div style={{ background: '#21262d', borderRadius: '6px', padding: '1rem', margin: '1rem 0', fontFamily: 'monospace', fontSize: '0.85rem', lineHeight: 1.8 }}>
          <div style={{ color: '#79c0ff' }}>Step 1: Forward pass produces hidden state h</div>
          <div style={{ color: '#3fb950' }}>Step 2: Main head predicts token t+1 from h</div>
          <div style={{ color: '#e3b341' }}>Step 3: MTP head predicts token t+2 from h (parallel with step 2)</div>
          <div style={{ color: '#79c0ff' }}>Step 4: Verify t+1 and t+2 in next forward pass</div>
          <div style={{ color: '#8b949e' }}>        If both accepted: 2 tokens generated in ~1 forward pass</div>
          <div style={{ color: '#8b949e' }}>        If t+2 rejected: fall back to standard decoding from t+1</div>
        </div>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          Expected speedup: 1.3-1.8x for decode phase, depending on MTP prediction accuracy
          and the acceptance rate for speculative tokens. The MTP head's accuracy is highest
          for common token sequences (function words, syntactic patterns) and lower for
          content words and rare tokens.
        </p>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Training with MTP</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          During training, MTP adds an auxiliary loss for the t+2 prediction. The total training
          loss becomes:
        </p>
        <div style={{ background: '#21262d', borderRadius: '6px', padding: '1rem', margin: '1rem 0', fontFamily: 'monospace', fontSize: '0.85rem', color: '#c9d1d9' }}>
          L_total = L_next_token + lambda * L_mtp
        </div>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          The MTP loss encourages the model's hidden representations to encode information
          about not just the immediate next token but also future tokens. This has been shown
          to improve representation quality and downstream task performance, even when MTP
          is not used at inference time.
        </p>
      </div>
    </div>
  );
}
