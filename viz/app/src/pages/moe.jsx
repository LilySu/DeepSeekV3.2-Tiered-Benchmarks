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

const groupColors = [
  '#da3633', '#e3b341', '#3fb950', '#58a6ff',
  '#bc8cff', '#f78166', '#39d353', '#79c0ff',
];

export default function MoEPage() {
  const [nGroup, setNGroup] = useState(8);
  const [topkGroup, setTopkGroup] = useState(4);
  const [topK, setTopK] = useState(8);
  const nExperts = 256;
  const expertsPerGroup = nExperts / nGroup;
  const expertsPerSelectedGroup = Math.floor(topK / topkGroup);

  const groupGrid = useMemo(() => {
    const groups = [];
    for (let g = 0; g < nGroup; g++) {
      const experts = [];
      for (let e = 0; e < Math.min(expertsPerGroup, 32); e++) {
        experts.push(g * expertsPerGroup + e);
      }
      groups.push({ id: g, experts, selected: g < topkGroup });
    }
    return groups;
  }, [nGroup, topkGroup, expertsPerGroup]);

  return (
    <div>
      <h2 style={{ color: '#58a6ff', marginBottom: '1rem' }}>
        Mixture of Experts with Grouped Routing
      </h2>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Configuration</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7, marginBottom: '1rem' }}>
          DeepSeek-V3 uses 256 fine-grained routed experts organized into groups. Tokens are routed
          hierarchically: first select top groups, then select top experts within each group.
          This reduces the routing search space and naturally aligns with expert-parallel deployment.
        </p>
        <div>
          <span style={statBox}>n_experts = {nExperts}</span>
          <span style={statBox}>n_group = {nGroup}</span>
          <span style={statBox}>topk_group = {topkGroup}</span>
          <span style={statBox}>top_k = {topK}</span>
          <span style={statBox}>experts/group = {expertsPerGroup}</span>
          <span style={statBox}>experts/selected_group = {expertsPerSelectedGroup}</span>
        </div>

        <div style={{ marginTop: '1rem' }}>
          <label style={{ color: '#8b949e', marginRight: '1rem', fontSize: '0.85rem' }}>
            Groups:
            <select
              value={nGroup}
              onChange={e => { setNGroup(+e.target.value); setTopkGroup(Math.min(topkGroup, +e.target.value)); }}
              style={{ marginLeft: '0.5rem', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', padding: '0.25rem', borderRadius: '4px' }}
            >
              {[1, 2, 4, 8, 16].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label style={{ color: '#8b949e', marginRight: '1rem', fontSize: '0.85rem' }}>
            TopK Groups:
            <select
              value={topkGroup}
              onChange={e => setTopkGroup(+e.target.value)}
              style={{ marginLeft: '0.5rem', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', padding: '0.25rem', borderRadius: '4px' }}
            >
              {Array.from({ length: nGroup }, (_, i) => i + 1).map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label style={{ color: '#8b949e', fontSize: '0.85rem' }}>
            Top-K Experts:
            <select
              value={topK}
              onChange={e => setTopK(+e.target.value)}
              style={{ marginLeft: '0.5rem', background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', padding: '0.25rem', borderRadius: '4px' }}
            >
              {[2, 4, 8, 16].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Routing Flow
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', margin: '0.75rem 0' }}>
          <div style={{ background: '#21262d', border: '1px solid #ffa657', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
            Token x<br/><small>7168 dims</small>
          </div>
          <span style={{ color: '#484f58', fontSize: '1.2rem' }}>&rarr;</span>
          <div style={{ background: '#21262d', border: '1px solid #8957e5', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
            Gate Linear<br/><small>7168 &rarr; {nExperts}</small>
          </div>
          <span style={{ color: '#484f58', fontSize: '1.2rem' }}>&rarr;</span>
          <div style={{ background: '#21262d', border: '1px solid #58a6ff', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
            Group Scores<br/><small>max per group</small>
          </div>
          <span style={{ color: '#484f58', fontSize: '1.2rem' }}>&rarr;</span>
          <div style={{ background: '#21262d', border: '1px solid #3fb950', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
            Top-{topkGroup} Groups<br/><small>of {nGroup}</small>
          </div>
          <span style={{ color: '#484f58', fontSize: '1.2rem' }}>&rarr;</span>
          <div style={{ background: '#21262d', border: '1px solid #da3633', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
            Top-{expertsPerSelectedGroup}/Group<br/><small>= {topK} total</small>
          </div>
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>
          Expert Group Visualization
          <span style={{ fontSize: '0.8rem', color: '#8b949e', marginLeft: '0.5rem' }}>
            (green = selected, dim = not selected)
          </span>
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(nGroup, 4)}, 1fr)`, gap: '0.75rem' }}>
          {groupGrid.map((group) => (
            <div key={group.id} style={{
              background: group.selected ? '#1a2f1a' : '#161b22',
              border: `1px solid ${group.selected ? '#238636' : '#30363d'}`,
              borderRadius: '6px',
              padding: '0.75rem',
              opacity: group.selected ? 1 : 0.5,
            }}>
              <div style={{
                color: groupColors[group.id % groupColors.length],
                fontWeight: 600,
                marginBottom: '0.5rem',
                fontSize: '0.85rem',
              }}>
                Group {group.id} ({expertsPerGroup} experts)
                {group.selected && <span style={{ color: '#3fb950', marginLeft: '0.5rem' }}>SELECTED</span>}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
                {group.experts.slice(0, 16).map((expertId, i) => (
                  <div key={expertId} style={{
                    width: '16px',
                    height: '16px',
                    borderRadius: '2px',
                    background: (group.selected && i < expertsPerSelectedGroup)
                      ? groupColors[group.id % groupColors.length]
                      : '#21262d',
                    border: '1px solid #30363d',
                    fontSize: '6px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#8b949e',
                  }} title={`Expert ${expertId}`}>
                  </div>
                ))}
                {expertsPerGroup > 16 && (
                  <span style={{ fontSize: '0.7rem', color: '#8b949e', alignSelf: 'center' }}>
                    +{expertsPerGroup - 16} more
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={card}>
        <h3 style={{ color: '#79c0ff', marginBottom: '0.75rem' }}>Expert FFN (SwiGLU)</h3>
        <p style={{ color: '#c9d1d9', lineHeight: 1.7 }}>
          Each routed expert is a SwiGLU FFN with intermediate_size = 2048. Plus one shared expert
          (always active, 2x intermediate = 4096) processes all tokens.
        </p>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem' }}>
          <thead>
            <tr>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Component</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Shape</th>
              <th style={{ padding: '0.5rem', textAlign: 'left', color: '#58a6ff', borderBottom: '1px solid #21262d' }}>Params/Expert</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>gate_proj</td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>7168 &rarr; 2048</td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>14.7M</td></tr>
            <tr><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>up_proj</td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>7168 &rarr; 2048</td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>14.7M</td></tr>
            <tr><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>down_proj</td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>2048 &rarr; 7168</td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>14.7M</td></tr>
            <tr style={{ background: '#1a2f1a' }}><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Total per expert</td><td style={{ padding: '0.5rem' }}></td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>44.1M</td></tr>
            <tr><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>All 256 experts</td><td style={{ padding: '0.5rem' }}></td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>11.3B</td></tr>
            <tr><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>Active (8 experts)</td><td style={{ padding: '0.5rem' }}></td><td style={{ padding: '0.5rem', fontFamily: 'monospace' }}>352.8M</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
