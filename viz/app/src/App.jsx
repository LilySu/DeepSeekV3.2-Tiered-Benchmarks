import React from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import MLAPage from './pages/mla';
import MoEPage from './pages/moe';
import MTPPage from './pages/mtp';
import OptimizationPage from './pages/optimization';
import YaRNRoPEPage from './pages/yarn_rope';

const navStyle = {
  display: 'flex',
  gap: '0.5rem',
  padding: '1rem 2rem',
  background: '#161b22',
  borderBottom: '1px solid #30363d',
  flexWrap: 'wrap',
};

const linkStyle = {
  color: '#8b949e',
  textDecoration: 'none',
  padding: '0.5rem 1rem',
  borderRadius: '6px',
  fontSize: '0.9rem',
  transition: 'all 0.2s',
};

const activeLinkStyle = {
  ...linkStyle,
  color: '#58a6ff',
  background: '#21262d',
};

const headerStyle = {
  background: '#0d1117',
  color: '#c9d1d9',
  minHeight: '100vh',
};

const titleStyle = {
  padding: '1.5rem 2rem 0.5rem',
  background: '#0d1117',
  borderBottom: '1px solid #21262d',
};

function NavItem({ to, children }) {
  return (
    <NavLink
      to={to}
      style={({ isActive }) => isActive ? activeLinkStyle : linkStyle}
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  return (
    <div style={headerStyle}>
      <div style={titleStyle}>
        <h1 style={{ color: '#58a6ff', fontSize: '1.5rem', marginBottom: '0.25rem' }}>
          DeepSeek-V3 671B
        </h1>
        <p style={{ color: '#8b949e', fontSize: '0.85rem' }}>
          Interactive Architecture Visualization | arXiv: 2412.19437
        </p>
      </div>
      <nav style={navStyle}>
        <NavItem to="/">MLA</NavItem>
        <NavItem to="/moe">MoE</NavItem>
        <NavItem to="/mtp">MTP</NavItem>
        <NavItem to="/yarn-rope">YaRN RoPE</NavItem>
        <NavItem to="/optimization">Optimization</NavItem>
      </nav>
      <div style={{ padding: '2rem' }}>
        <Routes>
          <Route path="/" element={<MLAPage />} />
          <Route path="/moe" element={<MoEPage />} />
          <Route path="/mtp" element={<MTPPage />} />
          <Route path="/yarn-rope" element={<YaRNRoPEPage />} />
          <Route path="/optimization" element={<OptimizationPage />} />
        </Routes>
      </div>
    </div>
  );
}
