import { colorFor } from '../constants';

/** Raw model probabilities for one window, highest first. */
export default function ProbabilityBars({ probs }) {
  if (!probs) return <p className="muted small">Class probabilities appear after the first 3-second window.</p>;
  const rows = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  return (
    <div className="prob-list">
      {rows.map(([name, p]) => (
        <div className="prob-row" key={name} title={`${name}: ${(p * 100).toFixed(1)}%`}>
          <span className="prob-name"><span className="swatch" style={{ background: colorFor(name) }} />{name}</span>
          <div className="prob-track">
            <div className="prob-fill" style={{ width: `${p * 100}%`, background: colorFor(name) }} />
          </div>
          <span className="prob-pct">{Math.round(p * 100)}%</span>
        </div>
      ))}
    </div>
  );
}
