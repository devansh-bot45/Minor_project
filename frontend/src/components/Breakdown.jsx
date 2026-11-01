import { colorFor, formatDuration } from '../constants';

/** Approximate time per activity (each prediction = one 1.5 s window step). */
export default function Breakdown({ secondsByActivity }) {
  const entries = Object.entries(secondsByActivity || {});
  if (!entries.length) return <p className="muted small">No predictions yet.</p>;
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  return (
    <table className="breakdown">
      <tbody>
        {entries.map(([activity, secs]) => (
          <tr key={activity}>
            <td className="bd-name"><span className="swatch" style={{ background: colorFor(activity) }} />{activity}</td>
            <td className="bd-bar">
              <div className="prob-track">
                <div className="prob-fill" style={{ width: `${(secs / total) * 100}%`, background: colorFor(activity) }} />
              </div>
            </td>
            <td className="bd-val">{formatDuration(secs)}</td>
            <td className="bd-pct muted">{Math.round((secs / total) * 100)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
