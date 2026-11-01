import { formatDuration, secondsByActivity, toSegments } from '../constants';
import Breakdown from './Breakdown';
import ConfidenceChart from './ConfidenceChart';
import Timeline from './Timeline';

/** Timeline + breakdown + confidence chart + segment table for a list of predictions. */
export default function SessionReport({ predictions }) {
  const segments = toSegments(predictions);
  return (
    <div className="report">
      <h3>Timeline</h3>
      <Timeline predictions={predictions} />
      <div className="report-grid">
        <div>
          <h3>Time per activity <span className="muted small">(approx.)</span></h3>
          <Breakdown secondsByActivity={secondsByActivity(predictions)} />
        </div>
        <div>
          <h3>Confidence over time</h3>
          <div className="chart-box tall"><ConfidenceChart predictions={predictions} /></div>
        </div>
      </div>
      <details className="segments">
        <summary>Segment table ({segments.length} segments)</summary>
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>From</th><th>To</th><th>Activity</th><th>Avg. confidence</th></tr></thead>
            <tbody>
              {segments.map((s, i) => (
                <tr key={i}>
                  <td>{formatDuration(s.start)}</td><td>{formatDuration(s.end)}</td><td>{s.activity}</td>
                  <td>{Math.round((s.conf / s.n) * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
