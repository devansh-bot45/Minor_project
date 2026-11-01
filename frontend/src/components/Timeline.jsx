import { SECONDS_PER_PREDICTION, colorFor, formatDuration, toSegments } from '../constants';

/** Horizontal activity timeline built from actual predictions. */
export default function Timeline({ predictions }) {
  if (!predictions?.length) {
    return <div className="timeline-empty muted small">The timeline fills in as predictions arrive (one every 1.5 s).</div>;
  }
  const segs = toSegments(predictions);
  const t0 = segs[0].start;
  const total = Math.max(segs[segs.length - 1].end - t0, SECONDS_PER_PREDICTION);
  const present = [...new Set(segs.map((s) => s.activity))];

  return (
    <div>
      <div className="timeline" role="img" aria-label="Activity timeline">
        {segs.map((s, i) => (
          <div
            key={i}
            className="timeline-seg"
            style={{
              left: `${((s.start - t0) / total) * 100}%`,
              width: `${((s.end - s.start) / total) * 100}%`,
              background: colorFor(s.activity),
            }}
            title={`${s.activity}: ${formatDuration(s.start)} – ${formatDuration(s.end)} · avg confidence ${Math.round((s.conf / s.n) * 100)}%`}
          />
        ))}
      </div>
      <div className="timeline-axis muted small">
        <span>{formatDuration(t0)}</span>
        <span>{formatDuration(t0 + total)}</span>
      </div>
      <div className="legend">
        {present.map((a) => (
          <span key={a} className="legend-item"><span className="swatch" style={{ background: colorFor(a) }} />{a}</span>
        ))}
      </div>
    </div>
  );
}
