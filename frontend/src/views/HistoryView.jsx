import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, ArrowLeft, Download, FileText, History, Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { api, errorMessage } from '../api';
import { SOURCE_LABELS, colorFor, formatDate, formatDuration } from '../constants';
import SessionReport from '../components/SessionReport';

export default function HistoryView({ backendOnline }) {
  const [sessions, setSessions] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(async () => {
    try {
      setSessions(await api.sessions());
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    if (backendOnline) {
      let cancelled = false;
      api.sessions()
        .then((list) => { if (!cancelled) { setSessions(list); setError(null); } })
        .catch((err) => { if (!cancelled) setError(errorMessage(err)); });
      return () => { cancelled = true; };
    }
  }, [backendOnline]);

  const open = async (id) => {
    setDetailLoading(true);
    try {
      setSelected(await api.session(id));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setDetailLoading(false);
    }
  };

  const remove = async (s) => {
    if (!window.confirm(`Delete “${s.name}” and its ${s.summary?.predictions ?? ''} predictions? This cannot be undone.`)) return;
    try {
      await api.deleteSession(s.id);
      setSelected(null);
      load();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  if (selected) {
    return (
      <div className="grid">
        <section className="card span-12">
          <div className="detail-head">
            <button className="btn small" onClick={() => setSelected(null)}><ArrowLeft size={14} /> All sessions</button>
            <div className="controls">
              <a className="btn small" href={api.exportCsvUrl(selected.id)} download><Download size={14} /> CSV</a>
              <a className="btn small" href={api.exportPdfUrl(selected.id)} download><FileText size={14} /> PDF</a>
              <button className="btn small danger" onClick={() => remove(selected)}><Trash2 size={14} /> Delete</button>
            </div>
          </div>
          <h2 className="detail-title">{selected.name}</h2>
          <div className="stats">
            <div className="stat"><span>{SOURCE_LABELS[selected.source]}</span><label>source</label></div>
            <div className="stat"><span>{formatDuration(selected.active_seconds)}</span><label>recording time</label></div>
            <div className="stat"><span>{selected.summary.predictions}</span><label>predictions</label></div>
            <div className="stat"><span>{selected.summary.dominant_activity || '—'}</span><label>most frequent</label></div>
          </div>
          <p className="muted small">
            Started {formatDate(selected.started_at)}
            {selected.meta?.recording && ` · replayed from ${selected.meta.recording}`}
            {selected.meta?.file && ` · file ${selected.meta.file}`}
          </p>
          {selected.predictions.length
            ? <SessionReport predictions={selected.predictions} />
            : <p className="muted">This session has no predictions (it was stopped before the first 3-second window).</p>}
        </section>
      </div>
    );
  }

  return (
    <div className="grid">
      <section className="card span-12">
        <div className="detail-head">
          <h2><History size={16} /> Session history</h2>
          <button className="btn small" onClick={load} disabled={!backendOnline}><RefreshCw size={14} /> Refresh</button>
        </div>
        <p className="muted small">Stored locally by the backend in SQLite (<code>backend/data/har.db</code>).</p>
        {error && <div className="banner error"><AlertCircle size={16} /><span>{error}</span></div>}
        {!backendOnline && <p className="muted">Backend offline — history is unavailable.</p>}
        {backendOnline && sessions === null && !error && <p className="muted"><Loader2 size={14} className="spin" /> Loading…</p>}
        {sessions?.length === 0 && (
          <div className="empty">
            <History size={28} />
            <p>No sessions yet. Record one on the Live tab or analyse a CSV with “Save to history” ticked.</p>
          </div>
        )}
        {sessions?.length > 0 && (
          <div className="table-wrap">
            <table className="data-table sessions">
              <thead>
                <tr><th>Session</th><th>Source</th><th>Started</th><th>Duration</th><th>Activities</th><th /></tr>
              </thead>
              <tbody>
                {sessions.map((s) => {
                  const secs = s.summary.seconds_by_activity;
                  const total = Object.values(secs).reduce((a, b) => a + b, 0) || 1;
                  return (
                    <tr key={s.id} className="clickable" onClick={() => open(s.id)}>
                      <td><strong>{s.name}</strong></td>
                      <td><span className={`tag src-${s.source}`}>{SOURCE_LABELS[s.source]}</span></td>
                      <td className="nowrap">{formatDate(s.started_at)}</td>
                      <td>{formatDuration(s.active_seconds)}</td>
                      <td className="mini-cell">
                        {s.summary.predictions ? (
                          <div className="mini-bar" title={Object.entries(secs).map(([a, v]) => `${a}: ${formatDuration(v)}`).join(' · ')}>
                            {Object.entries(secs).map(([a, v]) => (
                              <span key={a} style={{ width: `${(v / total) * 100}%`, background: colorFor(a) }} />
                            ))}
                          </div>
                        ) : <span className="muted small">no predictions</span>}
                        {s.summary.dominant_activity && <span className="small muted">mostly {s.summary.dominant_activity}</span>}
                      </td>
                      <td>{detailLoading ? null : <span className="muted small">View →</span>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
