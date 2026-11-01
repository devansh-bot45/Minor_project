import { useEffect, useState } from 'react';
import { AlertCircle, Brain, FlaskConical } from 'lucide-react';
import { api, errorMessage } from '../api';
import { colorFor } from '../constants';

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

export default function ModelView({ backendOnline }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!backendOnline) return;
    api.model().then(setInfo).catch((err) => setError(errorMessage(err)));
  }, [backendOnline]);

  if (!backendOnline) return <div className="grid"><section className="card span-12"><p className="muted">Backend offline.</p></section></div>;
  if (error) return <div className="grid"><div className="banner error span-12"><AlertCircle size={16} />{error}</div></div>;
  if (!info) return <div className="grid"><section className="card span-12"><p className="muted">Loading…</p></section></div>;

  const ev = info.evaluation;
  return (
    <div className="grid">
      <section className="card span-6">
        <h2><Brain size={16} /> Model</h2>
        <dl className="kv">
          <dt>Model version</dt><dd>{info.model_version}</dd>
          <dt>Architecture</dt><dd>{info.architecture}</dd>
          <dt>Parameters</dt><dd>{info.parameters.toLocaleString()}</dd>
          <dt>Input</dt><dd>{info.input_shape.join(' × ')} ({info.window_samples} samples at {info.sampling_rate_hz} Hz = 3 s, 8 channels)</dd>
          <dt>Raw sensor input</dt><dd>{info.raw_channels.join(', ')} — accel {info.units.accelerometer}, gyro {info.units.gyroscope}</dd>
          <dt>Window step</dt><dd>{info.window_step_samples} samples (1.5 s, 50% overlap)</dd>
          <dt>Uncertain if</dt><dd>confidence &lt; {info.postprocessing.confidence_threshold * 100}%</dd>
          <dt>Smoothing</dt><dd>EMA α = {info.postprocessing.ema_alpha}, majority vote over {info.postprocessing.majority_vote_window} windows (per session)</dd>
        </dl>
        <h3>Supported activities</h3>
        <div className="legend">
          {info.classes.map((c) => <span key={c} className="legend-item"><span className="swatch" style={{ background: colorFor(c) }} />{c}</span>)}
        </div>
      </section>

      <section className="card span-6">
        <h2>Pipeline</h2>
        <ol className="steps">
          <li>Phone samples accelerometer + gyroscope at 20 Hz (gyro converted from deg/s to rad/s).</li>
          <li>Every 1.5 s the last 60 samples (3 s) are sent to the backend.</li>
          <li>Zero-phase low-pass filter (5 Hz), gravity removed with a 0.3 Hz filter, plus |accel| and |gyro| → 8 channels — the exact function used to build the training data.</li>
          <li>StandardScaler (fitted at training time) → Conv1D + BiLSTM + attention model → 6 class probabilities.</li>
          <li>Rules: near-zero motion → Still; gentle motion cannot be Jogging/Stairs/Sports; confidence &lt; 40% → Uncertain. The UI flags whenever a rule changed the model’s answer.</li>
          <li>Per-session moving average + majority vote for a stable label.</li>
        </ol>
      </section>

      <section className="card span-12">
        <h2><FlaskConical size={16} /> Evaluation on the project’s recordings</h2>
        {!ev && <p className="muted">No evaluation report found. Run <code>python evaluate.py</code> in the backend folder.</p>}
        {ev && (
          <>
            <div className="banner warn">
              <AlertCircle size={16} />
              <span>
                Recordings marked “yes” were used to train this model, so their scores are optimistic. Only the
                rows marked “held out” measure generalisation, and they come from very few sessions.
                {ev.notes?.slice(3).map((n) => <span key={n}><br />{n}</span>)}
              </span>
            </div>
            <div className="stats">
              {ev.seen_in_training && <div className="stat"><span>{pct(ev.seen_in_training.pipeline_accuracy)}</span><label>accuracy, recordings seen in training ({ev.seen_in_training.windows} windows)</label></div>}
              {ev.held_out && <div className="stat"><span>{pct(ev.held_out.pipeline_accuracy)}</span><label>accuracy, held-out recordings ({ev.held_out.windows} windows)</label></div>}
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr><th>Recording</th><th>True label</th><th>In training?</th><th>Windows</th><th>Model acc.</th><th>Dashboard pipeline acc.</th><th>Predicted as (pipeline)</th></tr>
                </thead>
                <tbody>
                  {ev.recordings.map((r) => (
                    <tr key={r.file}>
                      <td>{r.file}</td>
                      <td>{r.label || <span className="muted">unlabelled</span>}</td>
                      <td>{r.used_in_training == null ? '—' : r.used_in_training ? 'yes' : <strong>held out</strong>}</td>
                      <td>{r.windows}</td>
                      <td>{pct(r.model_accuracy)}</td>
                      <td>{pct(r.pipeline_accuracy)}</td>
                      <td className="small">{Object.entries(r.pipeline_distribution).map(([a, n]) => `${a} ${n}`).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="muted small">Generated {new Date(ev.generated_at).toLocaleString()} by <code>backend/evaluate.py</code>.</p>
          </>
        )}
      </section>
    </div>
  );
}
