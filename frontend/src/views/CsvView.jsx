import { useRef, useState } from 'react';
import { AlertCircle, FileUp, Loader2, Upload } from 'lucide-react';
import { api, errorMessage } from '../api';
import { formatDuration } from '../constants';
import SessionReport from '../components/SessionReport';

export default function CsvView({ backendOnline, onOpenHistory }) {
  const [file, setFile] = useState(null);
  const [save, setSave] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const pick = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
  };

  const analyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.predictCsv(file, { save, name: file.name }));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid">
      <section className="card span-12">
        <h2><FileUp size={16} /> Analyse a recorded CSV</h2>
        <p className="muted small">
          Accepted formats: the Android sensor-logger format used for the project recordings
          (<code>Timestamp,Sensor Type,Value1,Value2,Value3</code>, type 1 = accelerometer, 4 = gyroscope), or{' '}
          <code>timestamp_ms,ax,ay,az,gx,gy,gz</code> (accelerometer m/s², gyroscope rad/s — this is also what the
          “Raw sensor CSV” button on the Live tab produces). Data is resampled to 20 Hz; gaps longer than 1 s are skipped.
        </p>

        <div
          className={`dropzone ${dragging ? 'drag' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]); }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        >
          <Upload size={22} />
          <span>{file ? <strong>{file.name}</strong> : 'Drop a CSV here or click to choose'}</span>
          {file && <span className="muted small">{(file.size / 1024).toFixed(0)} KB</span>}
          <input ref={inputRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => pick(e.target.files?.[0])} />
        </div>

        <div className="controls">
          <label className="check">
            <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} /> Save to history
          </label>
          <button className="btn primary" onClick={analyze} disabled={!file || loading || !backendOnline}>
            {loading ? <Loader2 size={16} className="spin" /> : <FileUp size={16} />} {loading ? 'Analysing…' : 'Analyse'}
          </button>
        </div>
        {!backendOnline && <p className="muted small">Backend offline.</p>}
        {error && <div className="banner error"><AlertCircle size={16} /><span>{error}</span></div>}
      </section>

      {result && (
        <section className="card span-12">
          <h2>Results for {result.file}</h2>
          <div className="stats">
            <div className="stat"><span>{formatDuration(result.info.duration_sec)}</span><label>usable data</label></div>
            <div className="stat"><span>{result.summary.predictions}</span><label>windows</label></div>
            <div className="stat"><span>{result.info.segments}</span><label>continuous segments</label></div>
            <div className="stat"><span>{result.info.format}</span><label>format</label></div>
          </div>
          {result.session && (
            <p className="muted small">
              Saved to history as “{result.session.name}”.{' '}
              <button className="link" onClick={onOpenHistory}>Open history</button>
            </p>
          )}
          <SessionReport predictions={result.timeline} />
        </section>
      )}
    </div>
  );
}
