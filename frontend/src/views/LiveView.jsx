import { useEffect, useRef, useState } from 'react';
import {
  Activity, AlertCircle, Download, Gauge, History, Loader2, Pause, Play, Radio, RotateCcw, Smartphone, Square,
} from 'lucide-react';
import { api, errorMessage } from '../api';
import { formatDuration, secondsByActivity } from '../constants';
import { rawLogToCsv, useSensorStream } from '../hooks/useSensorStream';
import ActivityHero from '../components/ActivityHero';
import Breakdown from '../components/Breakdown';
import ProbabilityBars from '../components/ProbabilityBars';
import SensorChart from '../components/SensorChart';
import Timeline from '../components/Timeline';

const MAX_IN_FLIGHT = 3; // skip a window rather than queueing requests if the server is slow

function downloadText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function LiveView({ backendOnline, authToken, onOpenHistory }) {
  const [source, setSource] = useState('live');
  const [samples, setSamples] = useState([]);
  const [sampleId, setSampleId] = useState('');
  const [sessionName, setSessionName] = useState('');
  const [session, setSession] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [latestPrediction, setLatestPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [skipped, setSkipped] = useState(0);
  const [finished, setFinished] = useState(null); // summary of the last stopped session

  const sessionRef = useRef(null);
  const inFlightRef = useRef(0);
  const stoppingRef = useRef(false);

  useEffect(() => {
    if (!backendOnline) return;
    api.samples().then((list) => {
      setSamples(list);
      setSampleId((id) => id || list.find((s) => s.label === 'Walking')?.id || list[0]?.id || '');
    }).catch(() => {});
  }, [backendOnline]);

  const onWindow = async (windowData, tSec) => {
    const sid = sessionRef.current?.id;
    if (!sid) return;
    if (inFlightRef.current >= MAX_IN_FLIGHT) {
      setSkipped((n) => n + 1);
      return;
    }
    inFlightRef.current += 1;
    try {
      const result = await api.predict(windowData, sid, tSec, authToken);
      if (sessionRef.current?.id !== sid) return; // session already stopped
      setLatestPrediction(result);
      setPredictions((prev) => [...prev, result]);
      setError(null);
    } catch (err) {
      setError(`Prediction failed: ${errorMessage(err)}`);
    } finally {
      inFlightRef.current -= 1;
    }
  };

  const sensor = useSensorStream({ onWindow, onReplayEnd: () => handleStop() });
  const running = sensor.status === 'running';
  const paused = sensor.status === 'paused';
  const active = running || paused;

  const handleStart = async () => {
    setError(null);
    setWarning(null);
    setFinished(null);
    setPredictions([]);
    setLatestPrediction(null);
    setSkipped(0);
    stoppingRef.current = false;
    try {
      let replay = null;
      if (source === 'replay') {
        replay = await api.sample(sampleId);
      }
      // sensor.start must be the first await on the live path (iOS permission prompt needs the tap).
      const info = await sensor.start({
        mode: source,
        samples: replay?.samples,
        beforeStart: async () => {
          const meta = replay
            ? { recording: replay.file, recording_label: replay.label, used_in_training: replay.used_in_training }
            : { user_agent: navigator.userAgent };
          const name = sessionName.trim() || (replay ? `Replay: ${replay.label || replay.file}` : 'Live session');
          const s = await api.createSession(name, source, meta);
          sessionRef.current = { ...s, replay };
          setSession(sessionRef.current);
        },
      });
      if (source === 'live' && !info.hasGyro) {
        setWarning('No gyroscope readings from this device: gyro channels are sent as 0, so predictions may be less accurate.');
      }
    } catch (err) {
      sessionRef.current = null;
      setSession(null);
      setError(errorMessage(err));
    }
  };

  async function handleStop() {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    const { activeSeconds, rawLog } = sensor.stop();
    const s = sessionRef.current;
    // Let predictions that are already on their way finish (max ~5 s) so the last windows are kept.
    for (let i = 0; i < 50 && inFlightRef.current > 0; i++) {
      await new Promise((r) => setTimeout(r, 100));
    }
    sessionRef.current = null;
    setSession(null);
    if (!s) return;
    try {
      await api.endSession(s.id, activeSeconds);
    } catch (err) {
      setError(`Could not finalise the session: ${errorMessage(err)}`);
    }
    setFinished({ session: s, activeSeconds, rawLog });
  }

  const statusPill = (() => {
    if (sensor.status === 'starting') return { cls: 'pill-wait', text: source === 'live' ? 'Waiting for sensors…' : 'Loading recording…' };
    if (running && sensor.mode === 'replay') return { cls: 'pill-replay', text: 'Replaying recording' };
    if (running) return { cls: 'pill-live', text: 'Live sensors' };
    if (paused) return { cls: 'pill-paused', text: 'Paused' };
    return { cls: 'pill-idle', text: 'Not recording' };
  })();

  const replayInfo = session?.replay;
  const latest = sensor.latest;
  const counts = secondsByActivity(predictions);

  return (
    <div className="grid">
      {replayInfo && (
        <div className="banner replay span-12">
          <RotateCcw size={16} />
          <span>
            <strong>Replay mode — not live sensor data.</strong> Streaming the recorded file “{replayInfo.file}”
            ({[
              replayInfo.label ? `recorded as ${replayInfo.label}` : 'no reliable label',
              replayInfo.used_in_training === true && 'part of the training data',
              replayInfo.used_in_training === false && 'not used in training',
            ].filter(Boolean).join(', ')}) through the same pipeline at 20 Hz.
          </span>
        </div>
      )}
      {error && <div className="banner error span-12"><AlertCircle size={16} /><span>{error}</span></div>}
      {warning && <div className="banner warn span-12"><AlertCircle size={16} /><span>{warning}</span></div>}

      {/* Current activity */}
      <section className="card span-7">
        <h2><Activity size={16} /> Current activity</h2>
        <ActivityHero
          prediction={latestPrediction}
          emptyText={active ? 'Collecting the first 3-second window…' : 'Start a session to see predictions.'}
        />
      </section>

      {/* Session controls */}
      <section className="card span-5">
        <h2><Radio size={16} /> Session</h2>
        <div className="session-top">
          <span className={`pill ${statusPill.cls}`}><span className="dot" />{statusPill.text}</span>
          <span className="duration" aria-label="Recording time">{formatDuration(sensor.activeSeconds)}</span>
        </div>

        {!active && (
          <>
            <div className="segmented" role="tablist" aria-label="Data source">
              <button role="tab" aria-selected={source === 'live'} className={source === 'live' ? 'on' : ''}
                      onClick={() => setSource('live')}><Smartphone size={14} /> Phone sensors</button>
              <button role="tab" aria-selected={source === 'replay'} className={source === 'replay' ? 'on' : ''}
                      onClick={() => setSource('replay')}><RotateCcw size={14} /> Replay recording</button>
            </div>
            {source === 'replay' && (
              <label className="field">
                <span>Recording</span>
                <select value={sampleId} onChange={(e) => setSampleId(e.target.value)} disabled={!samples.length}>
                  {samples.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label || 'Unlabelled'} — {s.file}
                      {s.used_in_training === false ? ' (held out)' : s.used_in_training ? ' (in training data)' : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {source === 'live' && (
              <p className="muted small">
                Open this page on a phone over HTTPS, hold or pocket the phone, then press Start.
                Laptops usually have no motion sensor.
              </p>
            )}
            <label className="field">
              <span>Session name (optional)</span>
              <input value={sessionName} onChange={(e) => setSessionName(e.target.value)} maxLength={80}
                     placeholder={source === 'live' ? 'e.g. Evening walk' : 'e.g. Stairs demo'} />
            </label>
          </>
        )}

        {active && (
          <div className="buffer">
            <div className="buffer-label muted small">
              <span>Next window</span>
              <span>{predictions.length} predictions{skipped ? ` · ${skipped} skipped (server busy)` : ''}</span>
            </div>
            <div className="meter slim"><div className="meter-fill" style={{ width: `${sensor.windowProgress * 100}%` }} /></div>
          </div>
        )}

        <div className="controls">
          {!active && (
            <button className="btn primary" onClick={handleStart}
                    disabled={!backendOnline || sensor.status === 'starting' || (source === 'replay' && !sampleId)}>
              {sensor.status === 'starting' ? <Loader2 size={16} className="spin" /> : <Play size={16} />}
              Start
            </button>
          )}
          {running && <button className="btn" onClick={sensor.pause}><Pause size={16} /> Pause</button>}
          {paused && <button className="btn primary" onClick={sensor.resume}><Play size={16} /> Resume</button>}
          {active && <button className="btn danger" onClick={handleStop}><Square size={16} /> Stop</button>}
        </div>
        {!backendOnline && <p className="muted small">Backend offline — start it to record sessions.</p>}

        {finished && !active && (
          <div className="finished">
            <p>
              <strong>Saved “{finished.session.name}”</strong> · {formatDuration(finished.activeSeconds)} ·{' '}
              {predictions.length} predictions
            </p>
            <div className="controls">
              <button className="btn small" onClick={onOpenHistory}><History size={14} /> View in history</button>
              {finished.rawLog.length > 0 && (
                <button className="btn small" onClick={() => downloadText(rawLogToCsv(finished.rawLog),
                  `raw_sensors_${finished.session.id}.csv`)}>
                  <Download size={14} /> Raw sensor CSV
                </button>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Timeline */}
      <section className="card span-12">
        <h2><History size={16} /> Session timeline</h2>
        <Timeline predictions={predictions} />
        {predictions.length > 0 && (
          <div className="mt">
            <h3>Time per activity <span className="muted small">(approx., 1.5 s per prediction)</span></h3>
            <Breakdown secondsByActivity={counts} />
          </div>
        )}
      </section>

      {/* Sensors */}
      <section className="card span-7">
        <h2><Smartphone size={16} /> Sensor readings {sensor.mode === 'replay' && <span className="tag">recorded</span>}</h2>
        <div className="readings">
          <div>
            <h3>Accelerometer <span className="muted small">m/s², incl. gravity</span></h3>
            <div className="axis-vals">
              {['X', 'Y', 'Z'].map((a, i) => (
                <div key={a} className="axis-val"><span>{latest ? latest[i].toFixed(2) : '—'}</span><label>{a}</label></div>
              ))}
            </div>
            <div className="chart-box"><SensorChart samples={sensor.chartData} offset={0} unit="m/s²" /></div>
          </div>
          <div>
            <h3>Gyroscope <span className="muted small">rad/s</span></h3>
            <div className="axis-vals">
              {['X', 'Y', 'Z'].map((a, i) => (
                <div key={a} className="axis-val"><span>{latest ? latest[i + 3].toFixed(2) : '—'}</span><label>{a}</label></div>
              ))}
            </div>
            <div className="chart-box"><SensorChart samples={sensor.chartData} offset={3} unit="rad/s" /></div>
          </div>
        </div>
      </section>

      {/* Probabilities */}
      <section className="card span-5">
        <h2><Gauge size={16} /> Model probabilities <span className="muted small">latest window</span></h2>
        <ProbabilityBars probs={latestPrediction?.all_probs} />
      </section>
    </div>
  );
}
