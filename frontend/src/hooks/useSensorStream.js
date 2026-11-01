import { useCallback, useEffect, useRef, useState } from 'react';
import { FRAME_SIZE, FS, STEP_SIZE } from '../constants';

const INTERVAL_MS = 1000 / FS;
const DEG_TO_RAD = Math.PI / 180;
const SENSOR_DETECT_TIMEOUT_MS = 2500;
const CHART_POINTS = 100;      // 5 s of samples in the live charts
const MAX_RAW_LOG = FS * 3600; // keep at most 1 hour of raw samples for export

/**
 * Streams 20 Hz samples [ax, ay, az, gx, gy, gz] from either
 *   - the phone's real motion sensors (mode 'live'), or
 *   - a recorded sample array replayed in real time (mode 'replay').
 * Calls onWindow(window60x6, tSec) for the first full window and then every 30 samples (1.5 s).
 * There is NO synthetic fallback: if sensors don't deliver data, start() throws.
 */
export function useSensorStream({ onWindow, onReplayEnd }) {
  const [status, setStatus] = useState('idle'); // idle | starting | running | paused
  const [mode, setMode] = useState(null);
  const [latest, setLatest] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [sampleCount, setSampleCount] = useState(0);
  const [hasGyro, setHasGyro] = useState(true);
  const [windowProgress, setWindowProgress] = useState(0);

  const latestRef = useRef(null);
  const gyroRef = useRef(false);
  const intervalRef = useRef(null);
  const windowRef = useRef([]);
  const sinceEmitRef = useRef(0);
  const emittedRef = useRef(false);
  const countRef = useRef(0);
  const rawLogRef = useRef([]);
  const chartRef = useRef([]);
  const replayRef = useRef(null);
  const modeRef = useRef(null);
  const wakeLockRef = useRef(null);
  const onWindowRef = useRef(onWindow);
  const onReplayEndRef = useRef(onReplayEnd);

  useEffect(() => {
    onWindowRef.current = onWindow;
    onReplayEndRef.current = onReplayEnd;
  });

  const handleMotion = useCallback((e) => {
    const a = e.accelerationIncludingGravity;
    if (!a || a.x == null) return;
    const r = e.rotationRate;
    gyroRef.current = !!(r && r.alpha != null);
    // rotationRate is in deg/s; the model was trained on rad/s.
    // W3C: beta = rotation about X, gamma = about Y, alpha = about Z.
    latestRef.current = [
      a.x, a.y, a.z,
      (r?.beta || 0) * DEG_TO_RAD, (r?.gamma || 0) * DEG_TO_RAD, (r?.alpha || 0) * DEG_TO_RAD,
    ];
  }, []);

  const clearTimer = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
  };

  const releaseWakeLock = () => {
    wakeLockRef.current?.release?.().catch(() => {});
    wakeLockRef.current = null;
  };

  const requestWakeLock = async () => {
    try {
      wakeLockRef.current = await navigator.wakeLock?.request('screen');
    } catch {
      /* not supported or denied: capture still works while the screen stays on */
    }
  };

  const tick = useCallback(() => {
    let sample;
    if (modeRef.current === 'replay') {
      const r = replayRef.current;
      if (!r || r.idx >= r.samples.length) {
        clearTimer();
        onReplayEndRef.current?.();
        return;
      }
      sample = r.samples[r.idx++];
    } else {
      sample = latestRef.current;
      if (!sample) return;
    }

    const buf = windowRef.current;
    buf.push(sample);
    if (buf.length > FRAME_SIZE) buf.shift();
    countRef.current += 1;
    sinceEmitRef.current += 1;
    if (rawLogRef.current.length < MAX_RAW_LOG) rawLogRef.current.push([Date.now(), ...sample]);

    const chart = chartRef.current;
    chart.push(sample);
    if (chart.length > CHART_POINTS) chart.shift();

    setLatest(sample);
    setChartData(chart.slice());
    setSampleCount(countRef.current);

    const needed = emittedRef.current ? STEP_SIZE : FRAME_SIZE;
    if (buf.length === FRAME_SIZE && sinceEmitRef.current >= needed) {
      sinceEmitRef.current = 0;
      emittedRef.current = true;
      const tSec = (countRef.current - FRAME_SIZE) / FS; // window start, in active seconds
      onWindowRef.current?.(buf.map((row) => row.slice()), tSec);
    }
    setWindowProgress(emittedRef.current ? sinceEmitRef.current / STEP_SIZE : buf.length / FRAME_SIZE);
  }, []);

  const resetBuffers = () => {
    windowRef.current = [];
    sinceEmitRef.current = 0;
    emittedRef.current = false;
    countRef.current = 0;
    rawLogRef.current = [];
    chartRef.current = [];
    setChartData([]);
    setSampleCount(0);
    setLatest(null);
    setWindowProgress(0);
  };

  /** Wait until the browser delivers a real motion event, or fail. */
  const detectSensor = () =>
    new Promise((resolve, reject) => {
      const started = Date.now();
      const check = setInterval(() => {
        if (latestRef.current) {
          clearInterval(check);
          resolve();
        } else if (Date.now() - started > SENSOR_DETECT_TIMEOUT_MS) {
          clearInterval(check);
          reject(new Error(
            'No motion-sensor data received. This device or browser does not provide an accelerometer ' +
            '(typical for laptops). Use a phone, replay a recording, or upload a CSV instead.'
          ));
        }
      }, 100);
    });

  /**
   * start({ mode: 'live' }) or start({ mode: 'replay', samples }).
   * `beforeStart` (async) runs after sensors are confirmed and before sampling begins,
   * e.g. to create the backend session. Must be called from a click handler (iOS permission).
   */
  const start = useCallback(async ({ mode: newMode, samples, beforeStart }) => {
    setStatus('starting');
    resetBuffers();
    try {
      if (newMode === 'live') {
        if (!window.isSecureContext) {
          throw new Error(
            'Motion sensors only work on HTTPS pages (or localhost). Start the frontend with ' +
            '`npm run dev:https` and open the https:// address on your phone.'
          );
        }
        if (typeof DeviceMotionEvent === 'undefined') {
          throw new Error('This browser does not support the DeviceMotion API.');
        }
        // iOS 13+: permission must be requested directly from the user's tap.
        if (typeof DeviceMotionEvent.requestPermission === 'function') {
          const permission = await DeviceMotionEvent.requestPermission();
          if (permission !== 'granted') throw new Error('Permission to use motion sensors was denied.');
        }
        latestRef.current = null;
        window.addEventListener('devicemotion', handleMotion);
        await detectSensor();
        setHasGyro(gyroRef.current);
      } else {
        if (!samples?.length) throw new Error('No recorded samples to replay.');
        replayRef.current = { samples, idx: 0 };
        setHasGyro(true);
      }

      await beforeStart?.();

      modeRef.current = newMode;
      setMode(newMode);
      setStatus('running');
      if (newMode === 'live') requestWakeLock();
      intervalRef.current = setInterval(tick, INTERVAL_MS);
      return { hasGyro: newMode === 'live' ? gyroRef.current : true };
    } catch (err) {
      window.removeEventListener('devicemotion', handleMotion);
      setStatus('idle');
      throw err;
    }
  }, [handleMotion, tick]);

  const pause = useCallback(() => {
    clearTimer();
    // Drop the partial window so no window ever spans the pause.
    windowRef.current = [];
    sinceEmitRef.current = 0;
    emittedRef.current = false;
    releaseWakeLock();
    setWindowProgress(0);
    setStatus('paused');
  }, []);

  const resume = useCallback(() => {
    if (modeRef.current === 'live') requestWakeLock();
    intervalRef.current = setInterval(tick, INTERVAL_MS);
    setStatus('running');
  }, [tick]);

  /** Stops streaming. Returns { activeSeconds, rawLog }. */
  const stop = useCallback(() => {
    clearTimer();
    window.removeEventListener('devicemotion', handleMotion);
    releaseWakeLock();
    const result = { activeSeconds: countRef.current / FS, rawLog: rawLogRef.current };
    modeRef.current = null;
    setStatus('idle');
    return result;
  }, [handleMotion]);

  useEffect(() => () => {
    clearTimer();
    window.removeEventListener('devicemotion', handleMotion);
    releaseWakeLock();
  }, [handleMotion]);

  return {
    status,
    mode,
    latest,
    chartData,
    hasGyro,
    activeSeconds: sampleCount / FS,
    windowProgress: Math.min(1, windowProgress),
    start,
    pause,
    resume,
    stop,
  };
}

/** Raw log rows [timestamp_ms, ax, ay, az, gx, gy, gz] -> CSV text (re-uploadable to /api/predict/csv). */
export function rawLogToCsv(rows) {
  const lines = ['timestamp_ms,ax,ay,az,gx,gy,gz'];
  for (const r of rows) lines.push([r[0], ...r.slice(1).map((v) => Number(v).toFixed(5))].join(','));
  return lines.join('\n');
}
