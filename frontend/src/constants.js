// One fixed colour per activity (validated categorical palette for the dark surface).
// Colour follows the activity everywhere: timeline, breakdown, probabilities.
export const ACTIVITY_COLORS = {
  Walking: '#3987e5',
  Jogging: '#d95926',
  Still: '#199e70',
  Stairs: '#c98500',
  'Hand Activity': '#d55181',
  Sports: '#9085e9',
  Uncertain: '#6b7280',
};

export const colorFor = (activity) => ACTIVITY_COLORS[activity] || '#6b7280';

// Must match backend/preprocessing.py
export const FS = 20;
export const FRAME_SIZE = 60;
export const STEP_SIZE = 30;
export const SECONDS_PER_PREDICTION = STEP_SIZE / FS;

export const RULE_LABELS = {
  'low-variance-still': 'Phone is almost motionless, so the low-variance rule set this to Still',
  'intensity-gate': 'Motion was too gentle for the model’s top choice, so the intensity rule picked the best allowed activity',
};

export const SOURCE_LABELS = {
  live: 'Live sensors',
  replay: 'Replayed recording',
  csv: 'CSV upload',
};

export function formatDuration(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(h ? 2 : 1, '0');
  const ss = String(sec).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

/** Seconds per activity from a list of predictions (each = one 1.5 s window step). */
export function secondsByActivity(predictions) {
  const out = {};
  for (const p of predictions) out[p.activity] = (out[p.activity] || 0) + SECONDS_PER_PREDICTION;
  return Object.fromEntries(Object.entries(out).sort((a, b) => b[1] - a[1]));
}

/** Merge consecutive predictions with the same activity into segments. */
export function toSegments(predictions) {
  const segs = [];
  for (const p of predictions) {
    const end = p.t_sec + SECONDS_PER_PREDICTION;
    const last = segs[segs.length - 1];
    if (last && last.activity === p.activity && p.t_sec - last.end < SECONDS_PER_PREDICTION * 2) {
      last.end = end;
      last.n += 1;
      last.conf += p.confidence;
    } else {
      segs.push({ activity: p.activity, start: p.t_sec, end, n: 1, conf: p.confidence });
    }
  }
  return segs;
}
