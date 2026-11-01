import axios from 'axios';

// Empty base URL = same origin (the Vite dev server proxies /api to Flask).
export const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const http = axios.create({ baseURL: API_BASE, timeout: 15000 });

/** Turn any axios error into a short human-readable message. */
export function errorMessage(err) {
  if (err?.response?.data?.error) return err.response.data.error;
  if (err?.code === 'ECONNABORTED') return 'The server took too long to respond.';
  if (err?.request && !err?.response) return 'Cannot reach the backend. Is `python app.py` running?';
  return err?.message || 'Something went wrong.';
}

export const api = {
  health: () => http.get('/api/health').then((r) => r.data),
  model: () => http.get('/api/model').then((r) => r.data),
  samples: () => http.get('/api/samples').then((r) => r.data.samples),
  sample: (id) => http.get(`/api/samples/${id}`).then((r) => r.data),

  predict: (data, sessionId, tSec, authToken) =>
    http
      .post('/api/predict', { data, session_id: sessionId, t_sec: tSec }, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
      })
      .then((r) => r.data),

  predictCsv: (file, { save, name }) => {
    const form = new FormData();
    form.append('file', file);
    if (save) form.append('save', 'true');
    if (name) form.append('name', name);
    return http.post('/api/predict/csv', form, { timeout: 120000 }).then((r) => r.data);
  },

  createSession: (name, source, meta) =>
    http.post('/api/sessions', { name, source, meta }).then((r) => r.data),
  endSession: (id, activeSeconds) =>
    http.post(`/api/sessions/${id}/end`, { active_seconds: activeSeconds }).then((r) => r.data),
  sessions: () => http.get('/api/sessions').then((r) => r.data.sessions),
  session: (id) => http.get(`/api/sessions/${id}`).then((r) => r.data),
  deleteSession: (id) => http.delete(`/api/sessions/${id}`).then((r) => r.data),
  exportCsvUrl: (id) => `${API_BASE}/api/sessions/${id}/export.csv`,
  exportPdfUrl: (id) => `${API_BASE}/api/sessions/${id}/report.pdf`,
};
