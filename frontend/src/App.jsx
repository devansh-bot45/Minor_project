import { useEffect, useState } from 'react';
import { Activity, Brain, FileUp, History, LogOut, Radio, Sparkles } from 'lucide-react';
import { api } from './api';
import LiveView from './views/LiveView';
import CsvView from './views/CsvView';
import HistoryView from './views/HistoryView';
import ModelView from './views/ModelView';
import Login from './components/Login';
import Reports from './components/Reports';

const TABS = [
  { id: 'live', label: 'Live', icon: Radio },
  { id: 'csv', label: 'Analyse CSV', icon: FileUp },
  { id: 'history', label: 'History', icon: History },
  { id: 'model', label: 'Model', icon: Brain },
];

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem('har_user'));
  } catch {
    return null;
  }
}

export default function App() {
  const [tab, setTab] = useState('live');
  const [health, setHealth] = useState(null); // null = checking, false = offline
  // Optional cloud sign-in (only used when the backend has Google/MongoDB/Gemini configured)
  const [authToken, setAuthToken] = useState(() => localStorage.getItem('har_auth_token'));
  const [user, setUser] = useState(readStoredUser);

  useEffect(() => {
    let cancelled = false;
    const check = () => api.health()
      .then((h) => !cancelled && setHealth(h))
      .catch(() => !cancelled && setHealth(false));
    check();
    const id = setInterval(check, 15000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const online = !!health;
  const cloud = online && health.cloud_reports_enabled;
  const tabs = cloud ? [...TABS, { id: 'reports', label: 'AI Reports', icon: Sparkles }] : TABS;

  const logout = () => {
    setAuthToken(null);
    setUser(null);
    localStorage.removeItem('har_auth_token');
    localStorage.removeItem('har_user');
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Activity size={22} />
          <div>
            <div className="brand-name">Activity Recognition</div>
            <div className="brand-sub">Conv1D + BiLSTM + Attention · phone sensors</div>
          </div>
        </div>
        <span className={`pill ${health === null ? 'pill-wait' : online ? 'pill-live' : 'pill-error'}`}
              title={online ? `Classes: ${health.classes.join(', ')}` : 'Start the Flask backend: python app.py'}>
          <span className="dot" />{health === null ? 'Checking backend…' : online ? 'Backend online' : 'Backend offline'}
        </span>
      </header>

      <nav className="tabs" aria-label="Sections">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button key={id} className={tab === id ? 'on' : ''} onClick={() => setTab(id)} aria-current={tab === id}>
            <Icon size={15} /> {label}
          </button>
        ))}
      </nav>

      {health === false && (
        <div className="banner error page-banner">
          The backend is not reachable. In a terminal run <code>cd backend && .venv/bin/python app.py</code> (see README),
          then this page reconnects automatically.
        </div>
      )}

      <main>
        {/* Live view stays mounted so a running session survives tab switches. */}
        <div hidden={tab !== 'live'}>
          <LiveView backendOnline={online} authToken={cloud ? authToken : null} onOpenHistory={() => setTab('history')} />
        </div>
        {tab === 'csv' && <CsvView backendOnline={online} onOpenHistory={() => setTab('history')} />}
        {tab === 'history' && <HistoryView backendOnline={online} />}
        {tab === 'model' && <ModelView backendOnline={online} />}
        {tab === 'reports' && cloud && (
          authToken ? (
            <div>
              <div className="signed-in">
                <span className="muted small">Signed in as {user?.email || user?.name}</span>
                <button className="btn small" onClick={logout}><LogOut size={14} /> Sign out</button>
              </div>
              <Reports backendUrl={import.meta.env.VITE_API_BASE_URL || ''} authToken={authToken} />
            </div>
          ) : (
            <Login backendUrl={import.meta.env.VITE_API_BASE_URL || ''} setAuthToken={setAuthToken} setUser={setUser} />
          )
        )}
      </main>

      <footer className="footer muted small">
        Predictions come from a machine-learning model and can be wrong. Not a medical device.
      </footer>
    </div>
  );
}
