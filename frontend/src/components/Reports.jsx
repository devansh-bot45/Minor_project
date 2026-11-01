import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion as Motion } from 'framer-motion';
import { FileText, Share2, Loader2, Sparkles, UserPlus, BarChart3, Download, Flame, Clock, TrendingUp } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const ACTIVITY_COLORS = {
  "Walking": "rgba(94, 129, 172, 0.85)",     // Muted Blue
  "Jogging": "rgba(191, 97, 106, 0.85)",     // Muted Red
  "Stairs": "rgba(208, 135, 112, 0.85)",     // Muted Orange
  "Still": "rgba(163, 190, 140, 0.85)",      // Muted Green
  "Hand Activity": "rgba(235, 203, 139, 0.85)", // Muted Yellow
  "Sports": "rgba(136, 192, 208, 0.85)",     // Muted Cyan
};

const Reports = ({ backendUrl, authToken }) => {
  const [reports, setReports] = useState({ own: [], shared: [] });
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [shareEmail, setShareEmail] = useState('');
  const [shareTarget, setShareTarget] = useState(null);
  const [downloadingId, setDownloadingId] = useState(null);
  const [sharing, setSharing] = useState(false);
  
  const [activeTab, setActiveTab] = useState('my_reports');
  const [viewedSharedIds, setViewedSharedIds] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('viewedSharedIds') || '[]');
    } catch {
      return [];
    }
  });

  const unreadCount = reports.shared.filter(r => !viewedSharedIds.includes(r._id)).length;

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'shared_reports' && reports.shared.length > 0) {
      const allSharedIds = reports.shared.map(r => r._id);
      const newViewed = [...new Set([...viewedSharedIds, ...allSharedIds])];
      setViewedSharedIds(newViewed);
      localStorage.setItem('viewedSharedIds', JSON.stringify(newViewed));
    }
  };

  const fetchReports = async () => {
    try {
      const res = await axios.get(`${backendUrl}/api/reports`, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      setReports(res.data);
      
      // If we are currently on the shared tab, mark new ones as read immediately
      if (activeTab === 'shared_reports' && res.data.shared.length > 0) {
        const allSharedIds = res.data.shared.map(r => r._id);
        const newViewed = [...new Set([...viewedSharedIds, ...allSharedIds])];
        setViewedSharedIds(newViewed);
        localStorage.setItem('viewedSharedIds', JSON.stringify(newViewed));
      }
    } catch (err) {
      console.error('Failed to fetch reports', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    axios.get(`${backendUrl}/api/reports`, { headers: { Authorization: `Bearer ${authToken}` } })
      .then((res) => setReports(res.data))
      .catch((err) => console.error('Failed to fetch reports', err))
      .finally(() => setLoading(false));
  }, [backendUrl, authToken]);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      await axios.post(`${backendUrl}/api/reports/generate`, {}, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      await fetchReports();
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || 'Failed to generate report. Make sure Gemini API key is set.');
    } finally {
      setGenerating(false);
    }
  };

  const handleShare = async (reportId) => {
    if (!shareEmail) return;
    setSharing(true);
    try {
      await axios.post(`${backendUrl}/api/reports/share`, {
        report_id: reportId,
        email: shareEmail
      }, {
        headers: { Authorization: `Bearer ${authToken}` }
      });
      alert(`Report successfully shared with ${shareEmail}`);
      setShareEmail('');
      setShareTarget(null);
      fetchReports();
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || 'Failed to share report.');
    } finally {
      setSharing(false);
    }
  };

  const handleDownloadPdf = async (reportId) => {
    setDownloadingId(reportId);
    try {
      const res = await axios.get(`${backendUrl}/api/reports/${reportId}/pdf`, {
        headers: { Authorization: `Bearer ${authToken}` },
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Health_Report_${reportId}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      alert('Failed to download PDF. There might be an issue with the report data or the server.');
    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', paddingTop: '40px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px', gap: '20px', flexWrap: 'wrap' }}>
        <h2 style={{ fontSize: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
          <Sparkles size={24} color="var(--primary-color)"/> AI Health Reports
        </h2>
        <button className="btn-start" onClick={handleGenerate} disabled={generating} style={{ width: 'auto', padding: '10px 24px' }}>
          {generating ? <Loader2 size={18} className="spin" /> : <Sparkles size={18} />}
          {generating ? 'Generating...' : 'Generate Report'}
        </button>
      </div>

      <div style={{ display: 'flex', gap: '24px', marginBottom: '24px', borderBottom: '1px solid var(--card-border)' }}>
        <button 
          onClick={() => handleTabChange('my_reports')}
          style={{
            background: 'transparent',
            border: 'none',
            color: activeTab === 'my_reports' ? 'var(--text-main)' : 'var(--text-muted)',
            fontWeight: activeTab === 'my_reports' ? '600' : '400',
            borderBottom: activeTab === 'my_reports' ? '2px solid var(--text-main)' : '2px solid transparent',
            padding: '12px 4px',
            cursor: 'pointer',
            fontSize: '0.95rem',
            transition: 'all 0.2s ease'
          }}
        >
          My Reports
        </button>
        <button 
          onClick={() => handleTabChange('shared_reports')}
          style={{
            background: 'transparent',
            border: 'none',
            color: activeTab === 'shared_reports' ? 'var(--text-main)' : 'var(--text-muted)',
            fontWeight: activeTab === 'shared_reports' ? '600' : '400',
            borderBottom: activeTab === 'shared_reports' ? '2px solid var(--text-main)' : '2px solid transparent',
            padding: '12px 4px',
            cursor: 'pointer',
            fontSize: '0.95rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'all 0.2s ease'
          }}
        >
          <UserPlus size={16} /> Shared With Me
          {unreadCount > 0 && activeTab !== 'shared_reports' && (
            <span style={{
              background: 'rgba(239, 68, 68, 0.9)', // Muted red for professional look
              color: '#fff',
              borderRadius: '12px',
              padding: '2px 8px',
              fontSize: '0.7rem',
              fontWeight: '600',
              marginLeft: '4px'
            }}>
              {unreadCount}
            </span>
          )}
        </button>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px' }}><Loader2 className="spin" size={32} /></div>
      ) : (
        <>
          {activeTab === 'my_reports' && (
            <>
              {reports.own.length === 0 ? (
                <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No reports generated yet.</div>
              ) : (
                reports.own.map(r => (
                  <ReportCard 
                    key={r._id} 
                    report={r} 
                    isShared={false} 
                    shareTarget={shareTarget}
                    setShareTarget={setShareTarget}
                    shareEmail={shareEmail}
                    setShareEmail={setShareEmail}
                    handleShare={handleShare}
                    handleDownloadPdf={handleDownloadPdf}
                    sharing={sharing}
                    downloadingId={downloadingId}
                  />
                ))
              )}
            </>
          )}

          {activeTab === 'shared_reports' && (
            <>
              {reports.shared.length === 0 ? (
                <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No shared reports received yet.</div>
              ) : (
                reports.shared.map(r => (
                  <ReportCard 
                    key={r._id} 
                    report={r} 
                    isShared={true} 
                    shareTarget={shareTarget}
                    setShareTarget={setShareTarget}
                    shareEmail={shareEmail}
                    setShareEmail={setShareEmail}
                    handleShare={handleShare}
                    handleDownloadPdf={handleDownloadPdf}
                    sharing={sharing}
                    downloadingId={downloadingId}
                  />
                ))
              )}
            </>
          )}
        </>
      )}
    </div>
  );
};

const ReportCard = ({ report, isShared, shareTarget, setShareTarget, shareEmail, setShareEmail, handleShare, handleDownloadPdf, sharing, downloadingId }) => {
    let barData = null;
    let doughnutData = null;

    if (report.stats && report.stats.hourly) {
      const labels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
      const activities = Object.keys(report.stats.totals || {});
      
      const datasets = activities.map(act => ({
        label: act,
        data: labels.map((_, i) => report.stats.hourly[i]?.[act] || 0),
        backgroundColor: ACTIVITY_COLORS[act] || "rgba(100, 100, 100, 0.8)",
      }));

      barData = { labels, datasets };
    }

    if (report.stats && report.stats.totals) {
      const labels = Object.keys(report.stats.totals);
      const data = Object.values(report.stats.totals);
      const backgroundColors = labels.map(act => ACTIVITY_COLORS[act] || "rgba(100, 100, 100, 0.8)");
      
      doughnutData = {
        labels,
        datasets: [{
          data,
          backgroundColor: backgroundColors,
          borderWidth: 1,
          borderColor: 'transparent'
        }]
      };
    }

    const barOptions = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'Minutes', color: '#94a3b8' } }
      },
      plugins: {
        legend: { labels: { color: '#f8fafc' } }
      }
    };

    const doughnutOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: '#f8fafc' } }
      }
    };

    const cals = report.stats?.total_calories || 0;
    let intensity = 'Low';
    let intensityColor = '#a3be8c'; // Green
    if (cals > 450) {
      intensity = 'High';
      intensityColor = '#bf616a'; // Red
    } else if (cals >= 240) {
      intensity = 'Moderate';
      intensityColor = '#d08770'; // Orange
    }

    return (
      <Motion.div className="card" style={{ marginBottom: '20px', textAlign: 'left' }} initial={{opacity:0}} animate={{opacity:1}}>
        <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={18} color="var(--primary-color)" />
          {isShared ? `Shared Report from ${report.sender_name || report.user_id}` : `Daily Summary - ${report.date}`}
        </h3>
        
        {report.stats && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px', marginBottom: '20px' }}>
            <div style={{ background: 'rgba(235, 94, 94, 0.1)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(235, 94, 94, 0.2)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#ef4444' }}>
                <Flame size={18} /> <span style={{ fontWeight: '600', fontSize: '0.9rem' }}>Energy Burned</span>
              </div>
              <div style={{ fontSize: '1.8rem', fontWeight: '800', color: '#ef4444' }}>
                {report.stats.total_calories || 0} <span style={{ fontSize: '1rem', fontWeight: '400' }}>kcal</span>
              </div>
            </div>
            
            <div style={{ background: 'rgba(94, 129, 172, 0.1)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(94, 129, 172, 0.2)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#5e81ac' }}>
                <Clock size={18} /> <span style={{ fontWeight: '600', fontSize: '0.9rem' }}>Active Time</span>
              </div>
              <div style={{ fontSize: '1.8rem', fontWeight: '800', color: '#5e81ac' }}>
                {Math.round(Object.values(report.stats.totals || {}).reduce((a, b) => a + b, 0))} <span style={{ fontSize: '1rem', fontWeight: '400' }}>min</span>
              </div>
            </div>

            <div style={{ background: `${intensityColor}1a`, padding: '16px', borderRadius: '12px', border: `1px solid ${intensityColor}33`, display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: intensityColor }}>
                <TrendingUp size={18} /> <span style={{ fontWeight: '600', fontSize: '0.9rem' }}>Intensity</span>
              </div>
              <div style={{ fontSize: '1.8rem', fontWeight: '800', color: intensityColor }}>
                {intensity}
              </div>
            </div>
          </div>
        )}

        <div style={{ background: 'var(--input-bg)', padding: '16px', borderRadius: '8px', fontSize: '0.95rem', lineHeight: '1.6', whiteSpace: 'pre-wrap', color: 'var(--text-main)', border: '1px solid var(--card-border)' }}>
          {report.report_text}
        </div>

        {report.stats && (
          <div style={{ marginTop: '20px', padding: '16px', background: 'var(--input-bg)', borderRadius: '8px', border: '1px solid var(--card-border)' }}>
            <h4 style={{ fontSize: '1rem', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--secondary-color)' }}>
              <BarChart3 size={18} /> Activity Timeline
            </h4>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
              {barData && (
                <div style={{ width: '100%', height: '250px' }}>
                  <Bar data={barData} options={barOptions} />
                </div>
              )}
              {doughnutData && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ flex: '1', height: '200px' }}>
                    <Doughnut data={doughnutData} options={doughnutOptions} />
                  </div>
                  <div style={{ flex: '1', color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
                    <strong>Total Activity Overview:</strong>
                    <p style={{ marginTop: '8px' }}>This chart breaks down your total daily movement into proportional segments.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--card-border)', display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', justifyContent: 'space-between' }}>
          
          <button 
            className="btn-stop" 
            onClick={() => handleDownloadPdf(report._id)} 
            disabled={downloadingId === report._id}
            style={{ width: 'auto', padding: '8px 16px', display: 'flex', gap: '8px', alignItems: 'center' }}
          >
            {downloadingId === report._id ? <Loader2 size={16} className="spin" /> : <Download size={16} />}
            {downloadingId === report._id ? 'Downloading...' : 'Download PDF'}
          </button>

          {!isShared && (
            <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end' }}>
              {shareTarget === report._id ? (
                <div style={{ display: 'flex', gap: '8px', width: '100%', maxWidth: '400px' }}>
                  <input 
                    type="email" 
                    placeholder="Enter user email..." 
                    value={shareEmail}
                    onChange={(e) => setShareEmail(e.target.value)}
                    style={{ flex: 1, padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--card-border)', background: 'var(--bg-color)', color: 'var(--text-main)' }}
                  />
                  <button 
                    onClick={() => handleShare(report._id)} 
                    disabled={sharing}
                    style={{ padding: '8px 16px', flex: 'none', background: 'var(--primary-color)', color: 'white', border: 'none', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}
                  >
                    {sharing ? <Loader2 size={16} className="spin" /> : 'Share'}
                  </button>
                  <button className="btn-stop" onClick={() => setShareTarget(null)} style={{ padding: '8px 16px', flex: 'none' }}>
                    Cancel
                  </button>
                </div>
              ) : (
                <button className="btn-stop" onClick={() => setShareTarget(report._id)} style={{ width: 'auto', padding: '8px 16px' }}>
                  <Share2 size={16} /> Share Report
                </button>
              )}
            </div>
          )}
        </div>

        {!isShared && report.shared_with && report.shared_with.length > 0 && (
          <div style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'right' }}>
            Shared with: {report.shared_with.join(', ')}
          </div>
        )}
      </Motion.div>
    );
};

// (ReportCard logic already ends above)

export default Reports;
