import { useMemo } from 'react';
import {
  CategoryScale, Chart as ChartJS, LineElement, LinearScale, PointElement, Tooltip,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { colorFor, formatDuration } from '../constants';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip);

/** Confidence of each prediction over time; points coloured by predicted activity. */
export default function ConfidenceChart({ predictions }) {
  const data = useMemo(() => ({
    datasets: [{
      label: 'Confidence',
      data: predictions.map((p) => ({ x: p.t_sec, y: p.confidence * 100, activity: p.activity })),
      borderColor: 'rgba(148, 163, 184, 0.5)',
      borderWidth: 2,
      pointRadius: predictions.length > 150 ? 0 : 3,
      pointHoverRadius: 5,
      pointBackgroundColor: predictions.map((p) => colorFor(p.activity)),
      pointBorderWidth: 0,
      tension: 0.2,
    }],
  }), [predictions]);

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
    scales: {
      x: {
        type: 'linear',
        grid: { display: false },
        ticks: { color: '#94a3b8', font: { size: 10 }, callback: (v) => formatDuration(v), maxTicksLimit: 6 },
      },
      y: {
        min: 0, max: 100,
        grid: { color: 'rgba(148, 163, 184, 0.12)' },
        border: { display: false },
        ticks: { color: '#94a3b8', font: { size: 10 }, callback: (v) => `${v}%`, stepSize: 25 },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => formatDuration(items[0].parsed.x),
          label: (c) => `${c.raw.activity}: ${c.parsed.y.toFixed(0)}% confidence`,
        },
      },
    },
  }), []);

  if (!predictions.length) return <div className="chart-empty muted small">No predictions yet.</div>;
  return <Line data={data} options={options} />;
}
