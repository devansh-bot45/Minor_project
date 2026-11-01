import { useMemo } from 'react';
import {
  CategoryScale, Chart as ChartJS, Legend, LineElement, LinearScale, PointElement, Tooltip,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

// First three slots of the validated palette (safe for all pairs).
const AXIS_COLORS = ['#3987e5', '#d95926', '#199e70'];

/** Live 3-axis line chart. `samples` = rows [ax, ay, az, gx, gy, gz]; offset 0 = accel, 3 = gyro. */
export default function SensorChart({ samples, offset, unit }) {
  const data = useMemo(() => ({
    labels: samples.map((_, i) => i),
    datasets: ['X', 'Y', 'Z'].map((axis, k) => ({
      label: axis,
      data: samples.map((row) => row[offset + k]),
      borderColor: AXIS_COLORS[k],
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
    })),
  }), [samples, offset]);

  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: { display: false },
      y: {
        grid: { color: 'rgba(148, 163, 184, 0.12)' },
        border: { display: false },
        ticks: { color: '#94a3b8', font: { size: 10 }, maxTicksLimit: 5 },
        title: { display: true, text: unit, color: '#94a3b8', font: { size: 10 } },
      },
    },
    plugins: {
      legend: { position: 'top', align: 'end', labels: { color: '#cbd5e1', boxWidth: 10, boxHeight: 2 } },
      tooltip: {
        callbacks: { title: () => '', label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(2)} ${unit}` },
      },
    },
  }), [unit]);

  if (!samples.length) return <div className="chart-empty muted small">Waiting for samples…</div>;
  return <Line data={data} options={options} />;
}
