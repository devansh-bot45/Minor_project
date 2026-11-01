import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// `npm run dev`        -> http://localhost:5173 (desktop testing, CSV upload, replay)
// `npm run dev:https`  -> https://<your-LAN-IP>:5173 (needed for phone motion sensors)
// API calls go to the same origin and are proxied to the Flask backend, so the phone
// never needs to reach the backend port directly (and HTTPS pages can't call plain HTTP anyway).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const backend = env.BACKEND_URL || 'http://localhost:5001'
  const proxy = { '/api': backend, '/predict': backend }
  const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin-allow-popups',
    'Cross-Origin-Embedder-Policy': 'unsafe-none',
  }
  return {
    plugins: [react(), ...(mode === 'https' ? [basicSsl()] : [])],
    server: { host: true, port: 5173, proxy, headers },
    preview: { host: true, port: 4173, proxy, headers },
  }
})
