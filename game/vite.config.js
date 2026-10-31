import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

// Fixture mode helper: serve a few repo artefacts read-only so the demo works with the backend offline.
//   /fixture/livesky.json            -> ../data/livesky/latest.json
//   /fixture/futures/<event_key>.json -> ../registry/futures/fans/<event_key>.json
const ROOT = path.resolve(__dirname, '..');
function fixtures() {
  return {
    name: 'od-fixtures',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const u = decodeURIComponent((req.url || '').split('?')[0]);
        let file = null;
        if (u === '/fixture/livesky.json') file = path.join(ROOT, 'data/livesky/latest.json');
        const m = u.match(/^\/fixture\/futures\/((?:train|test):\d+)\.json$/);
        if (m) file = path.join(ROOT, 'registry/futures/fans', `${m[1]}.json`);
        if (!file) return next();
        if (!fs.existsSync(file)) { res.statusCode = 404; res.setHeader('Content-Type', 'application/json'); return res.end('{"error":{"code":"NOT_FOUND"}}'); }
        res.setHeader('Content-Type', 'application/json');
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [fixtures()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.OD_API || 'http://localhost:8000', changeOrigin: true },
    },
  },
  build: { chunkSizeWarningLimit: 1500 },
});
