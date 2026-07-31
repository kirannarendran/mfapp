import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { initDB } from './db.js';
import fundsRouter from './routes/funds.js';
import advisorRouter from './routes/advisor.js';
import { startScheduler, checkMissedSync } from './services/scheduler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    console.log(`[HTTP] ${req.method} ${req.url} ${res.statusCode} - ${Date.now() - start}ms`);
  });
  next();
});
app.use(express.json());

// API Routes
app.use('/api', fundsRouter);
app.use('/api', advisorRouter);

// Serve static frontend in production
app.use(express.static(join(__dirname, '..', 'dist')));
app.use((req, res, next) => {
  if (!req.path.startsWith('/api')) {
    res.sendFile(join(__dirname, '..', 'dist', 'index.html'));
  } else {
    next();
  }
});

async function start() {
  initDB();
  console.log('[Server] Database initialized');
  startScheduler();
  console.log('[Server] Scheduler started');
  checkMissedSync();
  app.listen(PORT, () => {
    console.log(`[Server] Running on http://localhost:${PORT}`);
  });
}

start().catch(err => {
  console.error('[Server] Failed to start:', err);
  process.exit(1);
});
