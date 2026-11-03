import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { rateLimit } from 'express-rate-limit';
import { randomBytes } from 'crypto';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { initDB } from './db.js';
import fundsRouter from './routes/funds.js';
import advisorRouter from './routes/advisor.js';
import authRouter from './routes/auth.js';
import { startScheduler, checkMissedSync } from './services/scheduler.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

// Render sits behind one proxy hop; trust it so rate limits key on the real client IP
app.set('trust proxy', 1);

app.use(helmet({
  // CSP would need allowances for Google Sign-In, GA4 and Google avatars; left off until configured
  contentSecurityPolicy: false,
  // Google Sign-In uses a popup that must be able to message back to this window
  crossOriginOpenerPolicy: { policy: 'same-origin-allow-popups' },
}));

// The frontend is served from the same origin (and proxied by Vite in dev), so cross-origin
// access is off unless CORS_ORIGIN lists allowed origins (comma-separated)
app.use(cors({ origin: process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',').map(o => o.trim()) : false }));

const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 600,
  standardHeaders: 'draft-8',
  legacyHeaders: false,
  // The app polls sync status every 10s per open tab
  skip: (req) => req.path === '/sync/status',
  message: { error: 'Too many requests. Please try again in a few minutes.' },
});

// Each advisor call spends Groq tokens, so it gets a much tighter budget
const advisorLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  limit: 30,
  standardHeaders: 'draft-8',
  legacyHeaders: false,
  message: { error: 'AI request limit reached. Please try again later.' },
});

app.use('/api', apiLimiter);
app.use('/api/advisor', advisorLimiter);

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
app.use('/api/auth', authRouter);

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
  if (!process.env.JWT_SECRET) {
    // Never fall back to a known secret (tokens could be forged). A per-process random secret stays
    // secure but logs everyone out on each restart, so set JWT_SECRET in the environment.
    process.env.JWT_SECRET = randomBytes(48).toString('hex');
    console.warn('[Server] WARNING: JWT_SECRET is not set. Using a random secret; sessions will reset on every restart.');
  }
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
