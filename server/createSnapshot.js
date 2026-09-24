import Database from 'better-sqlite3';
import { existsSync, unlinkSync, statSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const SRC_PATH = join(__dirname, 'data', 'mf_tracker.db');
const DEST_PATH = join(__dirname, 'seed_snapshot.db');

export function createSnapshot() {
  if (!existsSync(SRC_PATH)) {
    console.error(`[Snapshot] Source database not found at ${SRC_PATH}`);
    process.exit(1);
  }

  console.log('[Snapshot] Reading from live database...');
  const src = new Database(SRC_PATH);

  if (existsSync(DEST_PATH)) {
    unlinkSync(DEST_PATH);
  }

  const dest = new Database(DEST_PATH);

  dest.exec(`
    CREATE TABLE funds (
      scheme_code INTEGER PRIMARY KEY,
      scheme_name TEXT NOT NULL,
      fund_house TEXT,
      category TEXT,
      type TEXT,
      isin TEXT,
      last_nav_date TEXT,
      last_nav REAL,
      last_updated TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE nav_history (
      scheme_code INTEGER NOT NULL,
      date TEXT NOT NULL,
      nav REAL NOT NULL,
      PRIMARY KEY (scheme_code, date)
    ) WITHOUT ROWID;
    CREATE INDEX idx_nav_history_date ON nav_history(date);
    CREATE TABLE fund_metrics (
      scheme_code INTEGER PRIMARY KEY,
      return_6m REAL,
      cagr_1y REAL,
      cagr_3y REAL,
      cagr_5y REAL,
      alpha REAL,
      beta REAL,
      sharpe REAL,
      sortino REAL,
      std_dev REAL,
      alpha_5y REAL,
      beta_5y REAL,
      sharpe_5y REAL,
      sortino_5y REAL,
      std_dev_5y REAL,
      upside_capture REAL,
      downside_capture REAL,
      upside_capture_3y REAL,
      downside_capture_3y REAL,
      computed_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE config (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE users (
      id TEXT PRIMARY KEY,
      google_id TEXT UNIQUE NOT NULL,
      email TEXT UNIQUE NOT NULL,
      name TEXT,
      first_name TEXT,
      last_name TEXT,
      avatar_url TEXT,
      age INTEGER,
      profession TEXT,
      investment_experience TEXT,
      monthly_investment_bracket TEXT,
      profile_completed INTEGER DEFAULT 0,
      created_at TEXT DEFAULT (datetime('now')),
      last_login_at TEXT DEFAULT (datetime('now'))
    );
  `);

  // Copy configs and ensure fresh last_successful_sync timestamp
  const configs = src.prepare('SELECT * FROM config').all();
  const insertConfig = dest.prepare('INSERT INTO config VALUES (?, ?, ?)');
  const nowIso = new Date().toISOString();
  let hasSyncKey = false;
  configs.forEach(c => {
    if (c.key === 'last_successful_sync') {
      hasSyncKey = true;
      insertConfig.run('last_successful_sync', Date.now().toString(), nowIso);
    } else {
      insertConfig.run(c.key, c.value, c.updated_at);
    }
  });
  if (!hasSyncKey) {
    insertConfig.run('last_successful_sync', Date.now().toString(), nowIso);
  }

  // Copy all funds
  const funds = src.prepare('SELECT * FROM funds').all();
  console.log(`[Snapshot] Copying ${funds.length} funds...`);
  const insertFund = dest.prepare('INSERT INTO funds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)');
  dest.transaction(() => {
    funds.forEach(f => insertFund.run(f.scheme_code, f.scheme_name, f.fund_house, f.category, f.type, f.isin, f.last_nav_date, f.last_nav, f.last_updated));
  })();

  // Copy all metrics
  const metrics = src.prepare('SELECT * FROM fund_metrics').all();
  console.log(`[Snapshot] Copying ${metrics.length} precomputed metric records...`);
  const insertMetric = dest.prepare('INSERT INTO fund_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)');
  dest.transaction(() => {
    metrics.forEach(m => insertMetric.run(m.scheme_code, m.return_6m, m.cagr_1y, m.cagr_3y, m.cagr_5y, m.alpha, m.beta, m.sharpe, m.sortino, m.std_dev, m.alpha_5y, m.beta_5y, m.sharpe_5y, m.sortino_5y, m.std_dev_5y, m.upside_capture, m.downside_capture, m.upside_capture_3y, m.downside_capture_3y, m.computed_at));
  })();

  // Curate comprehensive list of key funds for NAV history:
  // 1. Top 200 by Sharpe ratio
  const topSharpe = src.prepare(`
    SELECT scheme_code FROM fund_metrics 
    WHERE sharpe IS NOT NULL 
    ORDER BY sharpe DESC 
    LIMIT 200
  `).all().map(r => r.scheme_code);

  // 2. High-volume popular fund houses across key categories
  const popularKeywords = ['parag parikh', 'quant', 'hdfc', 'sbi', 'icici', 'mirae', 'nippon', 'axis', 'kotak', 'motilal'];
  const popularFunds = [];
  for (const kw of popularKeywords) {
    const matched = src.prepare(`
      SELECT scheme_code FROM funds 
      WHERE scheme_name LIKE ? 
      AND (scheme_name LIKE '%flexi%' OR scheme_name LIKE '%small cap%' OR scheme_name LIKE '%mid cap%' OR scheme_name LIKE '%large%' OR scheme_name LIKE '%index%')
      LIMIT 25
    `).all(`%${kw}%`).map(r => r.scheme_code);
    popularFunds.push(...matched);
  }

  // 3. Guarantee benchmark (100484) and sample portfolio funds
  const targetCodes = Array.from(new Set([100484, 122639, 120503, 118989, ...topSharpe, ...popularFunds]));
  console.log(`[Snapshot] Copying 5-year historical NAVs for ${targetCodes.length} curated funds + benchmark...`);

  // We keep last 5 years of daily NAVs to maximize coverage while staying under GitHub size limits
  const navStmt = src.prepare(`
    SELECT scheme_code, date, nav FROM nav_history 
    WHERE scheme_code = ? AND date >= date('now', '-5 years')
    ORDER BY date ASC
  `);
  const insertNav = dest.prepare('INSERT INTO nav_history VALUES (?, ?, ?)');

  dest.transaction(() => {
    for (const code of targetCodes) {
      const rows = navStmt.all(code);
      rows.forEach(n => insertNav.run(n.scheme_code, n.date, n.nav));
    }
  })();

  dest.pragma('vacuum');
  dest.close();
  src.close();

  const finalSize = statSync(DEST_PATH).size;
  console.log(`[Snapshot] ✅ Snapshot created at ${DEST_PATH} (${(finalSize / 1024 / 1024).toFixed(2)} MB)`);
}

// Run directly if invoked from CLI
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  createSnapshot();
}
