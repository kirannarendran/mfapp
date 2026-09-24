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

  // Copy configs
  const configs = src.prepare('SELECT * FROM config').all();
  const insertConfig = dest.prepare('INSERT INTO config VALUES (?, ?, ?)');
  configs.forEach(c => insertConfig.run(c.key, c.value, c.updated_at));

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

  // Copy benchmark nav history (100484) + top 100 funds with metrics
  const topFunds = src.prepare(`
    SELECT scheme_code FROM fund_metrics 
    WHERE sharpe IS NOT NULL 
    ORDER BY sharpe DESC 
    LIMIT 100
  `).all().map(r => r.scheme_code);

  const targetCodes = [100484, ...topFunds];
  console.log(`[Snapshot] Copying historical NAVs for benchmark and top ${topFunds.length} funds...`);
  const navStmt = src.prepare('SELECT * FROM nav_history WHERE scheme_code = ?');
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
