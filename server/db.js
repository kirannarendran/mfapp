import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdirSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dataDir = join(__dirname, 'data');
mkdirSync(dataDir, { recursive: true });

const DB_PATH = join(dataDir, 'mf_tracker.db');

let db = null;

export function initDB() {
  if (db) return db;

  db = new Database(DB_PATH);

  db.pragma('journal_mode = WAL');

  db.exec(`
    CREATE TABLE IF NOT EXISTS funds (
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

    CREATE TABLE IF NOT EXISTS nav_history (
      scheme_code INTEGER NOT NULL,
      date TEXT NOT NULL,
      nav REAL NOT NULL,
      PRIMARY KEY (scheme_code, date)
    ) WITHOUT ROWID;

    CREATE INDEX IF NOT EXISTS idx_nav_history_date ON nav_history(date);

    CREATE TABLE IF NOT EXISTS fund_metrics (
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

    CREATE TABLE IF NOT EXISTS config (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT DEFAULT (datetime('now'))
    );
  `);

  const columnsToAdd = [
    'return_6m REAL',
    'alpha_5y REAL', 'beta_5y REAL', 'sharpe_5y REAL', 'sortino_5y REAL', 'std_dev_5y REAL',
    'upside_capture_3y REAL', 'downside_capture_3y REAL'
  ];
  for (const col of columnsToAdd) {
    try {
      db.exec(`ALTER TABLE fund_metrics ADD COLUMN ${col}`);
    } catch (e) {
      // Ignore if column already exists
    }
  }

  db.prepare(`INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)`).run('risk_free_rate', '0.07');
  db.prepare(`INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)`).run('benchmark_code', '100484');

  return db;
}

export function getDB() {
  if (!db) {
    throw new Error('Database not initialized. Call initDB() first.');
  }
  return db;
}
