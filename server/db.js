import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdirSync, existsSync, copyFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dataDir = join(__dirname, 'data');
mkdirSync(dataDir, { recursive: true });

const DB_PATH = join(dataDir, 'mf_tracker.db');
const SEED_SNAPSHOT_PATH = join(__dirname, 'seed_snapshot.db');

let db = null;

export function initDB() {
  if (db) return db;

  // Auto-seed: If runtime database does not exist, copy the bundled seed snapshot
  const dbExists = existsSync(DB_PATH);
  if (!dbExists && existsSync(SEED_SNAPSHOT_PATH)) {
    console.log('[DB] No database found at runtime. Seeding from bundled snapshot...');
    copyFileSync(SEED_SNAPSHOT_PATH, DB_PATH);
  }

  db = new Database(DB_PATH);
  db.pragma('journal_mode = WAL');

  // Verify if existing database has fund data; if empty, restore from bundled snapshot
  try {
    const row = db.prepare("SELECT count(*) as count FROM sqlite_master WHERE type='table' AND name='funds'").get();
    if (row && row.count > 0) {
      const fundCount = db.prepare('SELECT count(*) as count FROM funds').get();
      if ((!fundCount || fundCount.count === 0) && existsSync(SEED_SNAPSHOT_PATH)) {
        console.log('[DB] Database is empty. Restoring from bundled seed snapshot...');
        db.close();
        copyFileSync(SEED_SNAPSHOT_PATH, DB_PATH);
        db = new Database(DB_PATH);
        db.pragma('journal_mode = WAL');
      }
    }
  } catch (err) {
    console.warn('[DB] Verification check warning:', err.message);
  }

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

    CREATE TABLE IF NOT EXISTS users (
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

  const userColumnsToAdd = [
    'first_name TEXT',
    'last_name TEXT',
    'age INTEGER',
    'profession TEXT',
    'investment_experience TEXT',
    'monthly_investment_bracket TEXT',
    'profile_completed INTEGER DEFAULT 0'
  ];
  for (const col of userColumnsToAdd) {
    try {
      db.exec(`ALTER TABLE users ADD COLUMN ${col}`);
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
