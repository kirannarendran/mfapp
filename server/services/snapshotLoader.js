import Database from 'better-sqlite3';
import { existsSync, renameSync, rmSync, writeFileSync } from 'fs';
import { DB_PATH, SEED_SNAPSHOT_PATH } from '../db.js';

// Published nightly by .github/workflows/nightly-nav-sync.yml
const DEFAULT_SNAPSHOT_URL = 'https://github.com/kirannarendran/mfapp/releases/download/data-latest/seed_snapshot.db';
const DOWNLOAD_TIMEOUT_MS = 30000;
const MIN_EXPECTED_FUNDS = 1000;

function latestNavDate(path) {
  const db = new Database(path, { readonly: true });
  try {
    if (db.pragma('quick_check', { simple: true }) !== 'ok') throw new Error('integrity check failed');
    const { funds, latest } = db.prepare('SELECT COUNT(*) AS funds, MAX(last_nav_date) AS latest FROM funds').get();
    if (funds < MIN_EXPECTED_FUNDS) throw new Error(`only ${funds} funds`);
    return latest;
  } finally {
    db.close();
  }
}

/**
 * In snapshot mode, replace the bundled snapshot with the latest published one before the DB is seeded.
 * Any failure keeps the bundled snapshot, so the app always boots with valid (if older) data.
 */
export async function refreshSeedSnapshot() {
  const isProduction = process.env.NODE_ENV === 'production' || process.env.RENDER === 'true';
  if (!isProduction || existsSync(DB_PATH)) return;

  const tempPath = `${SEED_SNAPSHOT_PATH}.download`;
  try {
    const url = process.env.SNAPSHOT_URL || DEFAULT_SNAPSHOT_URL;
    const response = await fetch(url, { signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
    if (!response.ok) throw new Error(`download returned ${response.status}`);
    writeFileSync(tempPath, Buffer.from(await response.arrayBuffer()));

    const downloaded = latestNavDate(tempPath);
    const bundled = latestNavDate(SEED_SNAPSHOT_PATH);
    if (downloaded < bundled) throw new Error(`published NAVs (${downloaded}) are older than bundled (${bundled})`);

    renameSync(tempPath, SEED_SNAPSHOT_PATH);
    console.log(`[Snapshot] Using latest published snapshot (NAVs through ${downloaded})`);
  } catch (err) {
    console.warn(`[Snapshot] Keeping bundled snapshot: ${err.message}`);
    rmSync(tempPath, { force: true });
  }
}
