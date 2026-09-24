import cron from 'node-cron';
import { syncFundRegistry, syncBenchmarkData, syncAllTrackedFunds } from './dataSync.js';
import { recomputeAllMetrics } from './metricsEngine.js';
import { fetchAndUpdateRiskFreeRate } from './rbiRateFetcher.js';
import { getDB, initDB } from '../db.js';

const isProduction = process.env.NODE_ENV === 'production' || process.env.RENDER === 'true';
const isSyncDisabled = process.env.DISABLE_BACKGROUND_SYNC === 'true' || isProduction;

let isSyncing = false;

export const syncState = {
  currentStep: '',
  progress: 0,
  total: 0,
  startTime: null
};

export function getSyncStatus() {
  let lastSyncTime = null;
  let lastSyncDate = null;
  
  try {
    const db = getDB();
    const row = db.prepare(`SELECT value, updated_at FROM config WHERE key = 'last_successful_sync'`).get();
    if (row && (row.value || row.updated_at)) {
      lastSyncTime = row.value ? parseInt(row.value, 10) : null;
      lastSyncDate = row.updated_at;
    } else {
      // Fallback: Check the latest fund record in the database
      const fundRow = db.prepare('SELECT max(last_updated) as latest FROM funds').get();
      if (fundRow && fundRow.latest) {
        lastSyncDate = fundRow.latest;
        lastSyncTime = new Date(fundRow.latest.includes('T') ? fundRow.latest : fundRow.latest.replace(' ', 'T') + 'Z').getTime();
      }
    }
  } catch (err) {
    // Database might not be fully initialized yet
  }

  return {
    isSyncing,
    lastSyncTime,
    lastSyncDate,
    mode: isSyncDisabled ? 'snapshot' : 'local',
    state: isSyncing ? syncState : null
  };
}

export async function runFullSync() {
  initDB();
  if (isSyncing) {
    console.log('[Scheduler] Sync is already running. Skipping request.');
    return;
  }
  
  isSyncing = true;
  const start = Date.now();
  syncState.startTime = start;
  syncState.progress = 0;
  syncState.total = 0;
  console.log(`[Scheduler] Daily sync started at ${new Date().toISOString()}`);

  // Step 1: Fetch risk-free rate
  try {
    syncState.currentStep = 'Fetching risk-free rate...';
    console.log('[Scheduler] Step 1/5: Fetching risk-free rate...');
    await fetchAndUpdateRiskFreeRate();
    console.log('[Scheduler] ✓ Risk-free rate updated');
  } catch (err) {
    console.error('[Scheduler] ✗ Risk-free rate fetch failed:', err);
  }

  // Step 2: Sync fund registry
  try {
    syncState.currentStep = 'Syncing fund registry...';
    console.log('[Scheduler] Step 2/5: Syncing fund registry...');
    const count = await syncFundRegistry();
    console.log(`[Scheduler] ✓ Fund registry synced (${count} funds)`);
  } catch (err) {
    console.error('[Scheduler] ✗ Fund registry sync failed:', err);
  }

  // Step 3: Sync benchmark data
  try {
    syncState.currentStep = 'Syncing benchmark data...';
    console.log('[Scheduler] Step 3/5: Syncing benchmark data...');
    await syncBenchmarkData();
    console.log('[Scheduler] ✓ Benchmark data synced');
  } catch (err) {
    console.error('[Scheduler] ✗ Benchmark sync failed:', err);
  }

  // Step 4: Sync all tracked funds
  try {
    syncState.currentStep = 'Syncing tracked fund NAVs...';
    console.log('[Scheduler] Step 4/5: Syncing tracked fund NAVs...');
    const result = await syncAllTrackedFunds(syncState);
    console.log(`[Scheduler] ✓ Tracked funds synced (${JSON.stringify(result)})`);
  } catch (err) {
    console.error('[Scheduler] ✗ Tracked funds sync failed:', err);
  }

  // Step 5: Recompute all metrics
  try {
    syncState.currentStep = 'Recomputing metrics...';
    console.log('[Scheduler] Step 5/5: Recomputing metrics...');
    await recomputeAllMetrics();
    console.log('[Scheduler] ✓ Metrics recomputed');
  } catch (err) {
    console.error('[Scheduler] ✗ Metrics recompute failed:', err);
  }

  // Record successful sync
  try {
    const db = getDB();
    db.prepare(`INSERT OR REPLACE INTO config (key, value, updated_at) VALUES ('last_successful_sync', ?, datetime('now'))`).run(Date.now().toString());
  } catch (err) {
    console.error('[Scheduler] ✗ Failed to update last_successful_sync:', err);
  }

  isSyncing = false;
  const duration = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`[Scheduler] Daily sync completed in ${duration}s`);
}

function getLastExpectedRun() {
  const now = new Date();
  let expected = new Date(now);
  expected.setUTCHours(17, 30, 0, 0); // 17:30 UTC = 11:00 PM IST

  if (now.getTime() < expected.getTime()) {
    // Expected run for today hasn't happened yet, so look at yesterday
    expected.setUTCDate(expected.getUTCDate() - 1);
  }

  // If the expected run falls on Saturday (6) or Sunday (0), move back to Friday
  while (expected.getUTCDay() === 0 || expected.getUTCDay() === 6) {
    expected.setUTCDate(expected.getUTCDate() - 1);
  }

  return expected;
}

export function checkMissedSync() {
  if (isSyncDisabled || process.env.SKIP_SYNC === 'true') {
    console.log('[Scheduler] Snapshot/Production mode active. Skipping automatic background sync on boot.');
    return;
  }
  try {
    const db = getDB();
    const row = db.prepare(`SELECT value, updated_at FROM config WHERE key = 'last_successful_sync'`).get();
    const lastSyncTime = row ? parseInt(row.value, 10) : 0;

    // Check if DB is empty (first boot / fresh deploy) — always sync in this case
    const fundCount = db.prepare('SELECT COUNT(*) as count FROM funds').get();
    const isEmpty = !fundCount || fundCount.count === 0;

    if (isEmpty) {
      console.log('[Scheduler] Database is empty (fresh deploy). Triggering full sync...');
      runFullSync().catch(err => console.error('[Scheduler] First-boot sync error:', err));
      return;
    }

    // Same-day guard: if we already synced today, skip repeated restarts.
    if (lastSyncTime > 0) {
      const lastSyncDate = new Date(lastSyncTime).toDateString();
      const todayDate = new Date().toDateString();
      if (lastSyncDate === todayDate) {
        console.log(`[Scheduler] Already synced today (${new Date(lastSyncTime).toISOString()}). Skipping catch-up sync.`);
        return;
      }
    }

    const lastExpected = getLastExpectedRun().getTime();

    if (lastSyncTime < lastExpected) {
      console.log(`[Scheduler] Missed a scheduled sync (Expected: ${new Date(lastExpected).toISOString()}). Triggering catch-up...`);
      runFullSync().catch(err => console.error('[Scheduler] Catch-up sync error:', err));
    } else {
      console.log(`[Scheduler] Up to date (Last sync: ${new Date(lastSyncTime).toISOString()})`);
    }
  } catch (err) {
    console.error('[Scheduler] Failed to check for missed sync:', err);
  }
}

export function startScheduler() {
  if (isSyncDisabled) {
    console.log('[Scheduler] Production mode: Cron background sync disabled. Database is served from verified snapshot.');
    return;
  }

  // 5:30 PM UTC = 11:00 PM IST, weekdays only
  cron.schedule('30 17 * * 1-5', async () => {
    await runFullSync();
  });

  console.log('[Scheduler] Cron job registered: 5:30 PM UTC (11:00 PM IST) on weekdays');
}
