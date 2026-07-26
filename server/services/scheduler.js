import cron from 'node-cron';
import { syncFundRegistry, syncBenchmarkData, syncAllTrackedFunds } from './dataSync.js';
import { recomputeAllMetrics } from './metricsEngine.js';
import { fetchAndUpdateRiskFreeRate } from './rbiRateFetcher.js';
import { getDB } from '../db.js';

let isSyncing = false;

export function getSyncStatus() {
  let lastSyncTime = null;
  let lastSyncDate = null;
  
  try {
    const db = getDB();
    const row = db.prepare(`SELECT value, updated_at FROM config WHERE key = 'last_successful_sync'`).get();
    if (row) {
      lastSyncTime = parseInt(row.value, 10);
      lastSyncDate = row.updated_at;
    }
  } catch (err) {
    // Database might not be fully initialized yet
  }

  return {
    isSyncing,
    lastSyncTime,
    lastSyncDate
  };
}

export async function runFullSync() {
  if (isSyncing) {
    console.log('[Scheduler] Sync is already running. Skipping request.');
    return;
  }
  
  isSyncing = true;
  const start = Date.now();
  console.log(`[Scheduler] Daily sync started at ${new Date().toISOString()}`);

  // Step 1: Fetch risk-free rate
  try {
    console.log('[Scheduler] Step 1/5: Fetching risk-free rate...');
    await fetchAndUpdateRiskFreeRate();
    console.log('[Scheduler] ✓ Risk-free rate updated');
  } catch (err) {
    console.error('[Scheduler] ✗ Risk-free rate fetch failed:', err);
  }

  // Step 2: Sync fund registry
  try {
    console.log('[Scheduler] Step 2/5: Syncing fund registry...');
    const count = await syncFundRegistry();
    console.log(`[Scheduler] ✓ Fund registry synced (${count} funds)`);
  } catch (err) {
    console.error('[Scheduler] ✗ Fund registry sync failed:', err);
  }

  // Step 3: Sync benchmark data
  try {
    console.log('[Scheduler] Step 3/5: Syncing benchmark data...');
    await syncBenchmarkData();
    console.log('[Scheduler] ✓ Benchmark data synced');
  } catch (err) {
    console.error('[Scheduler] ✗ Benchmark sync failed:', err);
  }

  // Step 4: Sync all tracked funds
  try {
    console.log('[Scheduler] Step 4/5: Syncing tracked fund NAVs...');
    const result = await syncAllTrackedFunds();
    console.log(`[Scheduler] ✓ Tracked funds synced (${JSON.stringify(result)})`);
  } catch (err) {
    console.error('[Scheduler] ✗ Tracked funds sync failed:', err);
  }

  // Step 5: Recompute all metrics
  try {
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
  if (process.env.NODE_ENV === 'production' || process.env.SKIP_SYNC === 'true') {
    console.log('[Scheduler] Production/SKIP_SYNC mode detected. Skipping heavy catch-up sync to preserve resources.');
    return;
  }
  try {
    const db = getDB();
    const row = db.prepare(`SELECT value, updated_at FROM config WHERE key = 'last_successful_sync'`).get();
    const lastSyncTime = row ? parseInt(row.value, 10) : 0;

    // Same-day guard: if we already synced today (any time), skip.
    // This prevents repeated restarts during development from re-triggering the full sync.
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
      console.log(`[Scheduler] System started and missed a scheduled sync (Expected: ${new Date(lastExpected).toISOString()}). Triggering catch-up sync...`);
      // Run in background without blocking startup
      runFullSync().catch(err => console.error('[Scheduler] Catch-up sync error:', err));
    } else {
      console.log(`[Scheduler] System is up to date (Last sync: ${new Date(lastSyncTime).toISOString()})`);
    }
  } catch (err) {
    console.error('[Scheduler] Failed to check for missed sync:', err);
  }
}



export function startScheduler() {
  if (process.env.NODE_ENV === 'production') {
    console.log('[Scheduler] Production mode detected. Cron scheduler disabled.');
    return;
  }
  // 5:30 PM UTC = 11:00 PM IST, weekdays only
  cron.schedule('30 17 * * 1-5', async () => {
    await runFullSync();
  });

  console.log('[Scheduler] Cron job registered: 5:30 PM UTC (11:00 PM IST) on weekdays');
}
