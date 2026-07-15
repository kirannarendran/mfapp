import { initDB } from './db.js';
import { syncFundRegistry, syncBenchmarkData } from './services/dataSync.js';
import { fetchAndUpdateRiskFreeRate } from './services/rbiRateFetcher.js';

async function seed() {
  console.log('[Seed] Starting initial data population...');
  initDB();
  console.log('[Seed] Database initialized');

  console.log('[Seed] Step 1/3: Fetching risk-free rate...');
  await fetchAndUpdateRiskFreeRate();

  console.log('[Seed] Step 2/3: Syncing fund registry (Direct Growth only)...');
  const fundCount = await syncFundRegistry();
  console.log(`[Seed] Synced ${fundCount} funds`);

  console.log('[Seed] Step 3/3: Syncing benchmark data...');
  await syncBenchmarkData();

  console.log('[Seed] ✅ Initial seeding complete!');
  console.log('[Seed] Run `npm run server` to start the backend.');
  process.exit(0);
}

seed().catch(err => {
  console.error('[Seed] ❌ Failed:', err);
  process.exit(1);
});
