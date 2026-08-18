import { getDB } from '../db.js';

const API_BASE = 'https://api.mfapi.in/mf';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Convert date from 'dd-mm-yyyy' to 'YYYY-MM-DD' format.
 */
function convertDate(ddmmyyyy) {
  const [dd, mm, yyyy] = ddmmyyyy.split('-');
  return `${yyyy}-${mm}-${dd}`;
}

async function fetchWithRetry(url, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(8000) });
      if (response.ok) return response;
      console.warn(`[Fetch] Non-OK status ${response.status} for ${url}`);
    } catch (error) {
      console.warn(`[Fetch] Error fetching ${url}: ${error.message}. Retrying ${i + 1}/${retries}...`);
    }
    await sleep(2000 * (i + 1));
  }
  throw new Error(`Failed to fetch ${url} after ${retries} retries`);
}

/**
 * Fetch the full fund list from the upstream API and upsert
 * only Direct Growth funds into the local database.
 */
export async function syncFundRegistry() {
  try {
    const response = await fetchWithRetry(API_BASE);
    const allFunds = await response.json();

    const directGrowthFunds = allFunds.filter(fund => {
      const name = fund.schemeName.toLowerCase();
      return (
        name.includes('direct') &&
        name.includes('growth') &&
        !name.includes('idcw') &&
        !name.includes('dividend')
      );
    });

    const db = getDB();
    const insert = db.prepare(`
      INSERT INTO funds (scheme_code, scheme_name) VALUES (?, ?)
      ON CONFLICT(scheme_code) DO UPDATE SET scheme_name = excluded.scheme_name
    `);

    const bulkInsert = db.transaction((funds) => {
      for (const fund of funds) {
        insert.run(fund.schemeCode, fund.schemeName);
      }
    });

    bulkInsert(directGrowthFunds);

    console.log(`[DataSync] Synced ${directGrowthFunds.length} Direct Growth funds`);
    return directGrowthFunds.length;
  } catch (error) {
    console.log(`[DataSync] Error syncing fund registry: ${error.message}`);
    throw error;
  }
}

/**
 * Fetch NAV history for a single scheme and insert new records
 * into the database. Also updates fund metadata.
 */
export async function syncNavData(schemeCode) {
  try {
    const db = getDB();

    // Check latest stored date
    const latest = db.prepare(
      `SELECT date FROM nav_history WHERE scheme_code = ? ORDER BY date DESC LIMIT 1`
    ).get(schemeCode);

    const latestDate = latest ? latest.date : null;

    // Fetch from upstream
    const response = await fetchWithRetry(`${API_BASE}/${schemeCode}`);
    const { meta, data } = await response.json();

    if (!data || data.length === 0) {
      console.log(`[DataSync] No NAV data returned for scheme ${schemeCode}`);
      return 0;
    }

    // Convert dates and filter to only new records
    const records = data.map(item => ({
      date: convertDate(item.date),
      nav: parseFloat(item.nav),
    }));

    const newRecords = latestDate
      ? records.filter(r => r.date > latestDate)
      : records;

    // Bulk insert new NAV records
    const insert = db.prepare(
      `INSERT OR IGNORE INTO nav_history (scheme_code, date, nav) VALUES (?, ?, ?)`
    );

    const bulkInsert = db.transaction((rows) => {
      for (const row of rows) {
        insert.run(schemeCode, row.date, row.nav);
      }
    });

    bulkInsert(newRecords);

    // Insert or update fund metadata from meta field
    db.prepare(`
      INSERT INTO funds (scheme_code, scheme_name, fund_house, category, type, isin, last_nav, last_nav_date, last_updated)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
      ON CONFLICT(scheme_code) DO UPDATE SET
          scheme_name = excluded.scheme_name,
          fund_house = excluded.fund_house,
          category = excluded.category,
          type = excluded.type,
          isin = excluded.isin,
          last_nav = excluded.last_nav,
          last_nav_date = excluded.last_nav_date,
          last_updated = datetime('now')
    `).run(
      schemeCode,
      meta.scheme_name || `Scheme ${schemeCode}`,
      meta.fund_house || null,
      meta.scheme_category || null,
      meta.scheme_type || null,
      meta.isin_growth || null,
      records[0].nav,
      records[0].date
    );

    console.log(`[DataSync] Synced ${newRecords.length} new NAV records for scheme ${schemeCode}`);
    return newRecords.length;
  } catch (error) {
    console.log(`[DataSync] Error syncing NAV for scheme ${schemeCode}: ${error.message}`);
    throw error;
  }
}

/**
 * Sync NAV data for the configured benchmark fund.
 */
export async function syncBenchmarkData() {
  try {
    const row = getDB().prepare('SELECT value FROM config WHERE key = ?').get('benchmark_code');
    if (!row) {
      throw new Error('benchmark_code not found in config');
    }

    const benchmarkCode = parseInt(row.value, 10);
    console.log(`[DataSync] Syncing benchmark data for scheme ${benchmarkCode}`);
    return await syncNavData(benchmarkCode);
  } catch (error) {
    console.log(`[DataSync] Error syncing benchmark data: ${error.message}`);
    throw error;
  }
}

/**
 * Sync NAV data for all tracked funds with rate limiting.
 */
export async function syncAllTrackedFunds(syncState = null) {
  try {
    const db = getDB();
    const funds = db.prepare('SELECT scheme_code, last_updated, fund_house FROM funds').all();
    const total = funds.length;

    console.log(`[DataSync] Starting sync for ${total} tracked funds`);

    if (syncState) {
      syncState.total = total;
    }

    let totalNewRecords = 0;
    const twelveHoursAgo = new Date(Date.now() - 12 * 60 * 60 * 1000);

    for (let i = 0; i < total; i++) {
      const { scheme_code, last_updated } = funds[i];

      // Optimization: Skip if updated in the last 12 hours AND already has metadata populated
      const hasMetadata = funds[i].fund_house != null;
      if (hasMetadata && last_updated && new Date(last_updated + 'Z') > twelveHoursAgo) {
        if (syncState) syncState.progress = i + 1;
        continue;
      }

      try {
        const newCount = await syncNavData(scheme_code);
        totalNewRecords += newCount;
      } catch (error) {
        console.log(`[DataSync] Failed to sync scheme ${scheme_code}: ${error.message}`);
      }

      if (syncState) {
        syncState.progress = i + 1;
      }

      if ((i + 1) % 100 === 0) {
        console.log(`[DataSync] Progress: ${i + 1}/${total} funds synced`);
      }

      if (i < total - 1) {
        await sleep(200);
      }
    }

    console.log(`[DataSync] Completed: ${totalNewRecords} total new NAV records across ${total} funds`);
    return totalNewRecords;
  } catch (error) {
    console.log(`[DataSync] Error in syncAllTrackedFunds: ${error.message}`);
    throw error;
  }
}
