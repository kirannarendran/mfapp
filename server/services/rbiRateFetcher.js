import { getDB } from '../db.js';

// India 10-Year Government Bond yield (monthly) published by the St. Louis Fed. Needs no API key.
const YIELD_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDIRLTLT01STM';

// The series lags about two months, so anything much older means the source has stopped updating
const MAX_YIELD_AGE_DAYS = 180;

/**
 * Parse the "date,value" CSV and return the latest usable observation, or null.
 */
export function parseLatestYield(csv) {
  const rows = csv.trim().split('\n').slice(1).reverse();
  for (const row of rows) {
    const [date, value] = row.trim().split(',');
    const yieldPercent = parseFloat(value);
    // Sanity check: Indian G-Sec yield should be between 1% and 15%
    if (yieldPercent >= 1 && yieldPercent <= 15) {
      return { date, yieldPercent };
    }
  }
  return null;
}

/**
 * Fetch the India 10-Year Government Bond yield and update the
 * risk-free rate in the config table.
 *
 * Throws if no fresh yield is available, leaving the stored rate unchanged.
 */
export async function fetchAndUpdateRiskFreeRate() {
  console.log('[RBI Rate] Fetching India 10Y yield from FRED...');
  const response = await fetch(YIELD_CSV_URL, { signal: AbortSignal.timeout(15000) });
  if (!response.ok) {
    throw new Error(`Yield source returned ${response.status}; keeping existing risk-free rate`);
  }

  const latest = parseLatestYield(await response.text());
  if (!latest) {
    throw new Error('No usable yield in source data; keeping existing risk-free rate');
  }

  const ageDays = (Date.now() - new Date(latest.date).getTime()) / (24 * 3600 * 1000);
  if (ageDays > MAX_YIELD_AGE_DAYS) {
    throw new Error(`Latest yield is from ${latest.date}, too old to use; keeping existing risk-free rate`);
  }

  const decimal = latest.yieldPercent / 100;
  getDB().prepare(
    `UPDATE config SET value = ?, updated_at = datetime('now') WHERE key = 'risk_free_rate'`
  ).run(String(decimal));

  console.log(`[RBI Rate] Updated risk-free rate to ${decimal} (${latest.yieldPercent}% as of ${latest.date})`);
  return decimal;
}
