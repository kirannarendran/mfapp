import { getDB } from '../db.js';

/**
 * Fetch the India 10-Year Government Bond yield and update the
 * risk-free rate in the config table.
 *
 * Strategy:
 *  1. Primary: Scrape from worldgovernmentbonds.com
 *  2. Fallback: Keep existing value unchanged
 */
export async function fetchAndUpdateRiskFreeRate() {
  let yieldPercent = null;

  // ── Attempt 1: World Government Bonds ────────────────────────────────────
  try {
    console.log('[RBI Rate] Fetching from worldgovernmentbonds.com...');
    const response = await fetch('https://www.worldgovernmentbonds.com/bond-yields/india/', {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html',
      },
    });

    if (response.ok) {
      const html = await response.text();

      // The page has yield values in elements. Look for patterns near "10 Years" or "10Y".
      // Typical pattern: a number like 6.89 or 7.12 near "10 Year" text
      // Try multiple regex patterns to be resilient to HTML structure changes
      const patterns = [
        /10\s*(?:Year|Y)[\s\S]{0,200}?(\d{1,2}\.\d{1,3})\s*%/i,
        /(\d{1,2}\.\d{1,3})\s*%[\s\S]{0,200}?10\s*(?:Year|Y)/i,
        /india-10-years[\s\S]{0,500}?(\d{1,2}\.\d{1,3})/i,
        /10Y[\s\S]{0,100}?(\d{1,2}\.\d{2,3})/i,
      ];

      for (const pattern of patterns) {
        const match = html.match(pattern);
        if (match) {
          const parsed = parseFloat(match[1]);
          // Sanity check: Indian G-Sec yield should be between 1% and 15%
          if (parsed >= 1 && parsed <= 15) {
            yieldPercent = parsed;
            console.log(`[RBI Rate] Found yield from worldgovernmentbonds.com: ${yieldPercent}%`);
            break;
          }
        }
      }
    }
  } catch (err) {
    console.warn(`[RBI Rate] worldgovernmentbonds.com fetch failed: ${err.message}`);
  }

  // ── Attempt 2: Trading Economics (fallback) ──────────────────────────────
  if (yieldPercent === null) {
    try {
      console.log('[RBI Rate] Trying tradingeconomics.com...');
      const response = await fetch('https://tradingeconomics.com/india/government-bond-yield', {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
          'Accept': 'text/html',
        },
      });

      if (response.ok) {
        const html = await response.text();
        // Look for the main yield value on the page
        const match = html.match(/id="p"[^>]*>(\d{1,2}\.\d{1,3})</i);
        if (match) {
          const parsed = parseFloat(match[1]);
          if (parsed >= 1 && parsed <= 15) {
            yieldPercent = parsed;
            console.log(`[RBI Rate] Found yield from tradingeconomics.com: ${yieldPercent}%`);
          }
        }
      }
    } catch (err) {
      console.warn(`[RBI Rate] tradingeconomics.com fetch failed: ${err.message}`);
    }
  }

  // ── Update DB or keep existing ───────────────────────────────────────────
  if (yieldPercent !== null) {
    const decimal = yieldPercent / 100;
    getDB().prepare(
      `UPDATE config SET value = ?, updated_at = datetime('now') WHERE key = 'risk_free_rate'`
    ).run(String(decimal));

    console.log(`[RBI Rate] Updated risk-free rate to ${decimal} (${yieldPercent}%)`);
    return decimal;
  } else {
    const existing = getDB().prepare('SELECT value FROM config WHERE key = ?').get('risk_free_rate');
    const currentRate = existing ? parseFloat(existing.value) : 0.07;
    console.warn(`[RBI Rate] Could not fetch updated rate, keeping existing value: ${currentRate} (${(currentRate * 100).toFixed(2)}%)`);
    return currentRate;
  }
}
