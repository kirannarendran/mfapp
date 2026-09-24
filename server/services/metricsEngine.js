import { getDB } from '../db.js';

export const EQUITY_BENCHMARK_CODE = 100484; // Franklin India NSE Nifty 50 Index Fund
export const DEBT_BENCHMARK_CODE = 120137;   // SBI 10 Year Constant Maturity Gilt Fund

/**
 * Check if a fund category belongs to Debt / Fixed Income / Money Market
 */
export function isDebtCategory(category = '') {
  if (!category) return false;
  const lower = category.toLowerCase();
  return (
    lower.includes('debt') ||
    lower.includes('income') ||
    lower.includes('gilt') ||
    lower.includes('liquid') ||
    lower.includes('money market') ||
    lower.includes('treasury') ||
    lower.includes('bond') ||
    lower.includes('overnight') ||
    lower.includes('constant maturity') ||
    lower.includes('ultra short') ||
    lower.includes('low duration') ||
    lower.includes('short duration') ||
    lower.includes('medium duration') ||
    lower.includes('long duration') ||
    lower.includes('banking and psu') ||
    lower.includes('corporate bond') ||
    lower.includes('credit risk') ||
    lower.includes('floater') ||
    lower.includes('dynamic bond') ||
    lower.includes('dynamic term')
  );
}

/**
 * Get the current risk-free rate from the config table.
 */
function getRiskFreeRate() {
  const row = getDB().prepare('SELECT value FROM config WHERE key = ?').get('risk_free_rate');
  return row ? parseFloat(row.value) : 0.07;
}

/**
 * Get the benchmark scheme code for a given category.
 */
function getBenchmarkCodeForCategory(category = '') {
  if (isDebtCategory(category)) {
    return DEBT_BENCHMARK_CODE;
  }
  const row = getDB().prepare('SELECT value FROM config WHERE key = ?').get('benchmark_code');
  return row ? parseInt(row.value, 10) : EQUITY_BENCHMARK_CODE;
}

/**
 * Calculate daily percentage returns from an array of {nav} rows.
 * Expects rows sorted chronologically (oldest first).
 */
function getDailyReturns(navRows) {
  const returns = [];
  for (let i = 1; i < navRows.length; i++) {
    const current = navRows[i].nav || navRows[i].fund_nav || navRows[i].bench_nav;
    const prev = navRows[i - 1].nav || navRows[i - 1].fund_nav || navRows[i - 1].bench_nav;
    returns.push((current - prev) / prev);
  }
  return returns;
}

/**
 * Format a date N years ago as 'YYYY-MM-DD'.
 */
function cutoffDateISO(years) {
  const d = new Date();
  d.setFullYear(d.getFullYear() - years);
  return d.toISOString().split('T')[0];
}

// ─── CAGR ─────────────────────────────────────────────────────────────────────

/**
 * Calculate Compound Annual Growth Rate for a fund over N years.
 */
export function calculateCAGR(schemeCode, years) {
  const cutoff = cutoffDateISO(years);
  const rows = getDB().prepare(
    `SELECT date, nav FROM nav_history WHERE scheme_code = ? AND date >= ? ORDER BY date DESC`
  ).all(schemeCode, cutoff);

  if (rows.length < 2) return null;

  const currentNav = rows[0].nav;
  const oldNav = rows[rows.length - 1].nav;

  if (oldNav <= 0 || currentNav <= 0) return null;

  // Actual time difference in years
  const newestMs = new Date(rows[0].date).getTime();
  const oldestMs = new Date(rows[rows.length - 1].date).getTime();
  const actualTimeDiff = (newestMs - oldestMs) / (365.25 * 24 * 3600 * 1000);

  if (actualTimeDiff < 0.5) return null;

  const cagr = (Math.pow(currentNav / oldNav, 1 / actualTimeDiff) - 1) * 100;
  return isFinite(cagr) ? parseFloat(cagr.toFixed(2)) : null;
}

// ─── Absolute Return ──────────────────────────────────────────────────────────

/**
 * Calculate absolute return for a fund over N months.
 */
export function calculateAbsoluteReturn(schemeCode, months) {
  const d = new Date();
  d.setMonth(d.getMonth() - months);
  const cutoff = d.toISOString().split('T')[0];

  const rows = getDB().prepare(
    `SELECT date, nav FROM nav_history WHERE scheme_code = ? AND date >= ? ORDER BY date DESC`
  ).all(schemeCode, cutoff);

  if (rows.length < 2) return null;

  const currentNav = rows[0].nav;
  const oldNav = rows[rows.length - 1].nav;

  if (oldNav <= 0) return null;

  const ret = (((currentNav - oldNav) / oldNav) * 100);
  return isFinite(ret) ? parseFloat(ret.toFixed(2)) : null;
}

// ─── Risk Metrics (Alpha, Beta, Sharpe, Sortino, StdDev) ──────────────────────

/**
 * Calculate risk metrics for a fund over the last N years,
 * aligned with a benchmark on matching dates.
 */
export function calculateRiskMetrics(schemeCode, benchmarkCode, years) {
  const cutoff = cutoffDateISO(years);
  const riskFreeRate = getRiskFreeRate();

  // INNER JOIN to get aligned data (only dates where both fund and benchmark have NAVs)
  const rows = getDB().prepare(`
    SELECT f.date, f.nav as fund_nav, b.nav as bench_nav
    FROM nav_history f
    INNER JOIN nav_history b ON f.date = b.date
    WHERE f.scheme_code = ? AND b.scheme_code = ? AND f.date >= ?
    ORDER BY f.date ASC
  `).all(schemeCode, benchmarkCode, cutoff);

  // Approximate minimum data points based on trading days (252/year)
  if (rows.length < (years * 252 * 0.8)) return null;

  // Calculate daily returns
  const fundReturns = [];
  const benchReturns = [];
  for (let i = 1; i < rows.length; i++) {
    const fPrev = rows[i - 1].fund_nav;
    const bPrev = rows[i - 1].bench_nav;
    if (fPrev > 0 && bPrev > 0) {
      fundReturns.push((rows[i].fund_nav - fPrev) / fPrev);
      benchReturns.push((rows[i].bench_nav - bPrev) / bPrev);
    }
  }

  const n = fundReturns.length;
  if (n < (years * 252 * 0.75)) return null;

  // Mean daily returns
  const meanFundRet = fundReturns.reduce((a, b) => a + b, 0) / n;
  const meanBenchRet = benchReturns.reduce((a, b) => a + b, 0) / n;

  // A. Standard Deviation (Annualized)
  const variance = fundReturns.reduce((sum, r) => sum + Math.pow(r - meanFundRet, 2), 0) / (n - 1);
  const stdDevDaily = Math.sqrt(Math.max(variance, 0));
  const stdDevAnnual = stdDevDaily * Math.sqrt(252) * 100;

  // B. Beta = Covariance(Fund, Bench) / Variance(Bench)
  let covariance = 0;
  let benchVariance = 0;
  for (let i = 0; i < n; i++) {
    covariance += (fundReturns[i] - meanFundRet) * (benchReturns[i] - meanBenchRet);
    benchVariance += Math.pow(benchReturns[i] - meanBenchRet, 2);
  }
  const beta = benchVariance > 1e-9 ? covariance / benchVariance : 1.0;

  // C. Alpha (Jensen's Alpha)
  const annualFundRet = Math.pow(Math.max(1 + meanFundRet, 0.0001), 252) - 1;
  const annualBenchRet = Math.pow(Math.max(1 + meanBenchRet, 0.0001), 252) - 1;
  const alpha = (annualFundRet - (riskFreeRate + beta * (annualBenchRet - riskFreeRate))) * 100;

  // D. Sharpe Ratio
  const sharpe = (stdDevAnnual > 0.001) ? (annualFundRet - riskFreeRate) / (stdDevAnnual / 100) : null;

  // E. Sortino Ratio
  const dailyRiskFree = Math.pow(1 + riskFreeRate, 1 / 252) - 1;
  const downsideSquaredSum = fundReturns.reduce((sum, r) => {
    const diff = r - dailyRiskFree;
    return sum + (diff < 0 ? Math.pow(diff, 2) : 0);
  }, 0);
  const downsideDevDaily = Math.sqrt(downsideSquaredSum / n);
  const downsideDevAnnual = downsideDevDaily * Math.sqrt(252);
  const sortino = (downsideDevAnnual > 0.0001) ? (annualFundRet - riskFreeRate) / downsideDevAnnual : null;

  return {
    stdDev: isFinite(stdDevAnnual) ? parseFloat(stdDevAnnual.toFixed(2)) : null,
    beta: isFinite(beta) ? parseFloat(beta.toFixed(2)) : null,
    alpha: isFinite(alpha) ? parseFloat(alpha.toFixed(2)) : null,
    sharpe: isFinite(sharpe) ? parseFloat(sharpe.toFixed(2)) : null,
    sortino: isFinite(sortino) ? parseFloat(sortino.toFixed(2)) : null,
  };
}

// ─── Capture Ratios ───────────────────────────────────────────────────────────

/**
 * Calculate upside and downside capture ratios over N years.
 */
export function calculateCaptureRatios(schemeCode, benchmarkCode, years) {
  const cutoff = cutoffDateISO(years);

  const rows = getDB().prepare(`
    SELECT f.date, f.nav as fund_nav, b.nav as bench_nav
    FROM nav_history f
    INNER JOIN nav_history b ON f.date = b.date
    WHERE f.scheme_code = ? AND b.scheme_code = ? AND f.date >= ?
    ORDER BY f.date ASC
  `).all(schemeCode, benchmarkCode, cutoff);

  if (rows.length < (years * 252 * 0.8)) return null;

  // Calculate daily returns
  const fundReturns = [];
  const benchReturns = [];
  for (let i = 1; i < rows.length; i++) {
    const fPrev = rows[i - 1].fund_nav;
    const bPrev = rows[i - 1].bench_nav;
    if (fPrev > 0 && bPrev > 0) {
      fundReturns.push((rows[i].fund_nav - fPrev) / fPrev);
      benchReturns.push((rows[i].bench_nav - bPrev) / bPrev);
    }
  }

  if (fundReturns.length < (years * 252 * 0.75)) return null;

  // Compound returns separately for upside and downside periods
  let upFund = 1, upBench = 1;
  let downFund = 1, downBench = 1;

  for (let i = 0; i < fundReturns.length; i++) {
    if (benchReturns[i] >= 0) {
      upBench *= (1 + benchReturns[i]);
      upFund *= (1 + fundReturns[i]);
    } else {
      downBench *= (1 + benchReturns[i]);
      downFund *= (1 + fundReturns[i]);
    }
  }

  const upsideCapture = Math.abs(upBench - 1) > 1e-7 ? ((upFund - 1) / (upBench - 1)) * 100 : null;
  const downsideCapture = Math.abs(downBench - 1) > 1e-7 ? ((downFund - 1) / (downBench - 1)) * 100 : null;

  return {
    upside: (upsideCapture !== null && isFinite(upsideCapture)) ? Math.round(upsideCapture) : null,
    downside: (downsideCapture !== null && isFinite(downsideCapture)) ? Math.round(downsideCapture) : null,
  };
}

// ─── Compute & Store ──────────────────────────────────────────────────────────

/**
 * Compute all metrics for a fund and store in the fund_metrics table.
 * Automatically chooses the asset-appropriate benchmark (Gilt for Debt, Nifty 50 for Equity).
 */
export function computeAndStoreMetrics(schemeCode) {
  const fundRow = getDB().prepare('SELECT category FROM funds WHERE scheme_code = ?').get(schemeCode);
  const category = fundRow ? fundRow.category : '';
  const benchmarkCode = getBenchmarkCodeForCategory(category);

  const ret6m = calculateAbsoluteReturn(schemeCode, 6);
  const cagr1y = calculateCAGR(schemeCode, 1);
  const cagr3y = calculateCAGR(schemeCode, 3);
  const cagr5y = calculateCAGR(schemeCode, 5);

  const risk3y = calculateRiskMetrics(schemeCode, benchmarkCode, 3);
  const risk5y = calculateRiskMetrics(schemeCode, benchmarkCode, 5);
  const capture3y = calculateCaptureRatios(schemeCode, benchmarkCode, 3);
  const capture5y = calculateCaptureRatios(schemeCode, benchmarkCode, 5);

  getDB().prepare(`
    INSERT OR REPLACE INTO fund_metrics
      (scheme_code, return_6m, cagr_1y, cagr_3y, cagr_5y, 
       alpha, beta, sharpe, sortino, std_dev, 
       alpha_5y, beta_5y, sharpe_5y, sortino_5y, std_dev_5y, 
       upside_capture, downside_capture, upside_capture_3y, downside_capture_3y, computed_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
  `).run(
    schemeCode,
    ret6m,
    cagr1y,
    cagr3y,
    cagr5y,
    risk3y?.alpha ?? null,
    risk3y?.beta ?? null,
    risk3y?.sharpe ?? null,
    risk3y?.sortino ?? null,
    risk3y?.stdDev ?? null,
    risk5y?.alpha ?? null,
    risk5y?.beta ?? null,
    risk5y?.sharpe ?? null,
    risk5y?.sortino ?? null,
    risk5y?.stdDev ?? null,
    capture5y?.upside ?? null,
    capture5y?.downside ?? null,
    capture3y?.upside ?? null,
    capture3y?.downside ?? null
  );

  console.log(`[MetricsEngine] Computed metrics for scheme ${schemeCode} (benchmark: ${benchmarkCode})`);
}

/**
 * Recompute metrics for all tracked funds.
 */
export function recomputeAllMetrics() {
  const funds = getDB().prepare('SELECT scheme_code FROM funds').all();
  const total = funds.length;

  console.log(`[MetricsEngine] Recomputing metrics for ${total} funds`);

  let processed = 0;
  for (const { scheme_code } of funds) {
    try {
      computeAndStoreMetrics(scheme_code);
      processed++;
    } catch (err) {
      console.error(`[MetricsEngine] Failed for scheme ${scheme_code}: ${err.message}`);
    }

    if (processed % 100 === 0) {
      console.log(`[MetricsEngine] Progress: ${processed}/${total}`);
    }
  }

  console.log(`[MetricsEngine] Completed: ${processed}/${total} funds processed`);
  return processed;
}
