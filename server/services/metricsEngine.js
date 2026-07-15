import { getDB } from '../db.js';

/**
 * Get the current risk-free rate from the config table.
 */
function getRiskFreeRate() {
  const row = getDB().prepare('SELECT value FROM config WHERE key = ?').get('risk_free_rate');
  return row ? parseFloat(row.value) : 0.07;
}

/**
 * Get the benchmark scheme code from the config table.
 */
function getBenchmarkCode() {
  const row = getDB().prepare('SELECT value FROM config WHERE key = ?').get('benchmark_code');
  return row ? parseInt(row.value, 10) : 100484;
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

  // Actual time difference in years
  const newestMs = new Date(rows[0].date).getTime();
  const oldestMs = new Date(rows[rows.length - 1].date).getTime();
  const actualTimeDiff = (newestMs - oldestMs) / (365.25 * 24 * 3600 * 1000);

  if (actualTimeDiff < 0.5) return null;

  const cagr = (Math.pow(currentNav / oldNav, 1 / actualTimeDiff) - 1) * 100;
  return parseFloat(cagr.toFixed(2));
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

  return parseFloat((((currentNav - oldNav) / oldNav) * 100).toFixed(2));
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
    fundReturns.push((rows[i].fund_nav - rows[i - 1].fund_nav) / rows[i - 1].fund_nav);
    benchReturns.push((rows[i].bench_nav - rows[i - 1].bench_nav) / rows[i - 1].bench_nav);
  }

  const n = fundReturns.length;

  // Mean daily returns
  const meanFundRet = fundReturns.reduce((a, b) => a + b, 0) / n;
  const meanBenchRet = benchReturns.reduce((a, b) => a + b, 0) / n;

  // A. Standard Deviation (Annualized)
  const variance = fundReturns.reduce((sum, r) => sum + Math.pow(r - meanFundRet, 2), 0) / (n - 1);
  const stdDevDaily = Math.sqrt(variance);
  const stdDevAnnual = stdDevDaily * Math.sqrt(252) * 100;

  // B. Beta = Covariance(Fund, Bench) / Variance(Bench)
  let covariance = 0;
  let benchVariance = 0;
  for (let i = 0; i < n; i++) {
    covariance += (fundReturns[i] - meanFundRet) * (benchReturns[i] - meanBenchRet);
    benchVariance += Math.pow(benchReturns[i] - meanBenchRet, 2);
  }
  const beta = covariance / benchVariance;

  // C. Alpha (Jensen's Alpha)
  const annualFundRet = Math.pow(1 + meanFundRet, 252) - 1;
  const annualBenchRet = Math.pow(1 + meanBenchRet, 252) - 1;
  const alpha = (annualFundRet - (riskFreeRate + beta * (annualBenchRet - riskFreeRate))) * 100;

  // D. Sharpe Ratio
  const sharpe = (annualFundRet - riskFreeRate) / (stdDevAnnual / 100);

  // E. Sortino Ratio
  const dailyRiskFree = Math.pow(1 + riskFreeRate, 1 / 252) - 1;
  const downsideSquaredSum = fundReturns.reduce((sum, r) => {
    const diff = r - dailyRiskFree;
    return sum + (diff < 0 ? Math.pow(diff, 2) : 0);
  }, 0);
  const downsideDevDaily = Math.sqrt(downsideSquaredSum / n);
  const downsideDevAnnual = downsideDevDaily * Math.sqrt(252);
  const sortino = (annualFundRet - riskFreeRate) / downsideDevAnnual;

  return {
    stdDev: parseFloat(stdDevAnnual.toFixed(2)),
    beta: parseFloat(beta.toFixed(2)),
    alpha: parseFloat(alpha.toFixed(2)),
    sharpe: parseFloat(sharpe.toFixed(2)),
    sortino: parseFloat(sortino.toFixed(2)),
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
    fundReturns.push((rows[i].fund_nav - rows[i - 1].fund_nav) / rows[i - 1].fund_nav);
    benchReturns.push((rows[i].bench_nav - rows[i - 1].bench_nav) / rows[i - 1].bench_nav);
  }

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

  const upsideCapture = ((upFund - 1) / (upBench - 1)) * 100;
  const downsideCapture = ((downFund - 1) / (downBench - 1)) * 100;

  return {
    upside: Math.round(upsideCapture),
    downside: Math.round(downsideCapture),
  };
}

// ─── Compute & Store ──────────────────────────────────────────────────────────

/**
 * Compute all metrics for a fund and store in the fund_metrics table.
 */
export function computeAndStoreMetrics(schemeCode) {
  const benchmarkCode = getBenchmarkCode();

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
    capture5y?.upside ?? null,       // Original upside_capture is 5y
    capture5y?.downside ?? null,
    capture3y?.upside ?? null,
    capture3y?.downside ?? null
  );

  console.log(`[MetricsEngine] Computed metrics for scheme ${schemeCode}`);
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
