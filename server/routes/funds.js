import { Router } from 'express';
import { getDB } from '../db.js';
import { syncNavData, syncFundRegistry, syncBenchmarkData, syncAllTrackedFunds } from '../services/dataSync.js';
import { computeAndStoreMetrics, recomputeAllMetrics } from '../services/metricsEngine.js';
import { fetchAndUpdateRiskFreeRate } from '../services/rbiRateFetcher.js';
import { getSyncStatus, runFullSync } from '../services/scheduler.js';

const router = Router();

// Convert YYYY-MM-DD to dd-mm-yyyy for API compatibility
function toDisplayDate(isoDate) {
  const [y, m, d] = isoDate.split('-');
  return `${d}-${m}-${y}`;
}

// ── GET /sync/status — check background sync status ───────────────────────────
router.get('/sync/status', (req, res) => {
  res.json(getSyncStatus());
});

// ── POST /sync/manual — trigger a manual sync ─────────────────────────────
router.post('/sync/manual', (req, res) => {
  const status = getSyncStatus();
  if (status.isSyncing) {
    return res.status(409).json({ error: 'Sync already in progress' });
  }
  
  // Trigger in background
  runFullSync().catch(err => {
    console.error('[Funds] Manual sync failed:', err);
  });
  
  res.status(202).json({ message: 'Sync started in the background' });
});

// ── GET /funds — search endpoint ──────────────────────────────────────────────
router.get('/funds', (req, res) => {
  try {
    const { search } = req.query;
    let limit = parseInt(req.query.limit) || 50;
    limit = Math.min(limit, 100);

    if (!search || search.length < 2) {
      return res.json({ funds: [], count: 0 });
    }

    const db = getDB();
    const rows = db.prepare(`
      SELECT scheme_code, scheme_name, fund_house, category, last_nav, last_nav_date
      FROM funds
      WHERE scheme_name LIKE ?
      ORDER BY scheme_name
      LIMIT ?
    `).all(`%${search}%`, limit);

    res.json({ funds: rows, count: rows.length });
  } catch (err) {
    console.error('[Funds] Search error:', err);
    res.status(500).json({ error: 'Failed to search funds' });
  }
});

// ── GET /funds/compare — compare multiple funds ──────────────────────────────
// MUST be defined BEFORE /funds/:schemeCode to avoid route conflicts
router.get('/funds/compare', async (req, res) => {
  try {
    const { codes } = req.query;
    if (!codes) {
      return res.status(400).json({ error: 'Missing codes parameter' });
    }

    const codeList = codes.split(',').map(c => parseInt(c.trim())).filter(c => !isNaN(c));
    if (codeList.length === 0) {
      return res.status(400).json({ error: 'No valid scheme codes provided' });
    }

    const db = getDB();
    const funds = [];

    for (const code of codeList) {
      const fund = db.prepare('SELECT * FROM funds WHERE scheme_code = ?').get(code);
      if (!fund) continue;

      let metrics = db.prepare('SELECT * FROM fund_metrics WHERE scheme_code = ?').get(code);

      // Compute on demand if missing or stale (>24h)
      if (!metrics || new Date(metrics.computed_at) < new Date(Date.now() - 24 * 60 * 60 * 1000)) {
        try {
          await syncNavData(code);
          await syncBenchmarkData();
          await computeAndStoreMetrics(code);
        } catch (e) {
          console.warn(`[Funds] Failed to compute metrics on-demand for ${code}:`, e.message);
        }
        metrics = db.prepare('SELECT * FROM fund_metrics WHERE scheme_code = ?').get(code);
      }

      funds.push({
        schemeCode: fund.scheme_code,
        schemeName: fund.scheme_name,
        fundHouse: fund.fund_house,
        category: fund.category,
        returns: {
          '6M': metrics?.return_6m ?? null,
          '1Y': metrics?.cagr_1y ?? null,
          '3Y': metrics?.cagr_3y ?? null,
          '5Y': metrics?.cagr_5y ?? null,
        },
        risk: {
          '3Y': {
            alpha: metrics?.alpha ?? null,
            beta: metrics?.beta ?? null,
            sharpe: metrics?.sharpe ?? null,
            sortino: metrics?.sortino ?? null,
            stdDev: metrics?.std_dev ?? null,
          },
          '5Y': {
            alpha: metrics?.alpha_5y ?? null,
            beta: metrics?.beta_5y ?? null,
            sharpe: metrics?.sharpe_5y ?? null,
            sortino: metrics?.sortino_5y ?? null,
            stdDev: metrics?.std_dev_5y ?? null,
          }
        },
        capture: {
          '3Y': {
            upside: metrics?.upside_capture_3y ?? null,
            downside: metrics?.downside_capture_3y ?? null,
          },
          '5Y': {
            upside: metrics?.upside_capture ?? null,
            downside: metrics?.downside_capture ?? null,
          }
        },
      });
    }

    res.json({ funds });
  } catch (err) {
    console.error('[Funds] Compare error:', err);
    res.status(500).json({ error: 'Failed to compare funds' });
  }
});

// ── GET /funds/screener — filter funds by metrics ───────────────────────────
router.get('/funds/screener', async (req, res) => {
  try {
    const sortBy = req.query.sortBy;
    let limit = parseInt(req.query.limit) || 50;
    limit = Math.min(limit, 100);
    const offset = parseInt(req.query.offset) || 0;

    const minCagr3Y = req.query.minCagr3Y !== undefined ? parseFloat(req.query.minCagr3Y) : -999;
    const minCagr5Y = req.query.minCagr5Y !== undefined ? parseFloat(req.query.minCagr5Y) : -999;
    
    // 5Y Metrics
    const maxBeta5y = req.query.maxBeta5y !== undefined ? parseFloat(req.query.maxBeta5y) : 999;
    const minSharpe5y = req.query.minSharpe5y !== undefined ? parseFloat(req.query.minSharpe5y) : -999;
    const minSortino5y = req.query.minSortino5y !== undefined ? parseFloat(req.query.minSortino5y) : -999;
    const maxSd5y = req.query.maxSd5y !== undefined ? parseFloat(req.query.maxSd5y) : 999;
    const minAlpha5y = req.query.minAlpha5y !== undefined ? parseFloat(req.query.minAlpha5y) : -999;
    const minUpCap5y = req.query.minUpCap5y !== undefined ? parseFloat(req.query.minUpCap5y) : -999;
    const maxDownCap5y = req.query.maxDownCap5y !== undefined ? parseFloat(req.query.maxDownCap5y) : 999;
    
    // 3Y Metrics
    const maxBeta3y = req.query.maxBeta3y !== undefined ? parseFloat(req.query.maxBeta3y) : 999;
    const minSharpe3y = req.query.minSharpe3y !== undefined ? parseFloat(req.query.minSharpe3y) : -999;
    const minSortino3y = req.query.minSortino3y !== undefined ? parseFloat(req.query.minSortino3y) : -999;
    const maxSd3y = req.query.maxSd3y !== undefined ? parseFloat(req.query.maxSd3y) : 999;
    const minAlpha3y = req.query.minAlpha3y !== undefined ? parseFloat(req.query.minAlpha3y) : -999;
    const minUpCap3y = req.query.minUpCap3y !== undefined ? parseFloat(req.query.minUpCap3y) : -999;
    const maxDownCap3y = req.query.maxDownCap3y !== undefined ? parseFloat(req.query.maxDownCap3y) : 999;

    if (req.query.minMlProb !== undefined) {
      return res.status(400).json({ error: 'The minMlProb parameter has been deprecated and replaced by minMlRankingScore.' });
    }

    const includeExperimental = req.query.includeExperimental === 'true';

    if (req.query.minMlRankingScore !== undefined && !includeExperimental) {
      return res.status(400).json({ error: 'minMlRankingScore requires includeExperimental=true.' });
    }
    
    const minMlRankingScoreStr = req.query.minMlRankingScore;
    let minMlRankingScore = null;
    if (minMlRankingScoreStr !== undefined) {
        minMlRankingScore = parseFloat(minMlRankingScoreStr);
        if (isNaN(minMlRankingScore) || minMlRankingScore < 0 || minMlRankingScore > 100) {
            return res.status(400).json({ error: 'minMlRankingScore must be a number between 0 and 100.' });
        }
    }

    const category = req.query.category; // Optional

    const db = getDB();
    
    let query = `
      SELECT f.scheme_code, f.scheme_name, f.fund_house, f.category, 
             m.cagr_3y, m.cagr_5y, 
             m.beta_5y, m.sharpe_5y, m.sortino_5y, m.std_dev_5y, m.alpha_5y, 
             m.upside_capture as upside_capture_5y, m.downside_capture as downside_capture_5y,
             m.beta as beta_3y, m.sharpe as sharpe_3y, m.sortino as sortino_3y, m.std_dev as std_dev_3y, m.alpha as alpha_3y,
             m.upside_capture_3y, m.downside_capture_3y
    `;

    if (includeExperimental) {
      query += `,
             CASE WHEN ml.ml_status = 'experimental' 
                       AND ml.ml_score_status = 'current'
                       AND ml.ml_training_cutoff_date IS NOT NULL
                       AND ml.ml_category_peer_count >= 10
                       AND ml.ml_ranking_score BETWEEN 0 AND 100
                       AND date(ml.ml_expires_at) IS NOT NULL
                       AND date('now') <= date(ml.ml_expires_at)
                  THEN ml.ml_ranking_score ELSE NULL END as ml_ranking_score,
             ml.ml_score_as_of_date,
             ml.ml_expires_at,
             ml.ml_model_version,
             ml.ml_training_cutoff_date,
             ml.ml_status,
             ml.ml_score_status,
             ml.ml_category_peer_count
      `;
    }

    query += `
      FROM funds f
      INNER JOIN fund_metrics m ON f.scheme_code = m.scheme_code
    `;

    if (includeExperimental) {
      query += ` LEFT JOIN ml_predictions ml ON f.scheme_code = ml.scheme_code `;
    }

    query += `
      WHERE (m.cagr_3y >= ? OR (? <= -10 AND m.cagr_3y IS NULL))
        AND (m.cagr_5y >= ? OR (? <= -10 AND m.cagr_5y IS NULL))
        AND (m.beta_5y <= ? OR (? >= 2 AND m.beta_5y IS NULL))
        AND (m.sharpe_5y >= ? OR (? <= -1 AND m.sharpe_5y IS NULL))
        AND (m.sortino_5y >= ? OR (? <= -999 AND m.sortino_5y IS NULL))
        AND (m.std_dev_5y <= ? OR (? >= 999 AND m.std_dev_5y IS NULL))
        AND (m.alpha_5y >= ? OR (? <= -999 AND m.alpha_5y IS NULL))
        AND (m.upside_capture >= ? OR (? <= -999 AND m.upside_capture IS NULL))
        AND (m.downside_capture <= ? OR (? >= 999 AND m.downside_capture IS NULL))
        AND (m.beta <= ? OR (? >= 2 AND m.beta IS NULL))
        AND (m.sharpe >= ? OR (? <= -1 AND m.sharpe IS NULL))
        AND (m.sortino >= ? OR (? <= -999 AND m.sortino IS NULL))
        AND (m.std_dev <= ? OR (? >= 999 AND m.std_dev IS NULL))
        AND (m.alpha >= ? OR (? <= -999 AND m.alpha IS NULL))
        AND (m.upside_capture_3y >= ? OR (? <= -999 AND m.upside_capture_3y IS NULL))
        AND (m.downside_capture_3y <= ? OR (? >= 999 AND m.downside_capture_3y IS NULL))
    `;

    if (includeExperimental) {
      query += `
        AND (
          ? IS NULL OR (
             ml.ml_status = 'experimental' 
             AND ml.ml_score_status = 'current'
             AND ml.ml_training_cutoff_date IS NOT NULL
             AND ml.ml_category_peer_count >= 10
             AND ml.ml_ranking_score BETWEEN 0 AND 100
             AND date(ml.ml_expires_at) IS NOT NULL
             AND date('now') <= date(ml.ml_expires_at)
             AND ml.ml_ranking_score >= ?
          )
        )
      `;
    }

    query += ` AND (f.category LIKE '%Equity%' OR f.category LIKE '%ELSS%') `;
    const params = [
      minCagr3Y, minCagr3Y, 
      minCagr5Y, minCagr5Y, 
      maxBeta5y, maxBeta5y, 
      minSharpe5y, minSharpe5y,
      minSortino5y, minSortino5y,
      maxSd5y, maxSd5y,
      minAlpha5y, minAlpha5y,
      minUpCap5y, minUpCap5y,
      maxDownCap5y, maxDownCap5y,
      maxBeta3y, maxBeta3y, 
      minSharpe3y, minSharpe3y,
      minSortino3y, minSortino3y,
      maxSd3y, maxSd3y,
      minAlpha3y, minAlpha3y,
      minUpCap3y, minUpCap3y,
      maxDownCap3y, maxDownCap3y
    ];
    
    if (includeExperimental) {
      params.push(minMlRankingScore, minMlRankingScore);
    }

    if (category === 'Others') {
      query += ` AND f.category NOT LIKE '%Large Cap%'
                 AND f.category NOT LIKE '%Mid Cap%'
                 AND f.category NOT LIKE '%Small Cap%'
                 AND f.category NOT LIKE '%Flexi Cap%'
                 AND f.category NOT LIKE '%Multi Cap%'
                 AND f.category NOT LIKE '%ELSS%'`;
    } else if (category && category !== 'All') {
      query += ` AND f.category LIKE ?`;
      params.push(`%${category}%`);
    }

    if (sortBy && ['cagr_3y', 'cagr_5y', 'alpha_5y', 'beta_5y', 'sharpe_5y', 'sortino_5y', 'ml_ranking_score'].includes(sortBy)) {
      query += ` ORDER BY ${sortBy} DESC NULLS LAST`;
    } else {
      query += ` ORDER BY m.cagr_3y DESC NULLS LAST`;
    }

    query += ` LIMIT ? OFFSET ?`;
    params.push(limit, offset);
    
    console.log("[DB] SQL:", query);
    console.log("[DB] Params:", params);

    const funds = db.prepare(query).all(...params);

    res.json({ funds, count: funds.length });
  } catch (err) {
    console.error('[Funds] Screener error:', err);
    res.status(500).json({ error: 'Failed to screen funds' });
  }
});

// ── GET /funds/:schemeCode — fund detail with NAV history ────────────────────
router.get('/funds/:schemeCode', async (req, res) => {
  try {
    const schemeCode = parseInt(req.params.schemeCode);
    if (isNaN(schemeCode)) {
      return res.status(400).json({ error: 'Invalid scheme code' });
    }

    const db = getDB();
    let fund = db.prepare('SELECT * FROM funds WHERE scheme_code = ?').get(schemeCode);
    let navRows = db.prepare(
      'SELECT date, nav FROM nav_history WHERE scheme_code = ? ORDER BY date DESC'
    ).all(schemeCode);

    // If fund NOT found in DB OR no NAV history, try to sync on-demand
    if (!fund || navRows.length === 0) {
      await syncNavData(schemeCode);
      fund = db.prepare('SELECT * FROM funds WHERE scheme_code = ?').get(schemeCode);
      navRows = db.prepare(
        'SELECT date, nav FROM nav_history WHERE scheme_code = ? ORDER BY date DESC'
      ).all(schemeCode);
    }

    if (!fund) {
      return res.status(404).json({ error: 'Fund not found' });
    }

    res.json({
      meta: {
        scheme_code: fund.scheme_code,
        scheme_name: fund.scheme_name,
        fund_house: fund.fund_house,
        scheme_category: fund.category,
        scheme_type: fund.type,
      },
      data: navRows.map(r => ({
        date: toDisplayDate(r.date),
        nav: String(r.nav),
      })),
    });
  } catch (err) {
    console.error('[Funds] Detail error:', err);
    res.status(500).json({ error: 'Failed to fetch fund details' });
  }
});

// ── GET /funds/:schemeCode/metrics — fund risk/return metrics ────────────────
router.get('/funds/:schemeCode/metrics', async (req, res) => {
  try {
    const schemeCode = parseInt(req.params.schemeCode);
    if (isNaN(schemeCode)) {
      return res.status(400).json({ error: 'Invalid scheme code' });
    }

    const db = getDB();
    let metrics = db.prepare('SELECT * FROM fund_metrics WHERE scheme_code = ?').get(schemeCode);

    // Compute if missing or stale (>24h)
    if (!metrics || new Date(metrics.computed_at) < new Date(Date.now() - 24 * 60 * 60 * 1000)) {
      try {
        await syncNavData(schemeCode);
        await syncBenchmarkData();
        await computeAndStoreMetrics(schemeCode);
      } catch (e) {
        console.warn(`[Funds] Failed to compute metrics on-demand for ${schemeCode}:`, e.message);
      }
      metrics = db.prepare('SELECT * FROM fund_metrics WHERE scheme_code = ?').get(schemeCode);
    }

    if (!metrics) {
      return res.status(404).json({ error: 'Metrics not available for this fund' });
    }

    res.json(metrics);
  } catch (err) {
    console.error('[Funds] Metrics error:', err);
    res.status(500).json({ error: 'Failed to fetch fund metrics' });
  }
});

// ── GET /benchmark — benchmark fund data ─────────────────────────────────────
router.get('/benchmark', async (req, res) => {
  try {
    const db = getDB();
    const config = db.prepare("SELECT value FROM config WHERE key = 'benchmark_code'").get();
    if (!config) {
      return res.status(404).json({ error: 'Benchmark not configured' });
    }

    const schemeCode = parseInt(config.value);
    let fund = db.prepare('SELECT * FROM funds WHERE scheme_code = ?').get(schemeCode);
    let navRows = db.prepare(
      'SELECT date, nav FROM nav_history WHERE scheme_code = ? ORDER BY date DESC'
    ).all(schemeCode);

    if (!fund || navRows.length === 0) {
      await syncNavData(schemeCode);
      fund = db.prepare('SELECT * FROM funds WHERE scheme_code = ?').get(schemeCode);
      navRows = db.prepare(
        'SELECT date, nav FROM nav_history WHERE scheme_code = ? ORDER BY date DESC'
      ).all(schemeCode);
    }

    if (!fund) {
      return res.status(404).json({ error: 'Benchmark fund not found' });
    }

    res.json({
      meta: {
        scheme_code: fund.scheme_code,
        scheme_name: fund.scheme_name,
        fund_house: fund.fund_house,
        scheme_category: fund.category,
        scheme_type: fund.type,
      },
      data: navRows.map(r => ({
        date: toDisplayDate(r.date),
        nav: String(r.nav),
      })),
    });
  } catch (err) {
    console.error('[Funds] Benchmark error:', err);
    res.status(500).json({ error: 'Failed to fetch benchmark data' });
  }
});

// ── GET /config — app configuration ──────────────────────────────────────────
router.get('/config', (req, res) => {
  try {
    const db = getDB();
    const rows = db.prepare('SELECT * FROM config').all();
    const config = {};
    for (const row of rows) {
      config[row.key] = row.value;
    }
    res.json(config);
  } catch (err) {
    console.error('[Funds] Config error:', err);
    res.status(500).json({ error: 'Failed to fetch configuration' });
  }
});

// ── POST /admin/sync — manual full sync ──────────────────────────────────────
router.post('/admin/sync', async (req, res) => {
  try {
    const results = {};

    console.log('[Admin] Starting manual sync...');

    try {
      await fetchAndUpdateRiskFreeRate();
      results.riskFreeRate = 'success';
    } catch (err) {
      console.error('[Admin] Risk-free rate sync failed:', err);
      results.riskFreeRate = `failed: ${err.message}`;
    }

    try {
      results.fundRegistry = await syncFundRegistry();
    } catch (err) {
      console.error('[Admin] Fund registry sync failed:', err);
      results.fundRegistry = `failed: ${err.message}`;
    }

    try {
      await syncBenchmarkData();
      results.benchmark = 'success';
    } catch (err) {
      console.error('[Admin] Benchmark sync failed:', err);
      results.benchmark = `failed: ${err.message}`;
    }

    try {
      results.trackedFunds = await syncAllTrackedFunds();
    } catch (err) {
      console.error('[Admin] Tracked funds sync failed:', err);
      results.trackedFunds = `failed: ${err.message}`;
    }

    try {
      await recomputeAllMetrics();
      results.metrics = 'success';
    } catch (err) {
      console.error('[Admin] Metrics recompute failed:', err);
      results.metrics = `failed: ${err.message}`;
    }

    console.log('[Admin] Manual sync complete');
    res.json({ status: 'complete', results });
  } catch (err) {
    console.error('[Admin] Sync error:', err);
    res.status(500).json({ error: 'Sync failed' });
  }
});

export default router;
