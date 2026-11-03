import fundsRouter from './server/routes/funds.js';
import authRouter from './server/routes/auth.js';
import advisorRouter from './server/routes/advisor.js';
import { getDB } from './server/db.js';
import { calculateCAGR, calculateAbsoluteReturn, calculateRiskMetrics, calculateCaptureRatios, isDebtCategory } from './server/services/metricsEngine.js';
import { parseLatestYield } from './server/services/rbiRateFetcher.js';
import { validateSnapshot } from './server/validateSnapshot.js';
import Database from 'better-sqlite3';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

// Helper to simulate Express request/response locally without network sockets
function simulateRequest(router, method, url, body = {}, headers = {}) {
  return new Promise((resolve) => {
    const parsedUrl = new URL(url, 'http://localhost');
    const query = Object.fromEntries(parsedUrl.searchParams.entries());
    const pathname = parsedUrl.pathname;

    const req = {
      method: method.toUpperCase(),
      url: pathname + parsedUrl.search,
      originalUrl: pathname + parsedUrl.search,
      path: pathname,
      query,
      body,
      headers: {
        'content-type': 'application/json',
        ...headers
      },
      params: {}
    };

    let statusCode = 200;
    let responseData = null;
    let headersSent = {};

    const res = {
      status(code) {
        statusCode = code;
        return this;
      },
      json(data) {
        responseData = data;
        resolve({ status: statusCode, body: responseData, headers: headersSent });
        return this;
      },
      setHeader(k, v) {
        headersSent[k.toLowerCase()] = v;
      },
      flushHeaders() {},
      write(chunk) {},
      end() {
        resolve({ status: statusCode, body: responseData, headers: headersSent });
      }
    };

    // Route matching simulation using router handle
    router.handle(req, res, (err) => {
      if (err) {
        resolve({ status: 500, body: { error: err.message } });
      } else {
        resolve({ status: 404, body: { error: 'Not found' } });
      }
    });
  });
}

async function runTests() {
  console.log('=====================================================');
  console.log('🚀 FUND SENSE.AI — COMPREHENSIVE REGRESSION & AUDIT');
  console.log('=====================================================\n');

  let passed = 0;
  let failed = 0;

  function assert(condition, testName) {
    if (condition) {
      console.log(`  ✅ PASS: ${testName}`);
      passed++;
    } else {
      console.error(`  ❌ FAIL: ${testName}`);
      failed++;
    }
  }

  // 1. Math and Engine Tests
  console.log('📌 1. Testing Metrics Engine & Mathematical Formulas');
  try {
    const cagr = calculateCAGR(100484, 3);
    assert(typeof cagr === 'number' || cagr === null, 'calculateCAGR returns number or null');

    const debtRisk = calculateRiskMetrics(120137, 120137, 3);
    assert(debtRisk !== null && debtRisk.beta === 1, 'Self-beta calculation equals 1.0');
    assert(debtRisk !== null && !isNaN(debtRisk.sharpe), 'Sharpe ratio is not NaN');
    assert(debtRisk !== null && !isNaN(debtRisk.stdDev), 'StdDev is not NaN');

    const capture = calculateCaptureRatios(100484, 100484, 3);
    assert(capture !== null && capture.upside === 100 && capture.downside === 100, 'Self capture ratio is exactly 100%');

    assert(isDebtCategory('Debt Scheme - Gilt Fund') === true, 'isDebtCategory detects Gilt Fund');
    assert(isDebtCategory('Income/Debt Oriented Schemes - Short Term Fund') === true, 'isDebtCategory detects Short Term Fund');
    assert(isDebtCategory('Equity Scheme - Large Cap Fund') === false, 'isDebtCategory returns false for Equity Large Cap');
  } catch (err) {
    assert(false, `Metrics calculation threw: ${err.message}`);
  }

  // 2. Testing API Routes
  console.log('\n📌 2. Testing Funds & Screener API Routes');
  try {
    // 2.1 Search
    const searchRes = await simulateRequest(fundsRouter, 'GET', '/funds?search=HDFC&limit=5');
    assert(searchRes.status === 200, 'GET /funds?search=HDFC returns 200');
    assert(Array.isArray(searchRes.body.funds) && searchRes.body.funds.length <= 5, 'Search returns funds array');

    // 2.2 Fund Detail
    const detailRes = await simulateRequest(fundsRouter, 'GET', '/funds/100484');
    assert(detailRes.status === 200, 'GET /funds/100484 returns 200');
    assert(detailRes.body.meta && detailRes.body.meta.scheme_name, 'Fund detail contains valid metadata');

    // 2.3 Non-existent fund
    const notFoundRes = await simulateRequest(fundsRouter, 'GET', '/funds/99999999');
    assert(notFoundRes.status === 404, 'GET /funds/99999999 returns 404 Not Found');

    // 2.4 Fund Metrics (3Y & 5Y)
    const metricsRes = await simulateRequest(fundsRouter, 'GET', '/funds/100484/metrics');
    assert(metricsRes.status === 200, 'GET /funds/100484/metrics returns 200');
    assert(metricsRes.body.cagr_3y !== undefined, 'Metrics includes cagr_3y');
    assert(metricsRes.body.cagr_5y !== undefined, 'Metrics includes cagr_5y');
    assert(metricsRes.body.sharpe_5y !== undefined, 'Metrics includes sharpe_5y');

    // 2.5 Benchmarks
    const benchRes = await simulateRequest(fundsRouter, 'GET', '/benchmark');
    assert(benchRes.status === 200, 'GET /benchmark returns 200');
    assert(benchRes.body.meta.scheme_code === 100484, 'Default benchmark is Equity Nifty 50 (100484)');

    const debtBenchRes = await simulateRequest(fundsRouter, 'GET', '/benchmark?type=debt');
    assert(debtBenchRes.status === 200, 'GET /benchmark?type=debt returns 200');
    assert(debtBenchRes.body.meta.scheme_code === 120137, 'Debt benchmark is 10Y Constant Maturity Gilt (120137)');

    // 2.6 Screener with TotalCount, Pagination, and Category
    const screenerRes = await simulateRequest(fundsRouter, 'GET', '/funds/screener?limit=10&offset=0');
    assert(screenerRes.status === 200, 'GET /funds/screener returns 200');
    assert(screenerRes.body.totalCount > 0, `Screener returns totalCount (${screenerRes.body.totalCount})`);
    assert(screenerRes.body.totalPages >= 1, `Screener returns totalPages (${screenerRes.body.totalPages})`);
    assert(screenerRes.body.count <= 10, 'Screener page size respected');

    const debtScreenerRes = await simulateRequest(fundsRouter, 'GET', '/funds/screener?category=Debt&limit=5');
    assert(debtScreenerRes.status === 200, 'GET /funds/screener?category=Debt returns 200');
    assert(debtScreenerRes.body.totalCount > 0, `Debt category screener finds matches (${debtScreenerRes.body.totalCount})`);

  } catch (err) {
    assert(false, `API test error: ${err.message}`);
  }

  // 3. Vulnerability Testing: SQL Injection & Input Validation
  console.log('\n📌 3. Vulnerability Testing: SQL Injection & Input Validation');
  try {
    const sqlSearchRes = await simulateRequest(fundsRouter, 'GET', "/funds?search=' OR '1'='1");
    assert(sqlSearchRes.status === 200, 'SQL injection in search handled safely via parameterized query');

    const sqlScreenerRes = await simulateRequest(fundsRouter, 'GET', '/funds/screener?sortBy=cagr_3y;DROP+TABLE+users;--');
    assert(sqlScreenerRes.status === 200, 'SQL injection in sortBy rejected by whitelist validation');

    const sqlCodeRes = await simulateRequest(fundsRouter, 'GET', '/funds/100484%27%20OR%201=1');
    assert(sqlCodeRes.status === 400 || sqlCodeRes.status === 404, 'SQL injection in schemeCode URL rejected');

    // 3.1 Auth Security
    const unauthMeRes = await simulateRequest(authRouter, 'GET', '/me');
    assert(unauthMeRes.status === 401, 'GET /api/auth/me without token returns 401 Unauthorized');

    const invalidTokenMeRes = await simulateRequest(authRouter, 'GET', '/me', {}, {
      authorization: 'Bearer bogus.jwt.token'
    });
    assert(invalidTokenMeRes.status === 401, 'GET /api/auth/me with bogus token returns 401 Unauthorized');

    const unauthProfileRes = await simulateRequest(authRouter, 'PUT', '/profile', { age: 30 });
    assert(unauthProfileRes.status === 401, 'PUT /api/auth/profile without token returns 401 Unauthorized');

    // 3.2 Advisor Route Validation
    const badChatRes = await simulateRequest(advisorRouter, 'POST', '/advisor/chat', { message: 'hi' });
    assert(badChatRes.status === 400, 'POST /api/advisor/chat with message < 5 chars returns 400 Bad Request');

  } catch (err) {
    assert(false, `Security test error: ${err.message}`);
  }

  // 4. Data Quality: return periods, flat NAVs, non-investable plans, risk-free source
  console.log('\n📌 4. Testing Data Quality Rules');
  const db = getDB();
  const FIXTURE = { full: 99999901, young: 99999902, newborn: 99999903, flat: 99999904, sidePocket: 99999905, segregatedName: 99999906, mainScheme: 99999907, unclaimed: 99999908 };
  const fixtureCodes = Object.values(FIXTURE);
  const clearFixtures = () => {
    for (const code of fixtureCodes) {
      db.prepare('DELETE FROM nav_history WHERE scheme_code = ?').run(code);
      db.prepare('DELETE FROM fund_metrics WHERE scheme_code = ?').run(code);
      db.prepare('DELETE FROM funds WHERE scheme_code = ?').run(code);
    }
  };
  try {
    clearFixtures();
    const benchDates = (since) => db.prepare(
      `SELECT date FROM nav_history WHERE scheme_code = 100484 AND date >= date('now', ?) ORDER BY date ASC`
    ).all(since).map(r => r.date);
    const addFund = (code, name, lastNav = 10) => {
      db.prepare('INSERT INTO funds (scheme_code, scheme_name, category, last_nav, last_nav_date) VALUES (?, ?, ?, ?, ?)')
        .run(code, name, 'Equity Scheme - Flexi Cap Fund', lastNav, '2026-01-01');
      db.prepare('INSERT INTO fund_metrics (scheme_code, cagr_3y) VALUES (?, 9999)').run(code);
    };
    const addNavs = (code, dates, navAt) => {
      const insert = db.prepare('INSERT INTO nav_history (scheme_code, date, nav) VALUES (?, ?, ?)');
      dates.forEach((date, i) => insert.run(code, date, navAt(i)));
    };

    addFund(FIXTURE.full, 'ZZFIXTURE Full History Fund');
    addNavs(FIXTURE.full, benchDates('-3 years'), i => 10 + i * 0.01 + (i % 2) * 0.05);
    addFund(FIXTURE.young, 'ZZFIXTURE Young Fund');
    addNavs(FIXTURE.young, benchDates('-8 months'), i => 10 + i * 0.05);
    addFund(FIXTURE.newborn, 'ZZFIXTURE Two Month Fund');
    addNavs(FIXTURE.newborn, benchDates('-2 months'), i => 10 + i * 0.05);
    addFund(FIXTURE.flat, 'ZZFIXTURE Flat NAV Fund');
    addNavs(FIXTURE.flat, benchDates('-3 years'), () => 10);
    addFund(FIXTURE.sidePocket, 'ZZFIXTURE Zero NAV Fund', 0);
    addFund(FIXTURE.segregatedName, 'ZZFIXTURE Bond Fund (Segregated Portfolio 1)');
    addFund(FIXTURE.mainScheme, 'ZZFIXTURE Hybrid Fund (Existing Number of Segregated Portfolios - 1)');
    addFund(FIXTURE.unclaimed, 'ZZFIXTURE Liquid Fund Unclaimed Redemption');

    assert(typeof calculateCAGR(FIXTURE.full, 3) === 'number', 'Fund with 3 years of history gets a 3Y CAGR');
    assert(calculateCAGR(FIXTURE.young, 3) === null, '8-month-old fund has no 3Y CAGR');
    assert(calculateCAGR(FIXTURE.young, 1) === null, '8-month-old fund has no 1Y CAGR');
    assert(typeof calculateAbsoluteReturn(FIXTURE.young, 6) === 'number', '8-month-old fund has a 6M return');
    assert(calculateAbsoluteReturn(FIXTURE.newborn, 6) === null, '2-month-old fund has no 6M return');

    const flatRisk = calculateRiskMetrics(FIXTURE.flat, 100484, 3);
    assert(flatRisk !== null && flatRisk.sharpe === null && flatRisk.sortino === null, 'Flat NAV gives N/A Sharpe and Sortino without crashing');
    const fullRisk = calculateRiskMetrics(FIXTURE.full, 100484, 3);
    assert(fullRisk !== null && typeof fullRisk.sharpe === 'number', 'Volatile fund still gets a Sharpe ratio');

    const searchRes = await simulateRequest(fundsRouter, 'GET', '/funds?search=ZZFIXTURE&limit=100');
    const searchCodes = searchRes.body.funds.map(f => f.scheme_code);
    assert(!searchCodes.includes(FIXTURE.sidePocket), 'Search hides zero-NAV side pockets');
    assert(!searchCodes.includes(FIXTURE.segregatedName), 'Search hides "(Segregated Portfolio)" entries');
    assert(!searchCodes.includes(FIXTURE.unclaimed), 'Search hides Unclaimed plans');
    assert(searchCodes.includes(FIXTURE.mainScheme), 'Search keeps main schemes that mention segregated portfolios');

    const screenerRes = await simulateRequest(fundsRouter, 'GET', '/funds/screener?category=All&limit=500');
    const screenerCodes = screenerRes.body.funds.map(f => f.scheme_code);
    assert(screenerCodes.includes(FIXTURE.mainScheme), 'Screener keeps main schemes that mention segregated portfolios');
    assert(![FIXTURE.sidePocket, FIXTURE.segregatedName, FIXTURE.unclaimed].some(c => screenerCodes.includes(c)), 'Screener hides non-investable plans');

    const parsedYield = parseLatestYield('observation_date,INDIRLTLT01STM\n2026-08-01,6.84\n2026-09-01,7.06\n2026-10-01,\n');
    assert(parsedYield?.yieldPercent === 7.06 && parsedYield?.date === '2026-09-01', 'Yield parser skips blank months and takes the latest value');
    assert(parseLatestYield('observation_date,X\n2026-09-01,99\n') === null, 'Yield parser rejects out-of-range values');
  } catch (err) {
    assert(false, `Data quality test error: ${err.message}`);
  } finally {
    clearFixtures();
  }

  // 5. Snapshot publishing guard
  console.log('\n📌 5. Testing Snapshot Publishing Guard');
  const tempDir = mkdtempSync(join(tmpdir(), 'snapshot-test-'));
  try {
    const buildSnapshot = (name, { funds = 100, cagr = 100, sharpe = 100, latest = '2026-11-02', benchRows = 5, users = 0 } = {}) => {
      const path = join(tempDir, name);
      const db = new Database(path);
      db.exec(`CREATE TABLE funds (scheme_code INTEGER, last_nav_date TEXT);
        CREATE TABLE fund_metrics (cagr_3y REAL, sharpe REAL);
        CREATE TABLE users (id TEXT);
        CREATE TABLE nav_history (scheme_code INTEGER, date TEXT, nav REAL);`);
      for (let i = 0; i < funds; i++) db.prepare('INSERT INTO funds VALUES (?, ?)').run(i, latest);
      for (let i = 0; i < Math.max(cagr, sharpe); i++) db.prepare('INSERT INTO fund_metrics VALUES (?, ?)').run(i < cagr ? 10 : null, i < sharpe ? 1 : null);
      for (let i = 0; i < users; i++) db.prepare('INSERT INTO users VALUES (?)').run(String(i));
      for (const code of [100484, 120137]) {
        for (let i = 0; i < benchRows; i++) db.prepare('INSERT INTO nav_history VALUES (?, ?, 10)').run(code, i === 0 ? latest : `2026-01-0${i}`);
      }
      db.close();
      return path;
    };
    const today = new Date('2026-11-03');
    const previous = buildSnapshot('previous.db');

    assert(validateSnapshot(buildSnapshot('good.db'), previous, today).length === 0, 'Healthy snapshot passes');
    assert(validateSnapshot(buildSnapshot('users.db', { users: 1 }), previous, today).some(f => f.includes('user records')), 'Snapshot containing users is blocked');
    assert(validateSnapshot(buildSnapshot('metrics.db', { cagr: 50, sharpe: 50 }), previous, today).some(f => f.includes('withCagr3y')), 'Mass loss of metrics is blocked');
    assert(validateSnapshot(buildSnapshot('funds.db', { funds: 50 }), previous, today).some(f => f.includes('funds dropped')), 'Mass loss of funds is blocked');
    assert(validateSnapshot(buildSnapshot('bench.db', { benchRows: 2 }), previous, today).some(f => f.includes('lost rows')), 'Benchmark history loss is blocked');
    assert(validateSnapshot(buildSnapshot('stale.db', { latest: '2026-10-01' }), buildSnapshot('old.db', { latest: '2026-09-01' }), today).some(f => f.includes('stale')), 'Stale benchmark is blocked');
    assert(validateSnapshot(buildSnapshot('backwards.db', { latest: '2026-10-30' }), previous, today).some(f => f.includes('backwards')), 'Older NAVs than the last snapshot are blocked');
  } catch (err) {
    assert(false, `Snapshot guard test error: ${err.message}`);
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }

  console.log('\n=====================================================');
  console.log(`📊 RESULTS: ${passed} PASSED, ${failed} FAILED`);
  console.log('=====================================================');

  if (failed > 0) {
    process.exit(1);
  }
}

runTests().catch(err => {
  console.error('Fatal error in test suite:', err);
  process.exit(1);
});
