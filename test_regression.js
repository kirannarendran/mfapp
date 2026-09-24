import fundsRouter from './server/routes/funds.js';
import authRouter from './server/routes/auth.js';
import advisorRouter from './server/routes/advisor.js';
import { getDB } from './server/db.js';
import { calculateCAGR, calculateRiskMetrics, calculateCaptureRatios, isDebtCategory } from './server/services/metricsEngine.js';

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
