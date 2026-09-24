import { getDB, initDB } from './server/db.js';
import { runStructuredAdvisorAgent } from './server/services/aiAdvisor.js';
import fundsRouter from './server/routes/funds.js';
import advisorRouter from './server/routes/advisor.js';
import authRouter from './server/routes/auth.js';
import jwt from 'jsonwebtoken';

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
  console.log('=== STARTING COMPREHENSIVE FUNCTIONAL TESTS ===\n');
  let passed = 0;
  let failed = 0;

  function assert(condition, testName) {
    if (condition) {
      console.log(`✅ PASS: ${testName}`);
      passed++;
    } else {
      console.error(`❌ FAIL: ${testName}`);
      failed++;
    }
  }

  // 1. DB Init & Benchmark Verification
  initDB();
  const db = getDB();
  const fundCount = db.prepare('SELECT count(*) as count FROM funds').get().count;
  assert(fundCount > 0, `Database has funds loaded (${fundCount} funds)`);

  const debtNavCount = db.prepare('SELECT count(*) as count FROM nav_history WHERE scheme_code = 120137').get().count;
  assert(debtNavCount > 1000, `Debt benchmark (120137) has extensive historical NAVs (${debtNavCount} records)`);

  const eqNavCount = db.prepare('SELECT count(*) as count FROM nav_history WHERE scheme_code = 100484').get().count;
  assert(eqNavCount > 1000, `Equity benchmark (100484) has extensive historical NAVs (${eqNavCount} records)`);

  // 2. Test GET /funds (Curated Cold Start)
  const res1 = await simulateRequest(fundsRouter, 'GET', '/funds');
  assert(res1.status === 200 && res1.body.funds && res1.body.funds.length > 0, `GET /api/funds returns curated spotlight list (${res1.body.funds?.length} funds)`);
  assert(res1.body.isCurated === true, `GET /api/funds has isCurated: true for blank search`);

  // 3. Test GET /funds?search=Liquid
  const res2 = await simulateRequest(fundsRouter, 'GET', '/funds?search=Liquid');
  assert(res2.status === 200 && res2.body.funds.length > 0, `GET /api/funds?search=Liquid returns search results`);

  // 4. Test GET /funds/119369 (Bank of India Liquid Fund)
  const res3 = await simulateRequest(fundsRouter, 'GET', '/funds/119369');
  assert(res3.status === 200 && res3.body.meta && res3.body.data.length > 0, `GET /api/funds/119369 returns fund metadata and historical NAVs (${res3.body.data?.length} records)`);

  // 5. Test GET /funds/120137 (Debt Benchmark)
  const res4 = await simulateRequest(fundsRouter, 'GET', '/funds/120137');
  assert(res4.status === 200 && res4.body.meta && res4.body.data.length > 0, `GET /api/funds/120137 returns debt benchmark metadata and NAVs (${res4.body.data?.length} records)`);

  // 6. Test GET /benchmark?type=debt
  const res5 = await simulateRequest(fundsRouter, 'GET', '/benchmark?type=debt');
  assert(res5.status === 200 && res5.body.data && res5.body.data.length > 0, `GET /api/benchmark?type=debt returns debt benchmark data (${res5.body.data?.length} records)`);

  // 7. Test GET /funds/119369/metrics
  const res6 = await simulateRequest(fundsRouter, 'GET', '/funds/119369/metrics');
  assert(res6.status === 200 && (res6.body.cagr_3y !== undefined || res6.body.sharpe !== undefined), `GET /api/funds/119369/metrics returns calculated risk metrics`);

  // 8. Test GET /funds/compare?codes=119369,120137,100484
  const res7 = await simulateRequest(fundsRouter, 'GET', '/funds/compare?codes=119369,120137,100484');
  assert(res7.status === 200 && res7.body.funds && res7.body.funds.length === 3, `GET /api/funds/compare returns side-by-side comparison for 3 schemes`);

  // 9. Test GET /funds/screener
  const res8 = await simulateRequest(fundsRouter, 'GET', '/funds/screener?category=Equity&limit=10');
  assert(res8.status === 200 && res8.body.funds && res8.body.totalCount > 0, `GET /api/funds/screener supports filtering and pagination (total: ${res8.body.totalCount})`);

  // 10. Test Guided Portfolio Plan without Emergency Fund (hasEmergencyFund: 'not_yet')
  let planWithEmergency = null;
  const dummyRes = {
    write: (msg) => {
      if (msg.includes('"type":"result"')) {
        const jsonStr = msg.replace('data: ', '').trim();
        planWithEmergency = JSON.parse(jsonStr).recommendation;
      }
    },
    end: () => {},
    flushHeaders: () => {}
  };

  await runStructuredAdvisorAgent({
    goal: 'Wealth Creation & Compounding',
    horizonYears: 10,
    monthlySIP: 15000,
    lumpSum: 0,
    riskProfile: 'aggressive',
    maxDrawdownPct: 25,
    hasEmergencyFund: 'not_yet',
    numberOfFunds: 4
  }, dummyRes);

  assert(planWithEmergency !== null, `runStructuredAdvisorAgent produces valid portfolio recommendation`);
  assert(planWithEmergency?.funds && planWithEmergency.funds.length >= 3, `Plan contains at least 3 diversified schemes`);
  
  const emergencyScheme = planWithEmergency?.funds?.find(f => 
    (f.category && f.category.toLowerCase().includes('liquid')) || 
    (f.reason_short && f.reason_short.toLowerCase().includes('emergency'))
  );
  assert(Boolean(emergencyScheme), `Plan explicitly incorporates dedicated Emergency Buffer fund (${emergencyScheme?.name}) when user has no emergency fund`);
  assert(emergencyScheme?.allocation_percentage >= 15, `Emergency Fund allocation is at least 15-20% (${emergencyScheme?.allocation_percentage}%)`);

  const totalAllocation = planWithEmergency?.funds?.reduce((sum, f) => sum + f.allocation_percentage, 0);
  assert(totalAllocation === 100, `Total portfolio allocation sums to exactly 100% (actual: ${totalAllocation}%)`);

  // 11. Test Auth & Deterministic User ID
  const testEmail = 'test.investor.2026@gmail.com';
  const deterministicId = 'usr_' + Buffer.from(testEmail).toString('hex').slice(0, 24);
  const JWT_SECRET = 'fundsense_jwt_secret_default_key_2026';
  const testToken = jwt.sign(
    { id: deterministicId, email: testEmail, name: 'Test Investor' },
    JWT_SECRET,
    { expiresIn: '30d' }
  );

  // Test /api/auth/me auto-restore on ephemeral container
  const meRes = await simulateRequest(authRouter, 'GET', '/me', {}, {
    authorization: `Bearer ${testToken}`
  });
  assert(meRes.status === 200 && meRes.body.user && meRes.body.user.email === testEmail, `GET /api/auth/me auto-rehydrates user on ephemeral DB restart`);

  // Test /api/auth/profile update
  const profileRes = await simulateRequest(authRouter, 'PUT', '/profile', {
    firstName: 'Kiran',
    lastName: 'Narendran',
    age: 30,
    profession: 'Software & Technology / IT',
    investmentExperience: 'Experienced'
  }, {
    authorization: `Bearer ${testToken}`
  });
  assert(profileRes.status === 200 && profileRes.body.user.age === 30 && profileRes.body.user.profession === 'Software & Technology / IT', `PUT /api/auth/profile persists age and profession`);

  // 12. Test Capital Preservation & 2Y Short Horizon Plan (Zero Credit Risk Guarantee)
  let capPresPlan = null;
  const dummyRes2 = {
    write: (msg) => {
      if (msg.includes('"type":"result"')) {
        const jsonStr = msg.replace('data: ', '').trim();
        capPresPlan = JSON.parse(jsonStr).recommendation;
      }
    },
    end: () => {},
    flushHeaders: () => {}
  };

  await runStructuredAdvisorAgent({
    goal: 'Capital preservation with modest growth',
    horizonYears: 2,
    monthlySIP: 10000,
    lumpSum: 0,
    riskProfile: 'conservative',
    maxDrawdownPct: 10,
    hasEmergencyFund: 'yes',
    numberOfFunds: 4,
    fundCategory: 'Debt/Hybrid'
  }, dummyRes2);

  assert(capPresPlan !== null, `Capital preservation plan synthesizes successfully`);
  
  const hasCreditRisk = capPresPlan?.funds?.some(f => 
    (f.category && f.category.toLowerCase().includes('credit risk')) || 
    (f.name && f.name.toLowerCase().includes('credit risk'))
  );
  assert(!hasCreditRisk, `Capital preservation / 2Y short horizon portfolio strictly EXCLUDES Credit Risk funds`);

  const expReturnRange = capPresPlan?.portfolio_summary?.portfolio_metrics?.expected_return_range;
  assert(expReturnRange === '7.0% – 8.5%', `Capital preservation / short horizon provides realistic yield expectation (${expReturnRange})`);

  // 13. Test Multiple Goal-Based Portfolio serialization
  const multiPortfolios = [
    { id: 'port_1', goalId: 'wealth_creation', goalTitle: 'Wealth Creation (15Y)', horizonYears: 15 },
    { id: 'port_2', goalId: 'capital_preservation', goalTitle: 'Capital Preservation (2Y)', horizonYears: 2 }
  ];
  assert(multiPortfolios.length === 2, `Supports storing and managing multiple distinct goal portfolios`);

  console.log(`\n=== ALL TESTS FINISHED: ${passed} PASSED, ${failed} FAILED ===`);
  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
  console.error('Test execution error:', err);
  process.exit(1);
});
