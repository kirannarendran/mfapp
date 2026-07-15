import test from 'node:test';
import assert from 'node:assert/strict';

const API_URL = 'http://localhost:3001/api/funds/screener';

test('ML Demotion API Validations', async (t) => {

    await t.test('Standard API omits the experimental metric', async () => {
        const res = await fetch(`${API_URL}?category=All`);
        const data = await res.json();
        assert.ok(data.funds.length > 0, "Should return funds");
        for (const fund of data.funds) {
            const keys = Object.keys(fund);
            assert.ok(!keys.some(k => k.startsWith('ml_')), "Standard projection contains no key beginning with ml_");
            assert.strictEqual(fund.raw_model_score, undefined, "raw_model_score should never be exposed");
        }
    });

    await t.test('Old minMlProb parameter explicitly rejected', async () => {
        const res = await fetch(`${API_URL}?minMlProb=50`);
        assert.strictEqual(res.status, 400, "Should reject minMlProb with 400 Bad Request");
        const data = await res.json();
        assert.match(data.error, /deprecated/, "Error message should explain deprecation");
    });

    await t.test('minMlRankingScore rejected without experimental mode', async () => {
        const res = await fetch(`${API_URL}?minMlRankingScore=50`);
        assert.strictEqual(res.status, 400, "Should reject with 400 Bad Request");
        const data = await res.json();
        assert.match(data.error, /requires includeExperimental=true/, "Error message should explain requirement");
    });

    await t.test('Invalid minMlRankingScore values rejected', async () => {
        const res1 = await fetch(`${API_URL}?includeExperimental=true&minMlRankingScore=-1`);
        assert.strictEqual(res1.status, 400, "Should reject -1");
        
        const res2 = await fetch(`${API_URL}?includeExperimental=true&minMlRankingScore=101`);
        assert.strictEqual(res2.status, 400, "Should reject 101");
        
        const res3 = await fetch(`${API_URL}?includeExperimental=true&minMlRankingScore=abc`);
        assert.strictEqual(res3.status, 400, "Should reject abc");
    });

    await t.test('Experimental mode enables the metric (with missing metadata)', async () => {
        const response = await fetch(`${API_URL}?includeExperimental=true&minMlRankingScore=0&category=All`);
        assert.strictEqual(response.status, 200);
        const data = await response.json();
        
        if (data.funds.length > 0) {
            const fund = data.funds[0];
            assert.ok('ml_ranking_score' in fund, 'ml_ranking_score key should be present in experimental mode');
            assert.strictEqual(fund.ml_ranking_score, null, 'ml_ranking_score should be null when metadata is incomplete');
            assert.strictEqual(fund.ml_training_cutoff_date, null, 'ml_training_cutoff_date should be null');
            assert.strictEqual(fund.ml_status, 'experimental', 'ml_status should be experimental');
            assert.strictEqual(fund.ml_score_status, 'model_metadata_incomplete', 'ml_score_status should indicate incomplete metadata');
        }
    });



});
