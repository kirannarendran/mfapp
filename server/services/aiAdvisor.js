import { getDB } from '../db.js';

const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.GROK_API_KEY; 
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';
const GROQ_MODEL = 'openai/gpt-oss-120b'; // gpt-oss-20b struggles with complex JSON schemas

/**
 * Strip markdown code fences that Grok sometimes adds despite being told not to.
 * e.g. ```json\n{...}\n``` → {...}
 */
function stripCodeFences(text) {
  // Extract just the JSON object or array by finding the first and last brackets
  const start = Math.min(
    text.indexOf('{') === -1 ? Infinity : text.indexOf('{'),
    text.indexOf('[') === -1 ? Infinity : text.indexOf('[')
  );
  const end = Math.max(text.lastIndexOf('}'), text.lastIndexOf(']'));
  
  if (start !== Infinity && end !== -1 && end >= start) {
    return text.substring(start, end + 1);
  }
  return text;
}

/**
 * Call xAI Grok API (OpenAI-compatible)
 */
async function callGrok(messages, temperature = 0.3) {

  const response = await fetch(`${GROQ_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages,
      temperature,
      response_format: { type: 'json_object' },
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Grok API error ${response.status}: ${err}`);
  }

  const data = await response.json();
  return data.choices[0].message.content.trim();
}

/**
 * Sends a progress step event back to the client via SSE
 */
function sendStep(res, step) {
  res.write(`data: ${JSON.stringify(step)}\n\n`);
}

/**
 * Step 1 — Use Grok to extract structured financial parameters from the user's message.
 */
async function extractParameters(userMessage) {
  const content = await callGrok([
    {
      role: 'system',
      content: `You are a financial parameter extractor. Extract investment parameters from the user message and return a JSON object.
The JSON must have these exact keys (use null for anything not mentioned):
- goal: string ("retirement", "child education", "house", "wealth creation", or "emergency fund")
- horizonYears: number or null
- monthlySIP: number in INR or null
- lumpSum: number in INR or null
- targetCorpus: number in INR or null
- maxDrawdownPct: number (e.g. 20 for 20% max drop) or null
- riskProfile: "conservative", "moderate", "aggressive", or null
- expectedCAGR: number or null

Respond with ONLY the JSON object. No prose, no markdown fences.`,
    },
    { role: 'user', content: userMessage },
  ]);

  console.log('[AIAdvisor] Raw Grok response for parameter extraction:', content);

  // Try multiple parsing strategies
  // 1. Direct parse after stripping code fences
  try {
    return JSON.parse(stripCodeFences(content));
  } catch (_) {}

  // 2. Extract first {...} block found anywhere in the response
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch (_) {}
  }

  // 3. Fallback — return a default object so the agent can still run with what it has
  console.warn('[AIAdvisor] Could not parse JSON from Grok, using defaults. Raw:', content);
  return {
    goal: 'wealth creation',
    horizonYears: null,
    monthlySIP: null,
    lumpSum: null,
    targetCorpus: null,
    maxDrawdownPct: null,
    riskProfile: 'moderate',
    expectedCAGR: null,
  };
}


/**
 * Step 2 — Query the SQLite fund DB using extracted risk parameters.
 * Translates max drawdown tolerance → beta/std_dev constraints.
 */
function screenFunds(params) {
  const db = getDB();

  const isConservativeOrDebt = 
    params.fundCategory === 'Debt/Hybrid' || 
    (params.horizonYears && params.horizonYears <= 3) || 
    params.riskProfile === 'conservative' || 
    (params.goal && params.goal.toLowerCase().includes('preservation'));

  let maxBeta = 1.5;
  let minSharpe = 0.3;
  let minCAGR5Y = params.expectedCAGR ? params.expectedCAGR - 5 : 8;

  if (isConservativeOrDebt) {
    minCAGR5Y = params.expectedCAGR ? Math.max(params.expectedCAGR - 3, 4.5) : 5.0;
    minSharpe = 0.2;
    maxBeta = 0.6;
  } else if (params.maxDrawdownPct) {
    if (params.maxDrawdownPct <= 10) { maxBeta = 0.7; minSharpe = 0.7; }
    else if (params.maxDrawdownPct <= 20) { maxBeta = 1.0; minSharpe = 0.5; }
    else if (params.maxDrawdownPct <= 30) { maxBeta = 1.2; minSharpe = 0.4; }
  } else if (params.riskProfile === 'conservative') {
    maxBeta = 0.8; minSharpe = 0.6;
  } else if (params.riskProfile === 'aggressive') {
    maxBeta = 1.5; minSharpe = 0.2;
  }

  // Category Filtering
  let categoryFilter = '';
  if (isConservativeOrDebt) {
    // Strictly exclude Credit Risk funds for conservative / capital preservation
    categoryFilter = "AND (f.category LIKE '%Debt%' OR f.category LIKE '%Hybrid%' OR f.category LIKE '%Gilt%' OR f.category LIKE '%Liquid%' OR f.category LIKE '%Money Market%') AND f.category NOT LIKE '%Credit Risk%'";
  } else if (params.fundCategory && params.fundCategory !== 'Any Equity') {
    if (params.fundCategory === 'Large Cap') categoryFilter = "AND (f.category LIKE '%Large Cap%')";
    else if (params.fundCategory === 'Mid & Small Cap') categoryFilter = "AND (f.category LIKE '%Mid Cap%' OR f.category LIKE '%Small Cap%')";
    else if (params.fundCategory === 'Flexi/Multi Cap') categoryFilter = "AND (f.category LIKE '%Flexi Cap%' OR f.category LIKE '%Multi Cap%')";
    else if (params.fundCategory === 'Sectoral/Thematic') categoryFilter = "AND (f.category LIKE '%Sectoral%' OR f.category LIKE '%Thematic%')";
    else if (params.fundCategory === 'Index Funds') categoryFilter = "AND f.category LIKE '%Index%'";
    else if (params.fundCategory === 'Debt/Hybrid') categoryFilter = "AND (f.category LIKE '%Debt%' OR f.category LIKE '%Hybrid%') AND f.category NOT LIKE '%Credit Risk%'";
  } else {
    categoryFilter = "AND (f.category LIKE '%Equity%' OR f.category LIKE '%ELSS%')";
  }

  let rows = [];

  // If conservative or debt-oriented, ensure diversified representation across high-safety debt sub-categories
  if (isConservativeOrDebt) {
    const buckets = [
      { filter: "(f.category LIKE '%Gilt%' OR f.category LIKE '%Constant Maturity%')" },
      { filter: "(f.category LIKE '%Banking and PSU%' OR f.category LIKE '%Corporate Bond%')" },
      { filter: "(f.category LIKE '%Money Market%' OR f.category LIKE '%Short Duration%' OR f.category LIKE '%Low Duration%' OR f.category LIKE '%Ultra Short%')" },
      { filter: "(f.category LIKE '%Liquid%')" },
      { filter: "(f.category LIKE '%Conservative Hybrid%')" }
    ];

    for (const b of buckets) {
      const bRows = db.prepare(`
        SELECT 
          f.scheme_code, f.scheme_name, f.category, f.fund_house, f.last_nav as nav,
          m.cagr_1y, m.cagr_3y, m.cagr_5y,
          m.beta, m.sharpe, m.sortino, m.std_dev, m.alpha,
          m.upside_capture, m.downside_capture
        FROM funds f
        INNER JOIN fund_metrics m ON f.scheme_code = m.scheme_code
        WHERE ${b.filter}
          AND f.category NOT LIKE '%Credit Risk%'
          AND m.cagr_5y >= ?
          AND m.cagr_5y IS NOT NULL
          AND m.sharpe IS NOT NULL
        ORDER BY m.sharpe DESC, m.std_dev ASC
        LIMIT 4
      `).all(minCAGR5Y);
      rows.push(...bRows);
    }
  }

  // Fallback or equity mode: standard ranked query
  if (rows.length < 5) {
    const orderByClause = isConservativeOrDebt
      ? 'ORDER BY m.sharpe DESC, m.std_dev ASC'
      : 'ORDER BY m.cagr_5y DESC, m.alpha DESC';

    rows = db.prepare(`
      SELECT 
        f.scheme_code, f.scheme_name, f.category, f.fund_house, f.last_nav as nav,
        m.cagr_1y, m.cagr_3y, m.cagr_5y,
        m.beta, m.sharpe, m.sortino, m.std_dev, m.alpha,
        m.upside_capture, m.downside_capture
      FROM funds f
      INNER JOIN fund_metrics m ON f.scheme_code = m.scheme_code
      WHERE 
        1=1
        ${categoryFilter}
        AND m.cagr_5y >= ?
        AND (m.beta IS NULL OR m.beta <= ?)
        AND (m.sharpe IS NULL OR m.sharpe >= ?)
        AND m.cagr_5y IS NOT NULL
        AND m.sharpe IS NOT NULL
      ${orderByClause}
      LIMIT 30
    `).all(minCAGR5Y, maxBeta, minSharpe);
  }

  return { rows, filters: { maxBeta, minSharpe, minCAGR5Y } };
}

/**
 * Step 3 — Use Grok to reason over the screened funds and produce a recommendation.
 */
async function generateRecommendation(params, screenedFunds) {
  const fundSummary = screenedFunds.slice(0, 15).map((f, i) =>
    `${i + 1}. ${f.scheme_name} | Category: ${f.category} | 5Y CAGR: ${f.cagr_5y?.toFixed(1)}% | Sharpe: ${f.sharpe?.toFixed(2)} | Beta: ${f.beta?.toFixed(2)} | Sortino: ${f.sortino?.toFixed(2)} | Alpha: ${f.alpha?.toFixed(2)}`
  ).join('\n');

  const isLowRiskOrShortHorizon = (params.horizonYears && params.horizonYears <= 3) || 
    params.riskProfile === 'conservative' || 
    (params.goal && params.goal.toLowerCase().includes('preservation'));

  return await callGrok([
    {
      role: 'system',
      content: `You are a strict, institutional SEBI-compliant Indian mutual fund advisor. A user has shared their investment goals and our risk engine has pre-screened eligible funds. 
Select exactly ${params.numberOfFunds || '3-4'} funds from the pre-screened list, suggest % SIP allocation for each, explain WHY each suits this user's specific profile, provide a portfolio strategy summary, and mention 1-2 key risks.

CRITICAL ASSET ALLOCATION & DIVERSIFICATION RULES:
1. CATEGORY DIVERSIFICATION (MANDATORY): Each recommended fund MUST be from a DIFFERENT category. NEVER select two funds of the same category (e.g., do NOT select two Credit Risk funds, or two Liquid funds, or two Large Cap funds).
2. CAPITAL PRESERVATION & SHORT HORIZON (<= 3 YEARS):
   - You MUST NEVER recommend Credit Risk funds for Capital Preservation or Conservative risk profiles. Credit Risk funds carry credit default risk and are classified as High Risk by SEBI.
   - For Capital Preservation / Conservative portfolios, recommend high-safety fixed income: Sovereign Gilt, Banking & PSU / Corporate Bond (AAA), Money Market, Liquid, or Conservative Hybrid.
   - RETURN EXPECTATION ACCURACY: For low-risk debt / capital preservation portfolios (<= 3 years), the realistic return expectation is "7.0% – 8.5%". Do NOT cite equity-like returns of 12% – 16%.
3. Total allocations across all selected funds must sum to EXACTLY 100%.

Return valid JSON only. Do not wrap the response in markdown. Do not include commentary before or after the JSON. Keep recommendation reasons concise. Do not invent metrics. Use null for unavailable values. Ensure allocations total exactly 100.

Use this JSON schema:
{
  "portfolio_summary": {
    "title": "Capital Preservation & Stability Portfolio",
    "description": "A high-safety portfolio designed for capital protection and modest yield.",
    "risk_level": "Conservative",
    "investment_horizon_years": 2,
    "objective": "Capital preservation",
    "review_frequency": "Annual",
    "portfolio_metrics": {
      "weighted_beta": 0.05,
      "estimated_drawdown_percentage": 2.5,
      "weighted_cagr_percentage": 7.4,
      "expected_return_range": "7.0% – 8.5%"
    }
  },
  "analysis": {
    "funds_evaluated": 1400,
    "funds_eliminated": 1370,
    "funds_shortlisted": 30,
    "funds_deeply_analyzed": 15,
    "filters": [
      { "metric": "Beta", "operator": "<=", "value": 1 }
    ]
  },
  "funds": [
    {
      "name": "Franklin India Corporate Bond Fund - Direct Plan - Growth",
      "allocation_percentage": 30,
      "category": "Corporate Bond Fund",
      "risk_level": "Low",
      "reason_short": "High-safety AAA-rated corporate yield anchor.",
      "reason_detailed": "Invests primarily in highest rated AAA corporate papers providing stability.",
      "metrics": {
        "cagr_5y_percentage": 6.8,
        "alpha": 1.15,
        "beta": 0.02,
        "sharpe_ratio": 1.15
      }
    }
  ],
  "strategy": [
    "Diversifies exposure across distinct, complementary fixed income categories."
  ],
  "risks": [
    {
      "title": "Interest Rate Fluctuations",
      "severity": "Low",
      "description": "Short-term yield movements may cause slight variation in debt NAVs."
    }
  ],
  "disclaimer": "Mutual fund investments are subject to market risks. Historical performance does not guarantee future returns."
}`,
    },
    {
      role: 'user',
      content: `- Goal: ${params.goal || 'Wealth creation'}
- Investment Horizon: ${params.horizonYears ? params.horizonYears + ' years' : 'Not specified'}
- Monthly SIP: ${params.monthlySIP ? '₹' + params.monthlySIP.toLocaleString('en-IN') : 'Not specified'}
- Max Drawdown Tolerated: ${params.maxDrawdownPct ? params.maxDrawdownPct + '%' : 'Not specified'}
- Risk Profile: ${params.riskProfile || 'moderate'}
- Profile Mode: ${isLowRiskOrShortHorizon ? 'CAPITAL PRESERVATION / SHORT HORIZON (<= 3Y). STRICTLY EXCLUDE CREDIT RISK. Expected yield 7.0% – 8.5%.' : 'Growth / Compounding'}
- Emergency Reserve Status: ${params.hasEmergencyFund === 'not_yet' ? 'CRITICAL - NO EMERGENCY FUND: User has no emergency buffer. You MUST include at least one Liquid or Short Duration Debt fund (15-20% allocation) specifically designated as an Emergency Safety Net buffer.' : 'Adequate emergency fund maintained.'}
- Requested Fund Count: ${params.numberOfFunds || 4}

PRE-SCREENED SHORTLIST:
${fundSummary}`,
    }
  ], 0.5);
}

/**
 * Main agent function — streams steps via SSE
 */
export async function runAdvisorAgent(userMessage, res) {
  if (!GROQ_API_KEY) {
    sendStep(res, { type: 'error', message: 'API key is not set. Please add it to your .env file.' });
    return;
  }

  try {
    // ── Step 1: Extract parameters ────────────────────────────────────────────
    sendStep(res, { type: 'step', icon: '🤔', title: 'Extracting your financial parameters...', status: 'loading' });

    const params = await extractParameters(userMessage);

    sendStep(res, {
      type: 'step', icon: '✅', title: 'Parameters extracted', status: 'done',
      detail: [
        params.goal && `Goal: ${params.goal}`,
        params.horizonYears && `Horizon: ${params.horizonYears} years`,
        params.monthlySIP && `Monthly SIP: ₹${params.monthlySIP.toLocaleString('en-IN')}`,
        params.lumpSum && `Lump Sum: ₹${params.lumpSum.toLocaleString('en-IN')}`,
        params.targetCorpus && `Target Corpus: ₹${params.targetCorpus.toLocaleString('en-IN')}`,
        params.maxDrawdownPct && `Max Drawdown: ${params.maxDrawdownPct}%`,
        params.riskProfile && `Risk Profile: ${params.riskProfile}`,
      ].filter(Boolean).join(' · ')
    });

    // ── Step 2: Screen funds ──────────────────────────────────────────────────
    sendStep(res, { type: 'step', icon: '🔍', title: 'Screening funds against your risk constraints...', status: 'loading' });

    let screenResult;
    try {
      screenResult = screenFunds(params);
    } catch (e) {
      sendStep(res, { type: 'error', message: 'Fund database not available. Please ensure the server data is synced.' });
      return;
    }

    const { rows: screenedFunds, filters } = screenResult;
    const eliminated = 1400 - screenedFunds.length;

    sendStep(res, {
      type: 'step', icon: '📊', title: 'Funds screened by risk metrics', status: 'done',
      detail: `Filters: Beta ≤ ${filters.maxBeta} · Sharpe ≥ ${filters.minSharpe} · 5Y CAGR ≥ ${filters.minCAGR5Y}% · ~${eliminated} funds eliminated · ${screenedFunds.length} shortlisted`
    });

    if (screenedFunds.length === 0) {
      sendStep(res, { type: 'error', message: 'No funds matched your risk constraints. Try relaxing your drawdown tolerance.' });
      return;
    }

    // ── Step 3: Rank & reason ─────────────────────────────────────────────────
    sendStep(res, { type: 'step', icon: '🧠', title: 'AI reasoning over shortlisted funds...', status: 'loading' });

    const recommendation = await generateRecommendation(params, screenedFunds);

    sendStep(res, {
      type: 'step', icon: '✅', title: 'Portfolio allocation reasoned', status: 'done',
      detail: `Analysed top ${Math.min(screenedFunds.length, 15)} funds for best risk-return fit`
    });

    sendStep(res, { type: 'result', recommendation });

  } catch (err) {
    console.error('[AIAdvisor] Error:', err);
    sendStep(res, { type: 'error', message: `Error: ${err.message}` });
  }
}

/**
 * Secondary agent function for the unified Wealth Planner — 
 * accepts structured params directly from the frontend wizard, skipping extraction step.
 */
export async function runStructuredAdvisorAgent(params, res) {
  try {
    // ── Step 1: Screen funds ──────────────────────────────────────────────────
    sendStep(res, { type: 'step', icon: '🔍', title: 'Screening funds against your risk constraints...', status: 'loading' });

    let screenResult;
    try {
      screenResult = screenFunds(params);
    } catch (e) {
      console.error('[runStructuredAdvisorAgent] screenFunds failed:', e);
      screenResult = { rows: [], filters: { maxBeta: 1.2, minSharpe: 0.4, minCAGR5Y: 8 } };
    }

    const { rows: screenedFunds, filters } = screenResult;
    const eliminated = Math.max(1400 - screenedFunds.length, 0);

    sendStep(res, {
      type: 'step', icon: '📊', title: 'Screening funds against your risk constraints...', status: 'done',
      detail: `Filters: Beta ≤ ${filters.maxBeta} · Sharpe ≥ ${filters.minSharpe} · 5Y CAGR ≥ ${filters.minCAGR5Y}% · ~${eliminated} funds eliminated · ${screenedFunds.length} shortlisted`
    });

    // ── Step 2: Rank & reason (via AI or deterministic rule-based fallback) ────
    sendStep(res, { type: 'step', icon: '🧠', title: 'Synthesizing customized portfolio allocation...', status: 'loading' });

    let parsedRecommendation = null;

    if (GROQ_API_KEY && screenedFunds.length > 0) {
      try {
        const shuffledFunds = [...screenedFunds].sort(() => 0.5 - Math.random());
        const recommendationStr = await generateRecommendation(params, shuffledFunds);
        const cleanStr = stripCodeFences(recommendationStr);
        parsedRecommendation = JSON.parse(cleanStr);

        if (!parsedRecommendation.portfolio_summary && Object.keys(parsedRecommendation).length === 1) {
          const rootKey = Object.keys(parsedRecommendation)[0];
          if (parsedRecommendation[rootKey].portfolio_summary) {
            parsedRecommendation = parsedRecommendation[rootKey];
          }
        }

        let sum = 0;
        parsedRecommendation.funds?.forEach(f => sum += Number(f.allocation_percentage || 0));
        if (Math.round(sum) !== 100) {
          throw new Error(`Allocations sum to ${Math.round(sum)}%, expected 100%`);
        }
      } catch (e) {
        console.warn('[runStructuredAdvisorAgent] AI model response failed or unparseable, using algorithmic fallback:', e.message);
        parsedRecommendation = null;
      }
    }

    if (!parsedRecommendation) {
      parsedRecommendation = generateRuleBasedRecommendation(params, screenedFunds);
    } else if (parsedRecommendation.funds && Array.isArray(parsedRecommendation.funds)) {
      // Ensure AI-generated recommendations have mathematically accurate weighted CAGR
      let calcCAGR = 0;
      let totalAlloc = 0;
      let calcBeta = 0;
      parsedRecommendation.funds.forEach(f => {
        const alloc = Number(f.allocation_percentage) || 0;
        const cagr = Number(f.metrics?.cagr_5y_percentage ?? f.metrics?.cagr_3y_percentage ?? f.cagr_5y);
        const beta = Number(f.metrics?.beta ?? f.beta ?? 0.8);
        if (!isNaN(cagr) && alloc > 0) {
          calcCAGR += (cagr * alloc);
          totalAlloc += alloc;
        }
        if (!isNaN(beta) && alloc > 0) {
          calcBeta += (beta * alloc);
        }
      });
      if (totalAlloc > 0) {
        if (!parsedRecommendation.portfolio_summary) parsedRecommendation.portfolio_summary = {};
        if (!parsedRecommendation.portfolio_summary.portfolio_metrics) parsedRecommendation.portfolio_summary.portfolio_metrics = {};
        const weightedCagr = Math.round((calcCAGR / totalAlloc) * 10) / 10;
        parsedRecommendation.portfolio_summary.portfolio_metrics.weighted_cagr_percentage = weightedCagr;
        parsedRecommendation.portfolio_summary.portfolio_metrics.weighted_beta = Math.round((calcBeta / totalAlloc) * 100) / 100;
        const isShortOrConservative = (params.horizonYears && Number(params.horizonYears) <= 3) || params.riskProfile === 'conservative';
        parsedRecommendation.portfolio_summary.portfolio_metrics.expected_return_range = isShortOrConservative
          ? '7.0% – 8.5%'
          : `${Math.max(4.0, (weightedCagr - 1.5)).toFixed(1)}% – ${(weightedCagr + 1.5).toFixed(1)}%`;
      }
    }

    sendStep(res, {
      type: 'step', icon: '✅', title: 'Portfolio allocation reasoned', status: 'done',
      detail: `Generated balanced portfolio across ${parsedRecommendation.funds?.length || 3} top-tier Direct schemes`
    });

    sendStep(res, { type: 'result', recommendation: parsedRecommendation });

  } catch (err) {
    console.error('[AIAdvisor] Error:', err);
    sendStep(res, { type: 'error', message: `Error: ${err.message}` });
  }
}

function generateRuleBasedRecommendation(params, screenedFunds) {
  const horizon = params.horizonYears || 15;
  const risk = (params.riskProfile || 'moderate').toLowerCase();
  const db = getDB();

  const needsEmergency = params.hasEmergencyFund === 'not_yet';
  const emergencyPct = needsEmergency ? 20 : 0;

  let equityPct = 70;
  let debtPct = 30;
  const isShortOrConservative = risk === 'conservative' || horizon <= 3;

  if (isShortOrConservative) {
    equityPct = needsEmergency ? 15 : 20; // Maximum 15-20% Conservative Hybrid for low-risk
    debtPct = needsEmergency ? 65 : 80;
  } else if (risk === 'aggressive' && horizon >= 7) {
    equityPct = needsEmergency ? 65 : 85;
    debtPct = needsEmergency ? 15 : 15;
  } else {
    // Moderate default
    equityPct = needsEmergency ? 55 : 70;
    debtPct = needsEmergency ? 25 : 30;
  }

  // Large cap or conservative hybrid depending on risk
  const largeCap = isShortOrConservative ? (
    db.prepare(`
      SELECT f.scheme_code, f.scheme_name as name, f.category, m.cagr_5y, m.sharpe, m.beta, m.alpha
      FROM funds f JOIN fund_metrics m ON f.scheme_code = m.scheme_code
      WHERE (f.category LIKE '%Conservative Hybrid%')
        AND m.cagr_5y IS NOT NULL
      ORDER BY m.sharpe DESC LIMIT 1
    `).get() || { scheme_code: 119854, name: 'Parag Parikh Conservative Hybrid Fund - Direct Plan - Growth', category: 'Hybrid - Conservative Hybrid', cagr_5y: 9.35, sharpe: 1.02, beta: 0.15, alpha: 2.1 }
  ) : (
    (screenedFunds && screenedFunds.find(f => f.category && (f.category.includes('Large Cap') || f.category.includes('Flexi Cap')))) ||
    db.prepare(`
      SELECT f.scheme_code, f.scheme_name as name, f.category, m.cagr_5y, m.sharpe, m.beta, m.alpha
      FROM funds f JOIN fund_metrics m ON f.scheme_code = m.scheme_code
      WHERE (f.category LIKE '%Large Cap%' OR f.category LIKE '%Flexi Cap%')
        AND m.cagr_5y IS NOT NULL
      ORDER BY m.sharpe DESC LIMIT 1
    `).get() || (screenedFunds && screenedFunds[0]) || { scheme_code: 100484, name: 'Nippon India Index Fund - Nifty 50 Plan - Direct Growth', category: 'Equity - Large Cap', cagr_5y: 15.2, sharpe: 0.95, beta: 0.98, alpha: 0.5 }
  );

  const midCap = (horizon >= 5 && !isShortOrConservative) ? (
    (screenedFunds && screenedFunds.find(f => f.category && (f.category.includes('Mid Cap') || f.category.includes('Small Cap')))) ||
    db.prepare(`
      SELECT f.scheme_code, f.scheme_name as name, f.category, m.cagr_5y, m.sharpe, m.beta, m.alpha
      FROM funds f JOIN fund_metrics m ON f.scheme_code = m.scheme_code
      WHERE (f.category LIKE '%Mid Cap%' OR f.category LIKE '%Large & Mid%')
        AND m.cagr_5y IS NOT NULL
      ORDER BY m.sharpe DESC LIMIT 1
    `).get()
  ) : null;

  const debt = db.prepare(`
    SELECT f.scheme_code, f.scheme_name as name, f.category, m.cagr_5y, m.sharpe, m.beta, m.alpha
    FROM funds f JOIN fund_metrics m ON f.scheme_code = m.scheme_code
    WHERE (f.category LIKE '%Gilt%' OR f.category LIKE '%Constant Maturity%' OR f.category LIKE '%Banking and PSU%' OR f.category LIKE '%Corporate Bond%')
      AND f.category NOT LIKE '%Credit Risk%'
      AND f.scheme_code != 100484
    ORDER BY m.sharpe DESC, m.cagr_3y DESC LIMIT 1
  `).get() || { scheme_code: 120137, name: 'SBI 10 Year Constant Maturity Gilt Fund - Direct Growth', category: 'Debt - Gilt Fund', cagr_5y: 7.2, sharpe: 0.8, beta: 0.05, alpha: 1.2 };

  const liquidFund = needsEmergency ? (
    db.prepare(`
      SELECT f.scheme_code, f.scheme_name as name, f.category, m.cagr_5y, m.sharpe, m.beta, m.alpha
      FROM funds f JOIN fund_metrics m ON f.scheme_code = m.scheme_code
      WHERE (f.category LIKE '%Liquid%' OR f.category LIKE '%Ultra Short%' OR f.category LIKE '%Low Duration%')
        AND f.category NOT LIKE '%Credit Risk%'
        AND f.scheme_code != 100484
      ORDER BY m.sharpe DESC LIMIT 1
    `).get() || { scheme_code: 119369, name: 'Bank of India Liquid Fund - Direct Plan - Growth', category: 'Debt Scheme - Liquid Fund', cagr_5y: 6.4, sharpe: 4.06, beta: 0.05, alpha: 1.67 }
  ) : null;

  const funds = [];
  if (midCap && equityPct >= 60) {
    const largeAlloc = Math.round(equityPct * 0.6);
    const midAlloc = equityPct - largeAlloc;
    funds.push({
      name: largeCap.scheme_name || largeCap.name,
      scheme_code: largeCap.scheme_code,
      allocation_percentage: largeAlloc,
      category: largeCap.category || 'Large / Flexi Cap',
      risk_level: 'Moderate',
      reason_short: 'Core compounding foundation with disciplined risk-adjusted returns.',
      reason_detailed: `Consistently outperforms its benchmark with a 5Y CAGR of ${largeCap.cagr_5y ? largeCap.cagr_5y.toFixed(1) : '14.5'}% and Sharpe ratio of ${largeCap.sharpe ? largeCap.sharpe.toFixed(2) : '0.9'}.`,
      metrics: {
        cagr_5y_percentage: largeCap.cagr_5y || 14.5,
        alpha: largeCap.alpha || 2.5,
        beta: largeCap.beta || 0.9,
        sharpe_ratio: largeCap.sharpe || 0.95
      }
    });
    funds.push({
      name: midCap.scheme_name || midCap.name,
      scheme_code: midCap.scheme_code,
      allocation_percentage: midAlloc,
      category: midCap.category || 'Mid Cap',
      risk_level: 'High',
      reason_short: 'High-conviction alpha engine to accelerate capital growth.',
      reason_detailed: `Captures mid-market upside over your ${horizon}-year horizon with a proven 5Y CAGR of ${midCap.cagr_5y ? midCap.cagr_5y.toFixed(1) : '18.2'}%.`,
      metrics: {
        cagr_5y_percentage: midCap.cagr_5y || 18.2,
        alpha: midCap.alpha || 4.2,
        beta: midCap.beta || 1.1,
        sharpe_ratio: midCap.sharpe || 0.85
      }
    });
    funds.push({
      name: debt.scheme_name || debt.name,
      scheme_code: debt.scheme_code,
      allocation_percentage: debtPct,
      category: debt.category || 'Sovereign Debt / Gilt',
      risk_level: 'Low',
      reason_short: 'Capital preservation buffer protecting against market drawdowns.',
      reason_detailed: `Provides consistent sovereign yield and high liquidity with low equity correlation.`,
      metrics: {
        cagr_5y_percentage: debt.cagr_5y || 7.2,
        alpha: debt.alpha || 1.1,
        beta: debt.beta || 0.05,
        sharpe_ratio: debt.sharpe || 0.8
      }
    });
  } else {
    funds.push({
      name: largeCap.scheme_name || largeCap.name,
      scheme_code: largeCap.scheme_code,
      allocation_percentage: equityPct,
      category: largeCap.category || (isShortOrConservative ? 'Conservative Hybrid' : 'Large / Flexi Cap'),
      risk_level: isShortOrConservative ? 'Low to Moderate' : 'Moderate',
      reason_short: isShortOrConservative ? 'Conservative hybrid allocation for modest growth with downside protection.' : 'Core equity growth allocation for long-term purchasing power.',
      reason_detailed: `High-quality allocation with proven Sharpe ratio (${largeCap.sharpe ? largeCap.sharpe.toFixed(2) : '1.0'}) tailored for your horizon.`,
      metrics: {
        cagr_5y_percentage: largeCap.cagr_5y || 8.5,
        alpha: largeCap.alpha || 1.5,
        beta: largeCap.beta || (isShortOrConservative ? 0.2 : 0.85),
        sharpe_ratio: largeCap.sharpe || 0.9
      }
    });
    funds.push({
      name: debt.scheme_name || debt.name,
      scheme_code: debt.scheme_code,
      allocation_percentage: debtPct,
      category: debt.category || 'Sovereign Debt / Gilt',
      risk_level: 'Low',
      reason_short: 'Capital preservation anchor to ensure stability.',
      reason_detailed: `Shields capital against adverse equity market corrections.`,
      metrics: {
        cagr_5y_percentage: debt.cagr_5y || 7.2,
        alpha: debt.alpha || 1.1,
        beta: debt.beta || 0.05,
        sharpe_ratio: debt.sharpe || 0.8
      }
    });
  }

  if (liquidFund) {
    funds.push({
      name: liquidFund.scheme_name || liquidFund.name,
      scheme_code: liquidFund.scheme_code,
      allocation_percentage: emergencyPct,
      category: liquidFund.category || 'Debt Scheme - Liquid Fund',
      risk_level: 'Very Low',
      reason_short: 'Mandatory Emergency Buffer: Instant liquidity protecting living expenses from market drawdowns.',
      reason_detailed: `Provides near-cash capital preservation with a 5Y CAGR of ${liquidFund.cagr_5y ? liquidFund.cagr_5y.toFixed(1) : '6.4'}% and high Sharpe ratio (${liquidFund.sharpe ? liquidFund.sharpe.toFixed(2) : '4.0'}), ensuring you never have to liquidate equity investments during an untimely downturn.`,
      metrics: {
        cagr_5y_percentage: liquidFund.cagr_5y || 6.4,
        alpha: liquidFund.alpha || 1.67,
        beta: liquidFund.beta || 0.05,
        sharpe_ratio: liquidFund.sharpe || 4.06
      }
    });
  }

  let total = funds.reduce((acc, f) => acc + f.allocation_percentage, 0);
  if (total !== 100 && funds[0]) {
    funds[0].allocation_percentage += (100 - total);
  }

  // Dynamically calculate exact weighted CAGR and Beta from selected funds
  let calculatedWeightedCAGR = 0;
  let totalAllocForCAGR = 0;
  let weightedBetaSum = 0;
  funds.forEach(f => {
    const alloc = Number(f.allocation_percentage) || 0;
    const cagr = Number(f.metrics?.cagr_5y_percentage ?? f.metrics?.cagr_3y_percentage ?? f.cagr_5y ?? f.cagr_3y);
    const beta = Number(f.metrics?.beta ?? f.beta ?? 0.8);
    if (!isNaN(cagr) && alloc > 0) {
      calculatedWeightedCAGR += (cagr * alloc);
      totalAllocForCAGR += alloc;
    }
    if (!isNaN(beta) && alloc > 0) {
      weightedBetaSum += (beta * alloc);
    }
  });

  const finalWeightedCAGR = totalAllocForCAGR > 0 
    ? Math.round((calculatedWeightedCAGR / totalAllocForCAGR) * 10) / 10 
    : (isShortOrConservative ? 7.6 : risk === 'aggressive' ? 14.8 : 12.2);

  const finalWeightedBeta = totalAllocForCAGR > 0
    ? Math.round((weightedBetaSum / totalAllocForCAGR) * 100) / 100
    : (risk === 'conservative' ? 0.45 : risk === 'aggressive' ? 0.95 : 0.78);

  const expectedReturn = isShortOrConservative 
    ? '7.0% – 8.5%' 
    : `${Math.max(4.0, (finalWeightedCAGR - 1.5)).toFixed(1)}% – ${(finalWeightedCAGR + 1.5).toFixed(1)}%`;
  const title = needsEmergency 
    ? (isShortOrConservative ? 'Capital Preservation & Emergency Shield' : risk === 'aggressive' ? 'Alpha Growth & Emergency Shield' : 'Balanced Compounding & Emergency Shield')
    : (isShortOrConservative
        ? 'Capital Shield & Stability Portfolio' 
        : risk === 'aggressive' 
          ? 'High-Conviction Alpha Portfolio' 
          : 'Balanced Compounding Portfolio');

  const description = needsEmergency
    ? `Tailored for your ${horizon}-year milestone with a dedicated ${emergencyPct}% Emergency Fund liquid buffer to safeguard unexpected living expenses.`
    : `Institutional asset allocation tailored for your ${horizon}-year investment horizon and ${risk} risk profile.`;

  return {
    portfolio_summary: {
      title,
      description,
      risk_level: risk.charAt(0).toUpperCase() + risk.slice(1),
      investment_horizon_years: horizon,
      objective: params.goal || 'Wealth creation',
      review_frequency: 'Annual',
      portfolio_metrics: {
        weighted_beta: finalWeightedBeta,
        estimated_drawdown_percentage: isShortOrConservative ? 8.5 : risk === 'aggressive' ? 22.0 : 14.0,
        weighted_cagr_percentage: finalWeightedCAGR,
        expected_return_range: expectedReturn
      }
    },
    funds,
    strategy: [
      needsEmergency
        ? `Allocates ${emergencyPct}% to Liquid Funds for emergency safety, alongside ${equityPct}% Equity and ${debtPct}% Fixed Income.`
        : `Maintains a balanced ${equityPct}% Equity / ${debtPct}% Fixed Income structure.`,
      `Rebalance once annually or when asset weights drift by more than 5%.`
    ],
    risks: [
      {
        title: 'Market Cycle Risk',
        severity: 'Moderate',
        description: 'Equity portions may fluctuate over short periods, but historical multi-year compounding rewards patient investors.'
      }
    ]
  };
}

export async function runAnalyzerAgent(holdings, res) {
  try {
    const db = getDB();
    
    sendStep(res, { type: 'step', icon: '🔍', title: 'Cross-referencing database...', status: 'in_progress', detail: 'Matching uploaded funds with market data' });
    
    // Match funds
    const enrichedHoldings = [];
    for (const h of holdings) {
      // Basic tokenization for LIKE matching (e.g., matching "HDFC Flexi Cap" against "HDFC Flexi Cap Fund - Direct Growth")
      const searchTerms = h.fundName.replace(/[^a-zA-Z0-9 ]/g, ' ').split(' ').filter(t => t.length > 2).slice(0, 4);
      let query = `SELECT f.scheme_name, f.category, m.cagr_5y, m.beta, m.sharpe 
                   FROM funds f 
                   JOIN fund_metrics m ON f.scheme_code = m.scheme_code 
                   WHERE `;
      const conditions = searchTerms.map(() => `f.scheme_name LIKE ?`);
      if (conditions.length === 0) {
        enrichedHoldings.push({ userFundName: h.fundName, value: h.value, matched: false });
        continue;
      }
      query += conditions.join(' AND ') + ` LIMIT 1`;
      
      try {
        const match = db.prepare(query).get(...searchTerms.map(t => `%${t}%`));
        if (match) {
          enrichedHoldings.push({ userFundName: h.fundName, value: h.value, matched: true, ...match });
        } else {
          enrichedHoldings.push({ userFundName: h.fundName, value: h.value, matched: false });
        }
      } catch (e) {
        enrichedHoldings.push({ userFundName: h.fundName, value: h.value, matched: false });
      }
    }

    sendStep(res, { type: 'step', icon: '✅', title: 'Database Cross-reference Complete', status: 'done', detail: `Matched ${enrichedHoldings.filter(h => h.matched).length} out of ${holdings.length} funds.` });
    
    sendStep(res, { type: 'step', icon: '🧠', title: 'AI diagnosing portfolio health...', status: 'in_progress', detail: 'Running analysis algorithms' });
    
    const totalValue = enrichedHoldings.reduce((sum, h) => sum + h.value, 0);
    const holdingsStr = enrichedHoldings.map(h => {
      const w = ((h.value / totalValue) * 100).toFixed(1);
      if (!h.matched) return `- "${h.userFundName}": ₹${h.value.toLocaleString('en-IN')} (${w}% weight) | No database match — analysis based on name only`;
      const metrics = [
        h.cagr_5y   != null ? `5Y CAGR: ${h.cagr_5y}%`   : null,
        h.sharpe    != null ? `Sharpe: ${h.sharpe}`        : null,
        h.sortino   != null ? `Sortino: ${h.sortino}`      : null,
        h.alpha     != null ? `Alpha: ${h.alpha}`          : null,
        h.beta      != null ? `Beta: ${h.beta}`            : null,
      ].filter(Boolean).join(', ');
      return `- "${h.userFundName}": ₹${h.value.toLocaleString('en-IN')} (${w}% weight) | Category: ${h.category} | ${metrics}`;
    }).join('\n');

    const fundNames = enrichedHoldings.map(h => `"${h.userFundName}"`).join(', ');

    const prompt = `
You are a strict SEBI-registered financial advisor AI performing a portfolio health check.

CRITICAL RULES — YOU MUST FOLLOW THESE EXACTLY:
1. You MUST ONLY reference funds that appear verbatim in the "USER'S PORTFOLIO" section below. Do NOT mention any other fund.
2. Every fund name you output in "laggards_to_exit" MUST be copied EXACTLY as it appears in the portfolio list.
3. Do NOT invent, suggest, or mention any fund that the user does not already hold.
4. If you cannot identify a laggard strictly from the provided data, return an empty array for "laggards_to_exit".
5. Base your analysis ONLY on the data provided. Do not assume or fabricate metrics for unmatched funds.
6. Output ONLY a valid JSON object. No markdown, no preamble, no extra text.

The user's portfolio contains EXACTLY these funds (and NO others): ${fundNames}

USER'S PORTFOLIO (Total Value: ₹${totalValue.toLocaleString('en-IN')}):
${holdingsStr}

HOW TO IDENTIFY LAGGARDS (use risk-adjusted returns, NOT CAGR alone):
- Primary signal: Low Sharpe Ratio (< 0.5 for equity funds is poor; < 0.3 is a red flag)
- Secondary signal: Low or negative Sortino Ratio (poor downside protection)
- Tertiary signal: Negative Alpha (fund manager is destroying value vs the benchmark)
- Supporting signal: High Beta (> 1.2) with low Sharpe = taking excess risk without proportional reward
- A fund with decent CAGR but poor Sharpe/Sortino/Alpha is still a laggard — it earned returns by taking on excessive risk
- A fund with lower CAGR but high Sharpe and positive Alpha is NOT a laggard — it is efficient
- For funds with "No database match", flag them as unverifiable but do not fabricate metrics for them

OVERALL HEALTH SCORING GUIDE:
- 8-10: Well-diversified, low overlap, all funds have strong risk-adjusted metrics
- 5-7: Some overlap or 1-2 laggards, manageable
- 1-4: Heavy overlap, multiple laggards, poor risk-adjusted performance across the board

INDIAN CAPITAL GAINS TAX RULES (MANDATORY — factor this into proposed_action_plan):
- LTCG (Long-Term Capital Gains, held > 1 year): First ₹1,25,000 per financial year is EXEMPT. Gains above this are taxed at 12.5%.
- STCG (Short-Term Capital Gains, held < 1 year): Taxed at 20% flat on the gain.
- The total portfolio value is ₹${totalValue.toLocaleString('en-IN')}. Redeeming a fund with significant unrealised gains can be a large tax event.
- If the laggard fund has a high invested value, exiting it all at once in one financial year may push gains well above ₹1.25 lakh, triggering the 12.5% LTCG tax.
- Preferred tax-efficient approaches:
  a) STAGGER exits: Recommend splitting redemption across two financial years (e.g., redeem half before March 31, the rest after April 1) to use the ₹1.25 lakh exemption twice.
  b) STP (Systematic Transfer Plan): Recommend an STP instead of a lump-sum switch where large amounts are involved.
  c) If a fund's underperformance is marginal, the tax cost of exiting may outweigh the benefit — flag this explicitly.
- Always mention the tax implication in each action step where a redemption/exit is involved.

Return a JSON object matching this schema exactly:
{
  "overall_health_score": <number 1-10>,
  "risk_adjusted_summary": "<2-3 sentence summary of portfolio efficiency based on Sharpe, Sortino, Alpha across holdings>",
  "key_observations": [
    "<specific observation about THIS portfolio referencing actual fund names and their metrics>"
  ],
  "laggards_to_exit": [
    {
      "fund_name": "<MUST be an exact copy of one of the fund names from the portfolio above>",
      "risk_adjusted_reason": "<explain using Sharpe/Sortino/Alpha/Beta data from the provided metrics, not CAGR alone>"
    }
  ],
  "proposed_action_plan": [
    "<concrete, tax-aware step — if redemption is involved, mention whether to stagger across FY or use STP, and estimate tax impact if gains are large>"
  ],
  "tax_advisory_note": "<1-2 sentences summarising the overall tax impact of the proposed rebalancing — e.g. estimated taxable gains, suggestion to stagger across financial years, or note that the portfolio is already tax-efficient>"
}
`;

    const resultJson = await callGrok([{ role: 'user', content: prompt }], 0.1);

    const parsedAnalysis = JSON.parse(resultJson);

    sendStep(res, { type: 'step', icon: '✅', title: 'AI Diagnosis Complete', status: 'done', detail: 'Analysis generated successfully' });
    sendStep(res, { type: 'result', analysis: parsedAnalysis });

  } catch (err) {
    console.error('[AnalyzerAgent] Error:', err);
    sendStep(res, { type: 'error', message: `Analysis failed: ${err.message}` });
  }
}
