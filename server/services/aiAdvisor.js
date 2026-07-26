import { getDB } from '../db.js';

const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.GROK_API_KEY; 
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

/**
 * Strip markdown code fences that Grok sometimes adds despite being told not to.
 * e.g. ```json\n{...}\n``` → {...}
 */
function stripCodeFences(text) {
  return text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/, '')
    .trim();
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

  let maxBeta = 1.5;
  let minSharpe = 0.3;
  let minCAGR5Y = params.expectedCAGR ? params.expectedCAGR - 5 : 8;

  if (params.maxDrawdownPct) {
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
  if (params.fundCategory && params.fundCategory !== 'Any Equity') {
    if (params.fundCategory === 'Large Cap') categoryFilter = "AND (f.category LIKE '%Large Cap%')";
    else if (params.fundCategory === 'Mid & Small Cap') categoryFilter = "AND (f.category LIKE '%Mid Cap%' OR f.category LIKE '%Small Cap%')";
    else if (params.fundCategory === 'Flexi/Multi Cap') categoryFilter = "AND (f.category LIKE '%Flexi Cap%' OR f.category LIKE '%Multi Cap%')";
    else if (params.fundCategory === 'Sectoral/Thematic') categoryFilter = "AND (f.category LIKE '%Sectoral%' OR f.category LIKE '%Thematic%')";
    else if (params.fundCategory === 'Index Funds') categoryFilter = "AND f.category LIKE '%Index%'";
    else if (params.fundCategory === 'Debt/Hybrid') categoryFilter = "AND (f.category LIKE '%Debt%' OR f.category LIKE '%Hybrid%')";
  } else {
    categoryFilter = "AND (f.category LIKE '%Equity%' OR f.category LIKE '%ELSS%')";
  }

  const orderByClause = params.riskProfile === 'conservative' 
    ? 'ORDER BY m.sharpe DESC, m.std_dev ASC'
    : 'ORDER BY m.cagr_5y DESC, m.alpha DESC';

  const rows = db.prepare(`
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
      AND m.beta IS NOT NULL
    ${orderByClause}
    LIMIT 30
  `).all(minCAGR5Y, maxBeta, minSharpe);

  return { rows, filters: { maxBeta, minSharpe, minCAGR5Y } };
}

/**
 * Step 3 — Use Grok to reason over the screened funds and produce a recommendation.
 */
async function generateRecommendation(params, screenedFunds) {
  const fundSummary = screenedFunds.slice(0, 15).map((f, i) =>
    `${i + 1}. ${f.scheme_name} | Category: ${f.category} | 5Y CAGR: ${f.cagr_5y?.toFixed(1)}% | Sharpe: ${f.sharpe?.toFixed(2)} | Beta: ${f.beta?.toFixed(2)} | Sortino: ${f.sortino?.toFixed(2)} | Alpha: ${f.alpha?.toFixed(2)}`
  ).join('\n');

  return await callGrok([
    {
      role: 'system',
      content: `You are an expert Indian mutual fund advisor. A user has shared their investment goals and our risk engine has pre-screened eligible funds. 
Select exactly ${params.numberOfFunds || '3-4'} funds from the pre-screened list, suggest % SIP allocation for each, explain WHY each suits this user's specific profile, provide a portfolio strategy summary, and mention 1-2 key risks.
IMPORTANT: Only recommend funds from the pre-screened list.

Return valid JSON only. Do not wrap the response in markdown. Do not include commentary before or after the JSON. Keep recommendation reasons concise. Do not invent metrics. Use null for unavailable values. Ensure allocations total exactly 100.

Use this JSON schema:
{
  "portfolio_summary": {
    "title": "Moderate Growth Portfolio",
    "description": "A diversified portfolio designed for long-term wealth creation.",
    "risk_level": "Moderate",
    "investment_horizon_years": 20,
    "objective": "Long-term growth",
    "review_frequency": "Annual",
    "portfolio_metrics": {
      "weighted_beta": 0.95,
      "estimated_drawdown_percentage": 15.5,
      "weighted_cagr_percentage": 18.2
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
      "name": "DSP India T.I.G.E.R. Fund",
      "allocation_percentage": 30,
      "category": "Thematic",
      "risk_level": "High",
      "reason_short": "Strong long-term growth potential.",
      "reason_detailed": "Detailed reasoning goes here.",
      "metrics": {
        "cagr_5y_percentage": 23.5,
        "alpha": 15.72,
        "beta": null,
        "sharpe_ratio": null
      }
    }
  ],
  "strategy": [
    "Diversifies exposure across multiple fund categories."
  ],
  "risks": [
    {
      "title": "Market volatility",
      "severity": "Moderate",
      "description": "The portfolio may experience short-term declines."
    }
  ],
  "disclaimer": "Mutual fund investments are subject to market risks. Historical performance does not guarantee future returns."
}`,
    },
    {
      role: 'user',
      content: `USER PROFILE:
- Goal: ${params.goal || 'Wealth creation'}
- Investment Horizon: ${params.horizonYears ? params.horizonYears + ' years' : 'Not specified'}
- Monthly SIP: ${params.monthlySIP ? '₹' + params.monthlySIP.toLocaleString('en-IN') : 'Not specified'}
- Max Drawdown Tolerated: ${params.maxDrawdownPct ? params.maxDrawdownPct + '%' : 'Not specified'}
- Requested Fund Count: ${params.numberOfFunds || 'Not specified'}

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
  if (!GROQ_API_KEY) {
    sendStep(res, { type: 'error', message: 'API key is not set. Please add it to your .env file.' });
    return;
  }

  try {
    // ── Step 1: Screen funds ──────────────────────────────────────────────────
    sendStep(res, { type: 'step', icon: '🔍', title: 'Screening funds against your risk constraints...', status: 'loading' });

    let screenResult;
    try {
      screenResult = screenFunds(params);
    } catch (e) {
      console.error('[runStructuredAdvisorAgent] screenFunds failed:', e);
      sendStep(res, { type: 'error', message: 'Fund database not available. Please ensure the server data is synced.' });
      return;
    }

    const { rows: screenedFunds, filters } = screenResult;
    const eliminated = 1400 - screenedFunds.length;

    sendStep(res, {
      type: 'step', icon: '📊', title: 'Screening funds against your risk constraints...', status: 'done',
      detail: `Filters: Beta ≤ ${filters.maxBeta} · Sharpe ≥ ${filters.minSharpe} · 5Y CAGR ≥ ${filters.minCAGR5Y}% · ~${eliminated} funds eliminated · ${screenedFunds.length} shortlisted`
    });

    if (screenedFunds.length === 0) {
      sendStep(res, { type: 'error', message: 'No funds matched your risk constraints. Try relaxing your drawdown tolerance.' });
      return;
    }
    
    // Randomize the shortlisted funds before sending to AI to increase portfolio variety
    const shuffledFunds = [...screenedFunds].sort(() => 0.5 - Math.random());

    // ── Step 2: Rank & reason ─────────────────────────────────────────────────
    sendStep(res, { type: 'step', icon: '🧠', title: 'AI reasoning over shortlisted funds...', status: 'loading' });

    const recommendationStr = await generateRecommendation(params, shuffledFunds);
    
    let parsedRecommendation;
    try {
      parsedRecommendation = JSON.parse(recommendationStr);
      
      if (!parsedRecommendation.portfolio_summary || !parsedRecommendation.funds) {
         throw new Error("Missing required portfolio fields (portfolio_summary, funds)");
      }
      
      let sum = 0;
      parsedRecommendation.funds.forEach(f => sum += Number(f.allocation_percentage || 0));
      if (Math.round(sum) !== 100) {
         throw new Error(`Portfolio allocations sum to ${Math.round(sum)}%, but must be exactly 100%`);
      }
      
    } catch (e) {
      sendStep(res, { type: 'error', message: `Server validation failed: ${e.message}` });
      return;
    }

    sendStep(res, {
      type: 'step', icon: '✅', title: 'AI reasoning over shortlisted funds...', status: 'done',
      detail: `Analysed top ${Math.min(screenedFunds.length, 15)} funds for best risk-return fit`
    });

    sendStep(res, { type: 'result', recommendation: parsedRecommendation });

  } catch (err) {
    console.error('[AIAdvisor] Error:', err);
    sendStep(res, { type: 'error', message: `Error: ${err.message}` });
  }
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
