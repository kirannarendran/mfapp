const BASE_URL = '/api';

export const EQUITY_BENCHMARK_CODE = 100484; // Franklin India NSE Nifty 50 Index Fund
export const DEBT_BENCHMARK_CODE = 120137;   // SBI 10 Year Constant Maturity Gilt Fund (CRISIL 10Y Gilt Benchmark)
export const BENCHMARK_INDEX_CODE = EQUITY_BENCHMARK_CODE; // Backward compatibility

/**
 * Check if a fund category belongs to Debt / Fixed Income / Money Market
 */
export const isDebtCategory = (category = '') => {
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
};

/**
 * Get benchmark scheme code and descriptive labels based on scheme category
 */
export const getBenchmarkInfoForCategory = (category = '') => {
  if (isDebtCategory(category)) {
    return {
      code: DEBT_BENCHMARK_CODE,
      name: '10Y Sovereign G-Sec Benchmark',
      shortName: '10Y G-Sec Index',
      fullName: 'CRISIL 10-Yr Constant Maturity Gilt Index',
      assetClass: 'Debt',
      reason: 'Debt and fixed income schemes are benchmarked against 10-Year Government Securities yield to measure interest-rate & credit alpha accurately.'
    };
  }
  return {
    code: EQUITY_BENCHMARK_CODE,
    name: 'Nifty 50 TRI Benchmark',
    shortName: 'Nifty 50 Index',
    fullName: 'NSE Nifty 50 Total Return Index',
    assetClass: 'Equity',
    reason: 'Equity mutual funds are benchmarked against the Nifty 50 Index to measure market beta and excess alpha generation.'
  };
};

/**
 * Search funds on the server (Direct Growth only).
 */
export const fetchFundList = async (searchTerm = '') => {
  try {
    const url = searchTerm
      ? `${BASE_URL}/funds?search=${encodeURIComponent(searchTerm)}&limit=50`
      : `${BASE_URL}/funds`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch fund list');
    const result = await response.json();
    return result.funds.map(f => ({
      schemeCode: f.scheme_code,
      schemeName: f.scheme_name,
      fundHouse: f.fund_house,
      category: f.category,
      lastNav: f.last_nav,
      lastNavDate: f.last_nav_date,
      cagr3y: f.cagr_3y,
      cagr5y: f.cagr_5y,
      sharpe: f.sharpe,
      beta: f.beta,
      isCurated: result.isCurated || false
    }));
  } catch (error) {
    console.error('Error fetching fund list:', error);
    throw error;
  }
};

/**
 * Fetch fund details (metadata + NAV history) from backend cache.
 */
export const fetchFundDetails = async (schemeCode) => {
  try {
    const response = await fetch(`${BASE_URL}/funds/${schemeCode}`);
    if (!response.ok) throw new Error('Failed to fetch fund details');
    return await response.json();
  } catch (error) {
    console.error('Error fetching fund details:', error);
    throw error;
  }
};

/**
 * Fetch pre-computed metrics for a fund.
 */
export const fetchFundMetrics = async (schemeCode) => {
  try {
    const response = await fetch(`${BASE_URL}/funds/${schemeCode}/metrics`);
    if (!response.ok) throw new Error('Failed to fetch fund metrics');
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error fetching fund metrics:', error);
    throw error;
  }
};

/**
 * Fetch filtered funds with pagination support.
 */
export const fetchScreenerFunds = async (params = {}) => {
  try {
    const query = new URLSearchParams();
    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== null && params[key] !== '') {
        if (key === 'category' && params[key] === 'All') return; // Skip 'All' category filter
        query.append(key, params[key]);
      }
    });

    const response = await fetch(`${BASE_URL}/funds/screener?${query.toString()}`);
    if (!response.ok) throw new Error('Failed to screen funds');
    return await response.json();
  } catch (error) {
    console.error('Error screening funds:', error);
    throw error;
  }
};

export const fetchSyncStatus = async () => {
  try {
    const response = await fetch(`${BASE_URL}/sync/status`);
    if (!response.ok) throw new Error('Failed to fetch sync status');
    return await response.json();
  } catch (error) {
    console.error('Error fetching sync status:', error);
    throw error;
  }
};

export const triggerManualSync = async () => {
  try {
    const response = await fetch(`${BASE_URL}/sync/manual`, {
      method: 'POST',
    });
    if (!response.ok) {
      if (response.status === 409) {
        return;
      }
      throw new Error('Failed to trigger manual sync');
    }
    return await response.json();
  } catch (error) {
    console.error('Error triggering manual sync:', error);
    throw error;
  }
};

/**
 * Fetch comparison data for multiple funds in a single request.
 */
export const fetchComparison = async (schemeCodes) => {
  try {
    const codes = schemeCodes.join(',');
    const response = await fetch(`${BASE_URL}/funds/compare?codes=${codes}`);
    if (!response.ok) throw new Error('Failed to fetch comparison');
    const result = await response.json();
    return result.funds;
  } catch (error) {
    console.error('Error fetching comparison:', error);
    throw error;
  }
};

/**
 * Fetch benchmark fund data (defaults to configured benchmark or takes explicit code/type).
 */
export const fetchBenchmark = async (benchmarkCode = null) => {
  try {
    const url = benchmarkCode ? `${BASE_URL}/benchmark?code=${benchmarkCode}` : `${BASE_URL}/benchmark`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch benchmark');
    return await response.json();
  } catch (error) {
    console.error('Error fetching benchmark:', error);
    throw error;
  }
};
