const BASE_URL = '/api';

export const BENCHMARK_INDEX_CODE = 100484; // Franklin Nifty 50 Index Fund

/**
 * Search funds on the server (Direct Growth only).
 * Server-side search with SQL — no more downloading 40k items.
 */
export const fetchFundList = async (searchTerm = '') => {
  try {
    const url = searchTerm
      ? `${BASE_URL}/funds?search=${encodeURIComponent(searchTerm)}&limit=50`
      : `${BASE_URL}/funds`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch fund list');
    const result = await response.json();
    // Return in the same format the frontend expects: array of { schemeCode, schemeName }
    return result.funds.map(f => ({
      schemeCode: f.scheme_code,
      schemeName: f.scheme_name,
    }));
  } catch (error) {
    console.error('Error fetching fund list:', error);
    throw error;
  }
};

/**
 * Fetch fund details (metadata + NAV history) from local backend cache.
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

export const fetchScreenerFunds = async (params) => {
  try {
    const query = new URLSearchParams();
    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== null) {
        if (key === 'category' && params[key] === 'All') return; // Skip 'All' category
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
 * Fetch benchmark fund data.
 */
export const fetchBenchmark = async () => {
  try {
    const response = await fetch(`${BASE_URL}/benchmark`);
    if (!response.ok) throw new Error('Failed to fetch benchmark');
    return await response.json();
  } catch (error) {
    console.error('Error fetching benchmark:', error);
    throw error;
  }
};
