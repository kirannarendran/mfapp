// Risk Free Rate assumption (approx annual 10-year G-Sec yield)
const RISK_FREE_RATE = 0.07;

// Helper to parse date dd-mm-yyyy to Date object
const parseDate = (dateStr) => {
    const [day, month, year] = dateStr.split('-');
    return new Date(`${year}-${month}-${day}`);
};

// Helper to filter data by years
const filterByYears = (data, years) => {
    if (!data || data.length === 0) return [];
    const latestDate = parseDate(data[0].date); // Assuming sorted desc
    const cutoffDate = new Date(latestDate);
    cutoffDate.setFullYear(latestDate.getFullYear() - years);

    return data.filter(item => parseDate(item.date) >= cutoffDate);
};

// 1. CAGR Calculation
export const calculateCAGR = (data, years) => {
    const filtered = filterByYears(data, years);
    if (filtered.length < 2) return null;

    // Get strictly the start and end of the period
    // API returns newest first. So index 0 is Current, last index is Oldest.
    const currentNav = parseFloat(filtered[0].nav);
    const oldNav = parseFloat(filtered[filtered.length - 1].nav);

    const actualTimeDiff = (parseDate(filtered[0].date) - parseDate(filtered[filtered.length - 1].date)) / (1000 * 3600 * 24 * 365.25);

    if (actualTimeDiff < 0.5) return null; // Not enough data

    const cagr = (Math.pow(currentNav / oldNav, 1 / actualTimeDiff) - 1) * 100;
    return cagr.toFixed(2);
};

// Helper to get daily returns
const getDailyReturns = (data) => {
    const returns = [];
    // Data is newest first. Iterate backwards to calculate chronological returns
    for (let i = data.length - 2; i >= 0; i--) {
        const current = parseFloat(data[i].nav);
        const prev = parseFloat(data[i + 1].nav);
        returns.push((current - prev) / prev);
    }
    return returns;
};

// 2. Risk Metrics Calculation (Alpha, Beta, StdDev, Sharpe, Sortino)
export const calculateRiskMetrics = (fundData, benchmarkData) => {
    // 1. Align data to same dates (intersection)
    const fundMap = new Map(fundData.map(i => [i.date, parseFloat(i.nav)]));
    const alignedFund = [];
    const alignedBench = [];

    // Use 3 years of data for risk ratios usually
    const cutoffDate = new Date();
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 3);

    benchmarkData.forEach(item => {
        const d = parseDate(item.date);
        if (d >= cutoffDate && fundMap.has(item.date)) {
            alignedBench.push({ date: item.date, nav: parseFloat(item.nav) });
            alignedFund.push({ date: item.date, nav: fundMap.get(item.date) });
        }
    });

    if (alignedFund.length < 200) return null; // Need sufficient data points

    // Calculate Daily Returns
    const fundReturns = getDailyReturns(alignedFund);
    const benchReturns = getDailyReturns(alignedBench);

    if (fundReturns.length !== benchReturns.length) return null;

    const n = fundReturns.length;

    // Mean Daily Returns
    const meanFundRet = fundReturns.reduce((a, b) => a + b, 0) / n;
    const meanBenchRet = benchReturns.reduce((a, b) => a + b, 0) / n;

    // A. Standard Deviation (Annualized)
    const variance = fundReturns.reduce((sum, r) => sum + Math.pow(r - meanFundRet, 2), 0) / (n - 1);
    const stdDevDaily = Math.sqrt(variance);
    const stdDevAnnual = stdDevDaily * Math.sqrt(252) * 100;

    // B. Beta
    // Covariance(Fund, Bench) / Variance(Bench)
    let covariance = 0;
    let benchVariance = 0;
    for (let i = 0; i < n; i++) {
        covariance += (fundReturns[i] - meanFundRet) * (benchReturns[i] - meanBenchRet);
        benchVariance += Math.pow(benchReturns[i] - meanBenchRet, 2);
    }
    const beta = covariance / benchVariance;

    // C. Alpha (Jensen's Alpha)
    // Alpha = FundReturn - (RiskFree + Beta * (BenchReturn - RiskFree))
    // We use annualized returns for this formula
    const annualFundRet = Math.pow(1 + meanFundRet, 252) - 1;
    const annualBenchRet = Math.pow(1 + meanBenchRet, 252) - 1;
    const alpha = (annualFundRet - (RISK_FREE_RATE + beta * (annualBenchRet - RISK_FREE_RATE))) * 100;

    // D. Sharpe Ratio
    // (FundReturn - RiskFree) / StdDev
    const sharpe = (annualFundRet - RISK_FREE_RATE) / (stdDevAnnual / 100);

    // E. Sortino Ratio
    // (FundReturn - RiskFree) / DownsideDeviation
    // Downside Deviation considers only negative returns relative to MAR (Risk Free)
    // Converted to daily risk free for comparison
    const dailyRiskFree = Math.pow(1 + RISK_FREE_RATE, 1 / 252) - 1;
    const downsideSquaredSum = fundReturns.reduce((sum, r) => {
        const diff = r - dailyRiskFree;
        return sum + (diff < 0 ? Math.pow(diff, 2) : 0);
    }, 0);
    const downsideDevDaily = Math.sqrt(downsideSquaredSum / n);
    const downsideDevAnnual = downsideDevDaily * Math.sqrt(252);
    const sortino = (annualFundRet - RISK_FREE_RATE) / downsideDevAnnual;

    return {
        stdDev: stdDevAnnual.toFixed(2),
        beta: beta.toFixed(2),
        alpha: alpha.toFixed(2),
        sharpe: sharpe.toFixed(2),
        sortino: sortino.toFixed(2)
    };
};

// 3. Upside / Downside Capture
export const calculateCaptureRatios = (fundData, benchmarkData) => {
    // Similar alignment logic - use 3Y or 5Y data
    const fundMap = new Map(fundData.map(i => [i.date, parseFloat(i.nav)]));
    const alignedFund = [];
    const alignedBench = [];

    // Look back 5 years for capture ratios if possible
    const cutoffDate = new Date();
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 5);

    benchmarkData.forEach(item => {
        const d = parseDate(item.date);
        if (d >= cutoffDate && fundMap.has(item.date)) {
            alignedBench.push({ date: item.date, nav: parseFloat(item.nav) });
            alignedFund.push({ date: item.date, nav: fundMap.get(item.date) });
        }
    });

    if (alignedFund.length < 200) return null;

    // Returns calculated monthly for Capture Ratios is standard, but we'll use daily compounded to monthly logic or just daily for simplicity as it aligns directionally.
    // Let's use daily for granularity.
    const fundReturns = getDailyReturns(alignedFund);
    const benchReturns = getDailyReturns(alignedBench);

    let upsideFundSum = 0;
    let upsideBenchSum = 0;
    let downsideFundSum = 0;
    let downsideBenchSum = 0;

    for (let i = 0; i < fundReturns.length; i++) {
        if (benchReturns[i] >= 0) {
            upsideBenchSum += benchReturns[i];
            upsideFundSum += fundReturns[i];
        } else {
            downsideBenchSum += benchReturns[i];
            downsideFundSum += fundReturns[i];
        }
    }

    // Simple Ratio: (Fund Up Return / Bench Up Return) * 100
    // Note: Mathematical definition uses geometric linking, but simple sum approximation is often used for daily high-freq data summaries.
    // Ideally we should compound them. Let's do simple compounding.

    // Re-calculating using compounded returns for Upside/Downside periods
    let upFund = 1;
    let upBench = 1;
    let downFund = 1;
    let downBench = 1;

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
        downside: Math.round(downsideCapture)
    };
};
