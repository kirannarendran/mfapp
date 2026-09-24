// Risk Free Rate assumption (approx annual 10-year G-Sec yield)
const RISK_FREE_RATE = 0.07;

// Helper to parse date dd-mm-yyyy to Date object
const parseDate = (dateStr) => {
    if (!dateStr) return new Date();
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

    const currentNav = parseFloat(filtered[0].nav);
    const oldNav = parseFloat(filtered[filtered.length - 1].nav);

    if (isNaN(currentNav) || isNaN(oldNav) || currentNav <= 0 || oldNav <= 0) return null;

    const actualTimeDiff = (parseDate(filtered[0].date) - parseDate(filtered[filtered.length - 1].date)) / (1000 * 3600 * 24 * 365.25);

    if (actualTimeDiff < 0.5) return null; // Not enough data

    const cagr = (Math.pow(currentNav / oldNav, 1 / actualTimeDiff) - 1) * 100;
    return isFinite(cagr) ? cagr.toFixed(2) : null;
};

// Helper to get daily returns
const getDailyReturns = (data) => {
    const returns = [];
    for (let i = data.length - 2; i >= 0; i--) {
        const current = parseFloat(data[i].nav);
        const prev = parseFloat(data[i + 1].nav);
        if (prev > 0 && !isNaN(current) && !isNaN(prev)) {
            returns.push((current - prev) / prev);
        }
    }
    return returns;
};

// 2. Risk Metrics Calculation (Alpha, Beta, StdDev, Sharpe, Sortino)
export const calculateRiskMetrics = (fundData, benchmarkData) => {
    if (!fundData || !benchmarkData) return null;

    const fundMap = new Map(fundData.map(i => [i.date, parseFloat(i.nav)]));
    const alignedFund = [];
    const alignedBench = [];

    const cutoffDate = new Date();
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 3);

    benchmarkData.forEach(item => {
        const d = parseDate(item.date);
        if (d >= cutoffDate && fundMap.has(item.date)) {
            alignedBench.push({ date: item.date, nav: parseFloat(item.nav) });
            alignedFund.push({ date: item.date, nav: fundMap.get(item.date) });
        }
    });

    if (alignedFund.length < 200) return null;

    const fundReturns = getDailyReturns(alignedFund);
    const benchReturns = getDailyReturns(alignedBench);

    if (fundReturns.length === 0 || fundReturns.length !== benchReturns.length) return null;

    const n = fundReturns.length;

    const meanFundRet = fundReturns.reduce((a, b) => a + b, 0) / n;
    const meanBenchRet = benchReturns.reduce((a, b) => a + b, 0) / n;

    // A. Standard Deviation (Annualized)
    const variance = fundReturns.reduce((sum, r) => sum + Math.pow(r - meanFundRet, 2), 0) / (n - 1);
    const stdDevDaily = Math.sqrt(Math.max(variance, 0));
    const stdDevAnnual = stdDevDaily * Math.sqrt(252) * 100;

    // B. Beta
    let covariance = 0;
    let benchVariance = 0;
    for (let i = 0; i < n; i++) {
        covariance += (fundReturns[i] - meanFundRet) * (benchReturns[i] - meanBenchRet);
        benchVariance += Math.pow(benchReturns[i] - meanBenchRet, 2);
    }
    const beta = benchVariance > 1e-9 ? covariance / benchVariance : 1.0;

    // C. Alpha (Jensen's Alpha)
    const annualFundRet = Math.pow(Math.max(1 + meanFundRet, 0.0001), 252) - 1;
    const annualBenchRet = Math.pow(Math.max(1 + meanBenchRet, 0.0001), 252) - 1;
    const alpha = (annualFundRet - (RISK_FREE_RATE + beta * (annualBenchRet - RISK_FREE_RATE))) * 100;

    // D. Sharpe Ratio
    const sharpe = (stdDevAnnual > 0.001) ? (annualFundRet - RISK_FREE_RATE) / (stdDevAnnual / 100) : null;

    // E. Sortino Ratio
    const dailyRiskFree = Math.pow(1 + RISK_FREE_RATE, 1 / 252) - 1;
    const downsideSquaredSum = fundReturns.reduce((sum, r) => {
        const diff = r - dailyRiskFree;
        return sum + (diff < 0 ? Math.pow(diff, 2) : 0);
    }, 0);
    const downsideDevDaily = Math.sqrt(downsideSquaredSum / n);
    const downsideDevAnnual = downsideDevDaily * Math.sqrt(252);
    const sortino = (downsideDevAnnual > 0.0001) ? (annualFundRet - RISK_FREE_RATE) / downsideDevAnnual : null;

    return {
        stdDev: isFinite(stdDevAnnual) ? stdDevAnnual.toFixed(2) : null,
        beta: isFinite(beta) ? beta.toFixed(2) : null,
        alpha: isFinite(alpha) ? alpha.toFixed(2) : null,
        sharpe: isFinite(sharpe) ? sharpe.toFixed(2) : null,
        sortino: isFinite(sortino) ? sortino.toFixed(2) : null
    };
};

// 3. Upside / Downside Capture
export const calculateCaptureRatios = (fundData, benchmarkData) => {
    if (!fundData || !benchmarkData) return null;

    const fundMap = new Map(fundData.map(i => [i.date, parseFloat(i.nav)]));
    const alignedFund = [];
    const alignedBench = [];

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

    const fundReturns = getDailyReturns(alignedFund);
    const benchReturns = getDailyReturns(alignedBench);

    if (fundReturns.length === 0) return null;

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

    const upsideCapture = Math.abs(upBench - 1) > 1e-7 ? ((upFund - 1) / (upBench - 1)) * 100 : null;
    const downsideCapture = Math.abs(downBench - 1) > 1e-7 ? ((downFund - 1) / (downBench - 1)) * 100 : null;

    return {
        upside: (upsideCapture !== null && isFinite(upsideCapture)) ? Math.round(upsideCapture) : null,
        downside: (downsideCapture !== null && isFinite(downsideCapture)) ? Math.round(downsideCapture) : null
    };
};
