import React, { useState, useEffect, useMemo } from 'react';
import { 
    fetchFundDetails, 
    fetchFundMetrics, 
    EQUITY_BENCHMARK_CODE, 
    DEBT_BENCHMARK_CODE, 
    getBenchmarkInfoForCategory, 
    isDebtCategory 
} from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const FundDetail = ({ schemeCode, onBack }) => {
    const [details, setDetails] = useState(null);
    const [benchmark, setBenchmark] = useState(null);
    const [selectedBenchmarkCode, setSelectedBenchmarkCode] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeRange, setTimeRange] = useState('1Y'); // 1M, 6M, 1Y, 3Y, 5Y, ALL
    const [stats, setStats] = useState(null);
    const [riskPeriod, setRiskPeriod] = useState('3Y'); // 3Y or 5Y

    // Initial load: Fund Details + Fund Metrics
    useEffect(() => {
        const loadFundData = async () => {
            try {
                setLoading(true);
                setError(null);
                
                const [fundRes, metricsRes] = await Promise.allSettled([
                    fetchFundDetails(schemeCode),
                    fetchFundMetrics(schemeCode)
                ]);
                
                if (fundRes.status === 'fulfilled' && fundRes.value) {
                    setDetails(fundRes.value);
                    
                    // Determine appropriate benchmark based on category
                    const category = fundRes.value.meta?.scheme_category || '';
                    const defaultBench = getBenchmarkInfoForCategory(category);
                    setSelectedBenchmarkCode(defaultBench.code);
                } else {
                    throw new Error('Failed to load fund details');
                }

                if (metricsRes.status === 'fulfilled' && metricsRes.value) {
                    const metricsData = metricsRes.value;
                    setStats({
                        returns: { 
                            '6M': metricsData.return_6m,
                            '1Y': metricsData.cagr_1y, 
                            '3Y': metricsData.cagr_3y, 
                            '5Y': metricsData.cagr_5y 
                        },
                        risk: {
                            '3Y': {
                                alpha: metricsData.alpha,
                                beta: metricsData.beta,
                                sharpe: metricsData.sharpe,
                                sortino: metricsData.sortino,
                                stdDev: metricsData.std_dev
                            },
                            '5Y': {
                                alpha: metricsData.alpha_5y,
                                beta: metricsData.beta_5y,
                                sharpe: metricsData.sharpe_5y,
                                sortino: metricsData.sortino_5y,
                                stdDev: metricsData.std_dev_5y
                            }
                        },
                        capture: {
                            '3Y': {
                                upside: metricsData.upside_capture_3y,
                                downside: metricsData.downside_capture_3y
                            },
                            '5Y': {
                                upside: metricsData.upside_capture,
                                downside: metricsData.downside_capture
                            }
                        }
                    });
                }
            } catch (err) {
                setError(err.message || 'Failed to load fund details');
            } finally {
                setLoading(false);
            }
        };

        if (schemeCode) {
            loadFundData();
        }
    }, [schemeCode]);

    // Benchmark loader: re-fetches when selected benchmark changes
    useEffect(() => {
        const loadBenchmark = async () => {
            if (!selectedBenchmarkCode) return;
            try {
                const benchData = await fetchFundDetails(selectedBenchmarkCode);
                setBenchmark(benchData);
            } catch (err) {
                console.warn('Failed to load benchmark data:', err);
            }
        };

        loadBenchmark();
    }, [selectedBenchmarkCode]);

    const fundCategory = details?.meta?.scheme_category || '';
    const isDebt = isDebtCategory(fundCategory);
    const benchmarkInfo = useMemo(() => {
        if (selectedBenchmarkCode === DEBT_BENCHMARK_CODE) {
            return {
                code: DEBT_BENCHMARK_CODE,
                name: '10Y Sovereign G-Sec Benchmark',
                shortName: '10Y G-Sec Index',
                fullName: 'CRISIL 10-Yr Constant Maturity Gilt Index',
                assetClass: 'Debt'
            };
        }
        return {
            code: EQUITY_BENCHMARK_CODE,
            name: 'Nifty 50 TRI Benchmark',
            shortName: 'Nifty 50 Index',
            fullName: 'NSE Nifty 50 Total Return Index',
            assetClass: 'Equity'
        };
    }, [selectedBenchmarkCode]);

    const filterAndNormalizeData = (fundRaw, benchmarkRaw) => {
        if (!fundRaw || !Array.isArray(fundRaw) || fundRaw.length === 0) {
            return { chartData: [], summary: null, emptyReason: 'no_raw_data' };
        }

        const now = new Date();
        let cutoffDate = new Date();

        switch (timeRange) {
            case '1M': cutoffDate.setMonth(now.getMonth() - 1); break;
            case '6M': cutoffDate.setMonth(now.getMonth() - 6); break;
            case '1Y': cutoffDate.setFullYear(now.getFullYear() - 1); break;
            case '3Y': cutoffDate.setFullYear(now.getFullYear() - 3); break;
            case '5Y': cutoffDate.setFullYear(now.getFullYear() - 5); break;
            case 'ALL': cutoffDate = new Date(0); break;
            default: cutoffDate.setFullYear(now.getFullYear() - 1);
        }

        const parseDate = (d) => {
            if (!d) return new Date();
            const parts = d.split('-');
            if (parts.length === 3) {
                return new Date(Number(parts[2]), Number(parts[1]) - 1, Number(parts[0]));
            }
            return new Date(d);
        };

        // Filter and reverse to chronological order (oldest first)
        const fundInRange = fundRaw
            .filter(item => parseDate(item.date) >= cutoffDate)
            .slice()
            .reverse();

        if (fundInRange.length === 0) {
            return { 
                chartData: [], 
                summary: null, 
                emptyReason: 'no_data_in_range',
                availableCount: fundRaw.length,
                latestAvailableDate: fundRaw[0]?.date
            };
        }

        if (fundInRange.length < 2) {
            return {
                chartData: [],
                summary: null,
                emptyReason: 'insufficient_points',
                availableCount: fundRaw.length,
                latestAvailableDate: fundRaw[0]?.date
            };
        }

        // Map benchmark by exact date for O(1) alignment
        const benchMap = new Map();
        if (benchmarkRaw && Array.isArray(benchmarkRaw)) {
            benchmarkRaw.forEach(b => {
                const nav = parseFloat(b.nav);
                if (!isNaN(nav) && nav > 0) {
                    benchMap.set(b.date, nav);
                }
            });
        }

        const startFundNav = parseFloat(fundInRange[0].nav);
        if (isNaN(startFundNav) || startFundNav <= 0) {
            return { chartData: [], summary: null, emptyReason: 'invalid_start_nav' };
        }

        // Find starting benchmark NAV aligned to fund's start date
        let startBenchNav = 0;
        let lastBenchNav = 0;
        for (let i = 0; i < fundInRange.length; i++) {
            const bNav = benchMap.get(fundInRange[i].date);
            if (bNav) {
                startBenchNav = bNav;
                lastBenchNav = bNav;
                break;
            }
        }

        // Fallback: If exact date match is not found in range, find closest benchmark date
        if (startBenchNav === 0 && benchmarkRaw && benchmarkRaw.length > 0) {
            const targetTime = parseDate(fundInRange[0].date).getTime();
            let minDiff = Infinity;
            let closestNav = 0;
            for (const b of benchmarkRaw) {
                const bNav = parseFloat(b.nav);
                if (!isNaN(bNav) && bNav > 0) {
                    const diff = Math.abs(parseDate(b.date).getTime() - targetTime);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestNav = bNav;
                    }
                }
            }
            if (closestNav > 0) {
                startBenchNav = closestNav;
                lastBenchNav = closestNav;
            }
        }

        const startTime = parseDate(fundInRange[0].date).getTime();
        const endTime = parseDate(fundInRange[fundInRange.length - 1].date).getTime();
        const totalYears = Math.max((endTime - startTime) / (365.25 * 24 * 60 * 60 * 1000), 0.05);

        let finalFundGrowth = 0;
        let finalBenchGrowth = 0;

        // Build normalized percentage growth series (Start Date = exactly 0.00%)
        const chartData = fundInRange.map((item) => {
            const currentFundNav = parseFloat(item.nav);
            const fundGrowth = ((currentFundNav - startFundNav) / startFundNav) * 100;
            finalFundGrowth = fundGrowth;

            // Carry forward benchmark NAV if exact date has weekend/holiday mismatch
            const matchedBenchNav = benchMap.get(item.date);
            if (matchedBenchNav) {
                lastBenchNav = matchedBenchNav;
            }
            const currentBenchNav = lastBenchNav > 0 ? lastBenchNav : startBenchNav;
            const indexGrowth = startBenchNav > 0 
                ? parseFloat((((currentBenchNav - startBenchNav) / startBenchNav) * 100).toFixed(2))
                : null;
            if (indexGrowth !== null) {
                finalBenchGrowth = indexGrowth;
            }

            return {
                date: item.date,
                fund: parseFloat(fundGrowth.toFixed(2)),
                index: indexGrowth,
                fundNav: currentFundNav.toFixed(2)
            };
        });

        const hasValidBenchmark = startBenchNav > 0;

        // Annualized CAGR for periods of 1 year or more
        const fundCAGR = totalYears >= 1 
            ? ((Math.pow(1 + finalFundGrowth / 100, 1 / totalYears) - 1) * 100).toFixed(2)
            : null;
        const benchCAGR = totalYears >= 1 && hasValidBenchmark
            ? ((Math.pow(1 + finalBenchGrowth / 100, 1 / totalYears) - 1) * 100).toFixed(2)
            : null;

        const summary = {
            fundTotalReturn: finalFundGrowth.toFixed(2),
            benchTotalReturn: hasValidBenchmark ? finalBenchGrowth.toFixed(2) : null,
            fundCAGR,
            benchCAGR,
            outperformance: hasValidBenchmark ? (finalFundGrowth - finalBenchGrowth).toFixed(2) : null,
            startNav: startFundNav.toFixed(2),
            currentNav: parseFloat(fundInRange[fundInRange.length - 1].nav).toFixed(2),
            totalYears: totalYears.toFixed(1),
            hasValidBenchmark
        };

        return { chartData, summary, emptyReason: null };
    };

    if (loading) {
        return (
            <div className="fund-detail animate-fade-in max-w-6xl mx-auto pb-16">
                <button onClick={onBack} className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2 font-medium cursor-pointer">
                    ← Back to Fund Universe
                </button>
                <div className="card mb-6 animate-pulse p-6">
                    <div className="h-7 w-2/3 bg-slate-200 rounded-lg mb-4" />
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[1, 2, 3, 4].map(i => (
                            <div key={i} className="h-12 bg-slate-100 rounded-lg" />
                        ))}
                    </div>
                </div>
                <div className="card p-8 h-80 flex flex-col items-center justify-center gap-3">
                    <div className="w-8 h-8 rounded-full border-2 border-finance-primary border-t-transparent animate-spin" />
                    <p className="text-sm text-slate-500 font-medium">Loading fund metrics and historical performance...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="fund-detail animate-fade-in max-w-6xl mx-auto pb-16">
                <button onClick={onBack} className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2 font-medium cursor-pointer">
                    ← Back to Fund Universe
                </button>
                <div className="card p-8 text-center flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center font-bold text-xl">!</div>
                    <p className="text-slate-800 font-medium">{error}</p>
                    <button 
                        onClick={() => window.location.reload()} 
                        className="px-4 py-2 bg-finance-primary text-white text-xs font-semibold rounded-xl hover:bg-finance-primary-dark transition-colors shadow-sm cursor-pointer mt-2"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    if (!details) return null;

    const { chartData, summary, emptyReason, availableCount, latestAvailableDate } = filterAndNormalizeData(details.data, benchmark?.data);
    const latestNav = details?.meta?.last_nav ?? details?.data?.[0]?.nav ?? null;
    const latestNavDate = details?.meta?.last_nav_date ?? details?.data?.[0]?.date ?? null;

    const has5YRiskData = Boolean(
        stats?.risk?.['5Y'] &&
        (stats.risk['5Y'].sharpe !== null && stats.risk['5Y'].sharpe !== undefined)
    );

    return (
        <div className="fund-detail animate-fade-in max-w-6xl mx-auto pb-16">
            <button onClick={onBack} className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2 font-medium cursor-pointer">
                ← Back to Fund Universe
            </button>

            {/* Fund Header Card with Prominent NAV */}
            <div className="card mb-6">
                <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-5 pb-5 border-b border-slate-100">
                    <div className="flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                                isDebt 
                                    ? 'bg-amber-50 text-amber-700 border-amber-200' 
                                    : 'bg-blue-50 text-blue-700 border-blue-200'
                            }`}>
                                {details.meta.scheme_category || 'Mutual Fund'}
                            </span>
                            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                                Direct Growth
                            </span>
                            <span className="text-xs text-slate-400">Scheme #{details.meta.scheme_code}</span>
                        </div>
                        <h2 className="text-2xl md:text-3xl font-extrabold text-slate-900 tracking-tight leading-tight">
                            {details.meta.scheme_name}
                        </h2>
                        <p className="text-sm font-medium text-slate-500 mt-1">
                            {details.meta.fund_house}
                        </p>
                    </div>

                    {/* Prominent NAV Section */}
                    <div className="flex flex-col sm:items-end justify-center shrink-0 bg-slate-50 sm:bg-transparent p-4 sm:p-0 rounded-2xl border sm:border-0 border-slate-100">
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                            Current NAV
                        </span>
                        <div className="text-3xl md:text-4xl font-black text-slate-900 tracking-tight mt-0.5">
                            ₹{latestNav ? parseFloat(latestNav).toFixed(2) : '--'}
                        </div>
                        {latestNavDate && (
                            <span className="text-xs text-slate-500 mt-1 flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block"></span>
                                As of {latestNavDate}
                            </span>
                        )}
                    </div>
                </div>

                {/* Key Summary Metrics (Showing 3Y & 5Y) */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                    <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <span className="block text-[11px] font-semibold uppercase text-slate-400">Fund House</span>
                        <span className="font-bold text-slate-800 text-sm truncate block mt-0.5">{details.meta.fund_house}</span>
                    </div>
                    <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <span className="block text-[11px] font-semibold uppercase text-slate-400">Asset Category</span>
                        <span className="font-bold text-slate-800 text-sm truncate block mt-0.5">{details.meta.scheme_category}</span>
                    </div>
                    <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <span className="block text-[11px] font-semibold uppercase text-slate-400">Trailing CAGR</span>
                        <span className="font-bold text-emerald-600 text-sm block mt-0.5">
                            3Y: {stats?.returns?.['3Y'] ? `${stats.returns['3Y']}%` : 'N/A'}
                            {stats?.returns?.['5Y'] && <span className="text-slate-400 font-normal"> • </span>}
                            {stats?.returns?.['5Y'] && <span>5Y: {stats.returns['5Y']}%</span>}
                        </span>
                    </div>
                    <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <span className="block text-[11px] font-semibold uppercase text-slate-400">Risk-Adjusted Sharpe</span>
                        <span className="font-bold text-slate-800 text-sm block mt-0.5">
                            3Y: {stats?.risk?.['3Y']?.sharpe ?? 'N/A'}
                            {stats?.risk?.['5Y']?.sharpe !== undefined && stats?.risk?.['5Y']?.sharpe !== null && (
                                <span className="text-slate-400 font-normal"> • 5Y: {stats.risk['5Y'].sharpe}</span>
                            )}
                        </span>
                    </div>
                </div>
            </div>

            {/* Performance Comparison Chart */}
            <div className="card">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4">
                    <div>
                        <div className="flex items-center gap-2 flex-wrap">
                            <h3 className="text-xl font-bold text-slate-900">Performance Comparison (%)</h3>
                            {summary?.hasValidBenchmark && summary?.outperformance !== null && summary?.outperformance !== undefined && (
                                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                                    parseFloat(summary.outperformance) >= 0 
                                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                                        : 'bg-rose-50 text-rose-700 border border-rose-200'
                                }`}>
                                    {parseFloat(summary.outperformance) >= 0 
                                        ? `+${summary.outperformance}% vs Benchmark` 
                                        : `${summary.outperformance}% vs Benchmark`}
                                </span>
                            )}
                        </div>
                        <p className="text-xs text-slate-500 mt-0.5">
                            Cumulative Return from start of period • Benchmark: <strong className="text-slate-700 font-semibold">{benchmarkInfo.fullName}</strong>
                            {summary?.fundCAGR && ` • Annualized: ${summary.fundCAGR}% CAGR`}
                            {summary?.benchCAGR && ` • Benchmark: ${summary.benchCAGR}% CAGR`}
                        </p>
                    </div>

                    {/* Time Range Selector */}
                    <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 rounded-xl border border-slate-200/60">
                        {['1M', '6M', '1Y', '3Y', '5Y', 'ALL'].map(range => (
                            <button
                                key={range}
                                onClick={() => setTimeRange(range)}
                                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all cursor-pointer ${
                                    timeRange === range ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
                                }`}
                            >
                                {range}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Benchmark Selector / Indicator */}
                <div className="mb-5 pb-3 border-b border-slate-100 flex items-center justify-between gap-3 flex-wrap text-xs">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-slate-400 font-medium">Benchmark Index:</span>
                        <button
                            onClick={() => setSelectedBenchmarkCode(EQUITY_BENCHMARK_CODE)}
                            className={`px-3 py-1 rounded-xl font-semibold transition-all border cursor-pointer ${
                                selectedBenchmarkCode === EQUITY_BENCHMARK_CODE
                                    ? 'bg-finance-primary text-white border-finance-primary shadow-sm'
                                    : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                            }`}
                        >
                            Nifty 50 TRI (Equity)
                        </button>
                        <button
                            onClick={() => setSelectedBenchmarkCode(DEBT_BENCHMARK_CODE)}
                            className={`px-3 py-1 rounded-xl font-semibold transition-all border cursor-pointer ${
                                selectedBenchmarkCode === DEBT_BENCHMARK_CODE
                                    ? 'bg-finance-primary text-white border-finance-primary shadow-sm'
                                    : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                            }`}
                        >
                            10Y Sovereign G-Sec (Debt)
                        </button>
                        {isDebt && selectedBenchmarkCode === DEBT_BENCHMARK_CODE && (
                            <span className="text-[11px] font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-200 flex items-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
                                Asset-Matched Benchmark for Debt
                            </span>
                        )}
                        {!isDebt && selectedBenchmarkCode === EQUITY_BENCHMARK_CODE && (
                            <span className="text-[11px] font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-200 flex items-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                                Standard Equity Benchmark
                            </span>
                        )}
                    </div>
                    <span className="text-[11px] text-slate-400 hidden sm:inline">
                        Switch benchmarks to compare against equity or fixed income
                    </span>
                </div>

                <div className="h-[400px] w-full">
                    {chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis
                                    dataKey="date"
                                    stroke="#94a3b8"
                                    tick={{ fontSize: 10 }}
                                    minTickGap={60}
                                />
                                <YAxis
                                    stroke="#94a3b8"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(val) => `${val >= 0 ? '+' : ''}${val}%`}
                                />
                                <Tooltip
                                    contentStyle={{ 
                                        backgroundColor: '#ffffff', 
                                        borderColor: '#e2e8f0', 
                                        color: '#0f172a', 
                                        borderRadius: '12px', 
                                        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.08)' 
                                    }}
                                    itemStyle={{ fontSize: '12px' }}
                                    formatter={(value, name) => [`${value >= 0 ? '+' : ''}${value}%`, name]}
                                    labelFormatter={(label) => `Date: ${label}`}
                                />
                                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                <Line
                                    name={`${details.meta.scheme_name.split('-')[0].trim()} (This Fund)`}
                                    type="monotone"
                                    dataKey="fund"
                                    stroke="#2563eb"
                                    strokeWidth={2.5}
                                    dot={false}
                                    connectNulls={true}
                                    activeDot={{ r: 6 }}
                                />
                                {summary?.hasValidBenchmark && (
                                    <Line
                                        name={`${benchmarkInfo.shortName} (Benchmark)`}
                                        type="monotone"
                                        dataKey="index"
                                        stroke="#f59e0b"
                                        strokeWidth={2}
                                        dot={false}
                                        connectNulls={true}
                                        activeDot={{ r: 5 }}
                                    />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    ) : emptyReason === 'no_data_in_range' ? (
                        <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3 p-6 text-center">
                            <div className="w-12 h-12 rounded-2xl bg-amber-50 text-amber-600 flex items-center justify-center shadow-sm">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <div>
                                <h4 className="font-bold text-slate-800 text-sm">No NAV records in the selected {timeRange} window</h4>
                                <p className="text-xs text-slate-500 mt-1 max-w-md">
                                    This scheme has {availableCount} historical NAV records, with the latest recorded on <strong>{latestAvailableDate}</strong>. (It may have matured or been closed).
                                </p>
                            </div>
                            <button
                                onClick={() => setTimeRange('ALL')}
                                className="px-4 py-2 bg-finance-primary hover:bg-finance-primary-dark text-white text-xs font-semibold rounded-xl transition-all shadow-sm cursor-pointer"
                            >
                                Switch to 'ALL' to View Historical Track Record
                            </button>
                        </div>
                    ) : emptyReason === 'insufficient_points' ? (
                        <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-2 p-6 text-center">
                            <p className="font-semibold text-slate-800 text-sm">Insufficient data points</p>
                            <p className="text-xs text-slate-400 max-w-md">
                                At least 2 historical NAV points are required to compute percentage growth in this period.
                            </p>
                            <button
                                onClick={() => setTimeRange('ALL')}
                                className="mt-2 text-xs font-semibold text-finance-primary hover:underline cursor-pointer"
                            >
                                Switch to 'ALL' Range →
                            </button>
                        </div>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2">
                            <svg className="w-8 h-8 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                            <p className="text-sm">Historical comparison data is not currently available for this scheme.</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Key Statistics Section */}
            {stats && (
                <div className="grid md:grid-cols-2 gap-6 mt-6">
                    {/* Returns Table */}
                    <div className="card">
                        <h3 className="text-xl mb-4 text-finance-primary">Trailing Returns (CAGR)</h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-sm text-finance-text-primary">
                                <thead>
                                    <tr className="border-b border-finance-border">
                                        <th className="px-6 py-4 font-medium">Period</th>
                                        <th className="px-6 py-4 font-medium text-right">Returns</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3 px-6 font-medium">6 Months</td>
                                        <td className="py-3 px-6 text-right text-finance-success font-semibold">
                                            {stats.returns['6M'] !== null && stats.returns['6M'] !== undefined ? `${stats.returns['6M']}%` : 'N/A'}
                                        </td>
                                    </tr>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3 px-6 font-medium">1 Year</td>
                                        <td className="py-3 px-6 text-right text-finance-success font-semibold">
                                            {stats.returns['1Y'] !== null && stats.returns['1Y'] !== undefined ? `${stats.returns['1Y']}%` : 'N/A'}
                                        </td>
                                    </tr>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3 px-6 font-medium">3 Years</td>
                                        <td className="py-3 px-6 text-right text-finance-success font-semibold">
                                            {stats.returns['3Y'] !== null && stats.returns['3Y'] !== undefined ? `${stats.returns['3Y']}%` : 'N/A'}
                                        </td>
                                    </tr>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3 px-6 font-medium">5 Years</td>
                                        <td className="py-3 px-6 text-right text-finance-success font-semibold">
                                            {stats.returns['5Y'] !== null && stats.returns['5Y'] !== undefined ? `${stats.returns['5Y']}%` : 'N/A'}
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Risk Ratios Table */}
                    <div className="card">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <h3 className="text-xl text-finance-primary">Risk Measures</h3>
                                <p className="text-xs text-finance-text-secondary mt-1">Calculated over selected period</p>
                            </div>
                            <div className="flex gap-2 bg-slate-100 p-1 rounded-xl border border-slate-200/60">
                                <button
                                    onClick={() => setRiskPeriod('3Y')}
                                    className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all cursor-pointer ${
                                        riskPeriod === '3Y' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
                                    }`}
                                >
                                    3 Years
                                </button>
                                <button
                                    onClick={() => setRiskPeriod('5Y')}
                                    className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all cursor-pointer ${
                                        riskPeriod === '5Y' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
                                    }`}
                                >
                                    5 Years
                                </button>
                            </div>
                        </div>
                        
                        {riskPeriod === '5Y' && !has5YRiskData ? (
                            <div className="p-6 bg-slate-50 rounded-2xl border border-slate-200/80 text-center flex flex-col items-center gap-2">
                                <svg className="w-8 h-8 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <h4 className="font-bold text-slate-800 text-sm">5-Year Risk Metrics Unavailable</h4>
                                <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                                    This fund does not yet have 5 continuous years of AMFI daily trading history. 5Y Alpha, Beta, Sharpe, and Sortino ratios require a minimum of 5 years of daily NAV records.
                                </p>
                                <button
                                    onClick={() => setRiskPeriod('3Y')}
                                    className="mt-2 px-3.5 py-1.5 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-xl shadow-xs cursor-pointer"
                                >
                                    Switch to 3-Year Ratios →
                                </button>
                            </div>
                        ) : stats.risk && stats.risk[riskPeriod] ? (
                            <div className="grid grid-cols-2 gap-y-4 gap-x-8 text-sm">
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Alpha</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.risk[riskPeriod].alpha ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Beta</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.risk[riskPeriod].beta ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Sharpe Ratio</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.risk[riskPeriod].sharpe ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Std. Dev</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.risk[riskPeriod].stdDev ? `${stats.risk[riskPeriod].stdDev}%` : 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Sortino</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.risk[riskPeriod].sortino ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Upside Capture</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.capture?.[riskPeriod]?.upside ?? '--'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Downside Capture</span>
                                    <span className="text-finance-text-primary font-semibold">{stats.capture?.[riskPeriod]?.downside ?? '--'}</span>
                                </div>
                            </div>
                        ) : (
                            <div className="text-finance-text-secondary text-center py-8">Not enough data for {riskPeriod} risk analysis</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default FundDetail;
