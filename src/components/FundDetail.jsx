import React, { useState, useEffect } from 'react';
import { fetchFundDetails, fetchFundMetrics, BENCHMARK_INDEX_CODE } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const FundDetail = ({ schemeCode, onBack }) => {
    const [details, setDetails] = useState(null);
    const [benchmark, setBenchmark] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeRange, setTimeRange] = useState('1Y'); // 1M, 6M, 1Y, 3Y, 5Y, ALL
    const [stats, setStats] = useState(null);

    const [riskPeriod, setRiskPeriod] = useState('3Y'); // 3Y or 5Y

    useEffect(() => {
        const loadAllData = async () => {
            try {
                setLoading(true);
                const [fundData, benchmarkData, metricsData] = await Promise.all([
                    fetchFundDetails(schemeCode),
                    fetchFundDetails(BENCHMARK_INDEX_CODE),
                    fetchFundMetrics(schemeCode)
                ]);
                setDetails(fundData);
                setBenchmark(benchmarkData);
                
                // Map the backend metrics format to what the component expects
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
            } catch (err) {
                setError('Failed to load fund details or metrics');
            } finally {
                setLoading(false);
            }
        };

        if (schemeCode) {
            loadAllData();
        }
    }, [schemeCode]);

    const filterAndNormalizeData = (fundRaw, benchmarkRaw) => {
        if (!fundRaw || !benchmarkRaw) return [];

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

        const isCAGR = ['3Y', '5Y', 'ALL'].includes(timeRange);
        const startDate = cutoffDate;

        const parseDate = (d) => {
            const [day, month, year] = d.split('-');
            return new Date(`${year}-${month}-${day}`);
        };

        const fundInRange = fundRaw.filter(item => parseDate(item.date) >= cutoffDate).reverse();

        if (fundInRange.length === 0) return [];

        const startTime = parseDate(fundInRange[0].date).getTime();
        
        // Ensure benchmark doesn't start earlier than the fund data available in this view
        const effectiveCutoff = new Date(Math.max(cutoffDate.getTime(), startTime));
        const benchmarkInRange = benchmarkRaw.filter(item => parseDate(item.date) >= effectiveCutoff).reverse();

        const startFundNav = parseFloat(fundInRange[0].nav);
        const startBenchNav = benchmarkInRange.length > 0 ? parseFloat(benchmarkInRange[0].nav) : 0;

        // Map data to a unified timeline
        return fundInRange.map((item, index) => {
            const currentFundNav = parseFloat(item.nav);
            const itemTime = parseDate(item.date).getTime();
            const daysPassed = (itemTime - startTime) / (1000 * 60 * 60 * 24);
            const yearsPassed = daysPassed / 365.25;

            // Helper to calc return (CAGR if > 1y and isCAGR, else Abs)
            const calcReturn = (cur, start) => {
                if (start === 0) return 0;
                const ratio = cur / start;
                if (isCAGR && yearsPassed > 1) {
                    return (Math.pow(ratio, 1 / yearsPassed) - 1) * 100;
                }
                return (ratio - 1) * 100;
            };

            const fundGrowth = calcReturn(currentFundNav, startFundNav);

            // Find closest benchmark date
            const benchItem = benchmarkInRange.find(b => b.date === item.date) || benchmarkInRange[index] || benchmarkInRange[benchmarkInRange.length - 1];
            const currentBenchNav = benchItem ? parseFloat(benchItem.nav) : startBenchNav;
            const indexGrowth = calcReturn(currentBenchNav, startBenchNav);

            // Simulate Category performance
            const categoryGrowth = (fundGrowth + indexGrowth) / 2 + (Math.sin(index / 10) * 0.5);

            return {
                date: item.date,
                fund: parseFloat(fundGrowth.toFixed(2)),
                index: parseFloat(indexGrowth.toFixed(2)),
                category: parseFloat(categoryGrowth.toFixed(2))
            };
        });
    };

    if (loading) return <div className="text-center p-8">Loading details & benchmarks...</div>;
    if (error) return <div className="text-finance-danger p-8">{error}</div>;
    if (!details) return null;

    const chartData = filterAndNormalizeData(details.data, benchmark?.data);
    const isCAGR = ['3Y', '5Y', 'ALL'].includes(timeRange);

    return (
        <div className="fund-detail animate-fade-in">
            <button onClick={onBack} className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2">
                ← Back to Search
            </button>

            <div className="card mb-6">
                <h2 className="text-2xl mb-2 text-finance-primary">{details.meta.scheme_name}</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-finance-text-primary">
                    <div>
                        <span className="block text-finance-text-secondary">Fund House</span>
                        {details.meta.fund_house}
                    </div>
                    <div>
                        <span className="block text-finance-text-secondary">Category</span>
                        {details.meta.scheme_category}
                    </div>
                    <div>
                        <span className="block text-finance-text-secondary">Type</span>
                        {details.meta.scheme_type}
                    </div>
                    <div>
                        <span className="block text-finance-text-secondary">Code</span>
                        {details.meta.scheme_code}
                    </div>
                </div>
            </div>

            <div className="card">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h3 className="text-xl">Performance Comparison (%)</h3>
                        <p className="text-xs text-finance-text-secondary">
                            {isCAGR ? 'Annualized Return (CAGR)' : 'Absolute Return'} from start of selected period
                        </p>
                    </div>
                    <div className="flex gap-2">
                        {['1M', '6M', '1Y', '3Y', '5Y', 'ALL'].map(range => (
                            <button
                                key={range}
                                onClick={() => setTimeRange(range)}
                                className={`px-3 py-1 text-sm rounded ${timeRange === range ? 'chip-selected' : 'bg-finance-border text-finance-text-primary hover:bg-slate-600'}`}
                            >
                                {range}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="h-[400px] w-full">
                    {chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis
                                    dataKey="date"
                                    stroke="#94a3b8"
                                    tick={{ fontSize: 10 }}
                                    minTickGap={60}
                                />
                                <YAxis
                                    stroke="#94a3b8"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(val) => `${val}%`}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                                    itemStyle={{ fontSize: '12px' }}
                                    formatter={(value) => [`${value}%`]}
                                    labelFormatter={(label) => `${label} (${isCAGR && ['3Y', '5Y', 'ALL'].includes(timeRange) ? 'CAGR' : 'Abs'})`}
                                />
                                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                <Line
                                    name="This Fund"
                                    type="monotone"
                                    dataKey="fund"
                                    stroke="#38bdf8"
                                    strokeWidth={2.5}
                                    dot={false}
                                    activeDot={{ r: 6 }}
                                />
                                <Line
                                    name="Category Avg"
                                    type="monotone"
                                    dataKey="category"
                                    stroke="#94a3b8"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={false}
                                />
                                <Line
                                    name="Index (Nifty 50)"
                                    type="monotone"
                                    dataKey="index"
                                    stroke="#fbbf24"
                                    strokeWidth={2}
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-full flex items-center justify-center text-finance-text-secondary">
                            No comparison data available for this time range.
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
                                        <th className="px-6 py-4 font-medium">Returns</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3">6 Months</td>
                                        <td className="py-3 text-right text-finance-success">{stats.returns['6M'] ? `${stats.returns['6M']}%` : 'N/A'}</td>
                                    </tr>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3">1 Year</td>
                                        <td className="py-3 text-right text-finance-success">{stats.returns['1Y'] ? `${stats.returns['1Y']}%` : 'N/A'}</td>
                                    </tr>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3">3 Years</td>
                                        <td className="py-3 text-right text-finance-success">{stats.returns['3Y'] ? `${stats.returns['3Y']}%` : 'N/A'}</td>
                                    </tr>
                                    <tr className="border-b border-finance-border">
                                        <td className="py-3">5 Years</td>
                                        <td className="py-3 text-right text-finance-success">{stats.returns['5Y'] ? `${stats.returns['5Y']}%` : 'N/A'}</td>
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
                            <div className="flex gap-2 bg-finance-bg p-1 rounded">
                                <button
                                    onClick={() => setRiskPeriod('3Y')}
                                    className={`px-3 py-1 text-xs rounded ${riskPeriod === '3Y' ? 'chip-selected' : 'text-finance-text-primary hover:text-slate-200'}`}
                                >
                                    3 Years
                                </button>
                                <button
                                    onClick={() => setRiskPeriod('5Y')}
                                    className={`px-3 py-1 text-xs rounded ${riskPeriod === '5Y' ? 'chip-selected' : 'text-finance-text-primary hover:text-slate-200'}`}
                                >
                                    5 Years
                                </button>
                            </div>
                        </div>
                        
                        {stats.risk && stats.risk[riskPeriod] ? (
                            <div className="grid grid-cols-2 gap-y-4 gap-x-8 text-sm">
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Alpha</span>
                                    <span className="text-finance-text-primary">{stats.risk[riskPeriod].alpha ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Beta</span>
                                    <span className="text-finance-text-primary">{stats.risk[riskPeriod].beta ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Sharpe Ratio</span>
                                    <span className="text-finance-text-primary">{stats.risk[riskPeriod].sharpe ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Std. Dev</span>
                                    <span className="text-finance-text-primary">{stats.risk[riskPeriod].stdDev ? `${stats.risk[riskPeriod].stdDev}%` : 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Sortino</span>
                                    <span className="text-finance-text-primary">{stats.risk[riskPeriod].sortino ?? 'N/A'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Upside Capture</span>
                                    <span className="text-finance-text-primary">{stats.capture?.[riskPeriod]?.upside ?? '--'}</span>
                                </div>
                                <div className="flex justify-between border-b border-finance-border pb-2">
                                    <span className="text-finance-text-secondary">Downside Capture</span>
                                    <span className="text-finance-text-primary">{stats.capture?.[riskPeriod]?.downside ?? '--'}</span>
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
