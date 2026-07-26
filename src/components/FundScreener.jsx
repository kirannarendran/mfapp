import React, { useState, useEffect, useMemo } from 'react';
import { fetchScreenerFunds } from '../api';

const METRICS = [
    { id: 'cagr3y', label: '3Y Return', filterLabel: 'Min 3Y Return', dbKey: 'cagr_3y', min: -10, max: 40, step: 1, suffix: '%', type: 'min' },
    { id: 'cagr5y', label: '5Y Return', filterLabel: 'Min 5Y Return', dbKey: 'cagr_5y', min: -10, max: 40, step: 1, suffix: '%', type: 'min' },
    { id: 'beta3y', label: '3Y Beta', filterLabel: 'Max 3Y Beta', dbKey: 'beta_3y', min: 0.5, max: 2.0, step: 0.1, suffix: '', type: 'max' },
    { id: 'beta5y', label: '5Y Beta', filterLabel: 'Max 5Y Beta', dbKey: 'beta_5y', min: 0.5, max: 2.0, step: 0.1, suffix: '', type: 'max' },
    { id: 'sharpe3y', label: '3Y Sharpe Ratio', filterLabel: 'Min 3Y Sharpe Ratio', dbKey: 'sharpe_3y', min: -1, max: 3, step: 0.1, suffix: '', type: 'min' },
    { id: 'sharpe5y', label: '5Y Sharpe Ratio', filterLabel: 'Min 5Y Sharpe Ratio', dbKey: 'sharpe_5y', min: -1, max: 3, step: 0.1, suffix: '', type: 'min' },
    { id: 'sortino3y', label: '3Y Sortino Ratio', filterLabel: 'Min 3Y Sortino Ratio', dbKey: 'sortino_3y', min: -1, max: 5, step: 0.1, suffix: '', type: 'min' },
    { id: 'sortino5y', label: '5Y Sortino Ratio', filterLabel: 'Min 5Y Sortino Ratio', dbKey: 'sortino_5y', min: -1, max: 5, step: 0.1, suffix: '', type: 'min' },
    { id: 'sd3y', label: '3Y Std Dev', filterLabel: 'Max 3Y Std Dev', dbKey: 'std_dev_3y', min: 5, max: 40, step: 1, suffix: '%', type: 'max' },
    { id: 'sd5y', label: '5Y Std Dev', filterLabel: 'Max 5Y Std Dev', dbKey: 'std_dev_5y', min: 5, max: 40, step: 1, suffix: '%', type: 'max' },
    { id: 'alpha3y', label: '3Y Alpha', filterLabel: 'Min 3Y Alpha', dbKey: 'alpha_3y', min: -5, max: 15, step: 0.5, suffix: '', type: 'min' },
    { id: 'alpha5y', label: '5Y Alpha', filterLabel: 'Min 5Y Alpha', dbKey: 'alpha_5y', min: -5, max: 15, step: 0.5, suffix: '', type: 'min' },
    { id: 'upCap3y', label: '3Y Up Capture', filterLabel: 'Min 3Y Upside Capture', dbKey: 'upside_capture_3y', min: 0, max: 150, step: 5, suffix: '%', type: 'min' },
    { id: 'upCap5y', label: '5Y Up Capture', filterLabel: 'Min 5Y Upside Capture', dbKey: 'upside_capture_5y', min: 0, max: 150, step: 5, suffix: '%', type: 'min' },
    { id: 'downCap3y', label: '3Y Down Capture', filterLabel: 'Max 3Y Downside Capture', dbKey: 'downside_capture_3y', min: 0, max: 150, step: 5, suffix: '%', type: 'max' },
    { id: 'downCap5y', label: '5Y Down Capture', filterLabel: 'Max 5Y Downside Capture', dbKey: 'downside_capture_5y', min: 0, max: 150, step: 5, suffix: '%', type: 'max' },
    { id: 'mlRankingScore', label: 'ML Ranking Score — Experimental', filterLabel: 'Min ML Ranking Score', dbKey: 'ml_ranking_score', min: 0, max: 100, step: 1, suffix: '', type: 'min', tooltip: 'Experimental category-relative model ranking from 0 to 100. It indicates relative model ordering among eligible funds in the same category and prediction period. It is not a probability, expected return, recommendation, or guarantee of future performance.' },
];

const FundScreener = ({ onBack, onSelectFund }) => {
    const [selectedMetrics, setSelectedMetrics] = useState(['cagr3y', 'cagr5y', 'beta5y', 'sharpe5y', 'alpha5y']);


    const [filters, setFilters] = useState({
        minCagr3Y: 10,
        minCagr5Y: 10,
        maxBeta5y: 1.2,
        minSharpe5y: 0.3,
        minSortino5y: 0.5,
        maxSd5y: 25,
        minAlpha5y: -2.0,
        minUpCap5y: 80,
        maxDownCap5y: 110,
        maxBeta3y: 1.2,
        minSharpe3y: 0.3,
        minSortino3y: 0.5,
        maxSd3y: 25,
        minAlpha3y: -2.0,
        minUpCap3y: 80,
        maxDownCap3y: 110,
        minMlRankingScore: 50,
        category: 'Large Cap Fund'
    });

    const initialWeights = {};
    METRICS.forEach(m => initialWeights[m.id] = m.id === 'mlRankingScore' ? 0 : 5);
    const [metricWeights, setMetricWeights] = useState(initialWeights);

    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const categories = ['All', 'Large Cap Fund', 'Mid Cap Fund', 'Small Cap Fund', 'Flexi Cap Fund', 'Multi Cap Fund', 'ELSS', 'Others'];

    const handleFilterChange = (e) => {
        setFilters({ ...filters, [e.target.name]: e.target.value });
    };

    const handleWeightChange = (metricId, val) => {
        setMetricWeights({ ...metricWeights, [metricId]: parseFloat(val) });
    };

    const toggleMetric = (metricId) => {
        if (selectedMetrics.includes(metricId)) {
            if (selectedMetrics.length > 1) {
                setSelectedMetrics(selectedMetrics.filter(id => id !== metricId));
            }
        } else {
            setSelectedMetrics([...selectedMetrics, metricId]);
        }
    };

    const handleScreen = async () => {
        setLoading(true);
        try {
            const activeFilters = { category: filters.category, includeExperimental: 'false' };
            METRICS.forEach(m => {
                if (selectedMetrics.includes(m.id)) {
                    const key = m.type === 'min' ? `min${m.id.charAt(0).toUpperCase() + m.id.slice(1)}` : `max${m.id.charAt(0).toUpperCase() + m.id.slice(1)}`;
                    if (m.id === 'cagr3y') activeFilters.minCagr3Y = filters.minCagr3Y;
                    else if (m.id === 'cagr5y') activeFilters.minCagr5Y = filters.minCagr5Y;
                    else activeFilters[key] = filters[key];
                }
            });

            const data = await fetchScreenerFunds(activeFilters);
            setResults(data.funds || []);
        } catch (error) {
            console.error("Failed to screen funds", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        handleScreen();
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedMetrics]);

    const scoredResults = useMemo(() => {
        if (!results || results.length === 0) return [];
        const bounds = {};
        selectedMetrics.forEach(metricId => {
            const metricDef = METRICS.find(m => m.id === metricId);
            let min = Infinity;
            let max = -Infinity;
            results.forEach(fund => {
                const val = fund[metricDef.dbKey];
                if (val !== null && val !== undefined) {
                    if (val < min) min = val;
                    if (val > max) max = val;
                }
            });
            bounds[metricId] = { min, max };
        });

        const mapped = results.map(fund => {
            let score = 0;
            let totalWeight = 0;
            selectedMetrics.forEach(metricId => {
                const metricDef = METRICS.find(m => m.id === metricId);
                const val = fund[metricDef.dbKey];
                const weight = metricWeights[metricId] || 0;
                
                if (weight > 0 && val !== null && val !== undefined) {
                    const { min, max } = bounds[metricId];
                    if (max > min) {
                        let normalized = (val - min) / (max - min);
                        if (metricDef.type === 'max') {
                            normalized = 1 - normalized;
                        }
                        score += normalized * weight;
                    } else {
                        score += 0.5 * weight;
                    }
                    totalWeight += weight;
                }
            });
            const finalScore = totalWeight > 0 ? (score / totalWeight) * 100 : 0;
            return { ...fund, compositeScore: finalScore };
        });
        
        return mapped.sort((a, b) => b.compositeScore - a.compositeScore);
    }, [results, selectedMetrics, metricWeights]);

    useEffect(() => {
        const timer = setTimeout(() => {
            handleScreen();
        }, 400);
        return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [filters, selectedMetrics]);

    const getFilterKey = (metric) => {
        if (metric.id === 'cagr3y') return 'minCagr3Y';
        if (metric.id === 'cagr5y') return 'minCagr5Y';
        return metric.type === 'min' ? `min${metric.id.charAt(0).toUpperCase() + metric.id.slice(1)}` : `max${metric.id.charAt(0).toUpperCase() + metric.id.slice(1)}`;
    };

    const getMedal = (index) => {
        if (index === 0) return '🥇';
        if (index === 1) return '🥈';
        if (index === 2) return '🥉';
        return null;
    };

    return (
        <div className="animate-fade-in pb-20 w-full">
            <h2 className="text-2xl font-bold mb-8 text-finance-text-primary">
                AI Fund Screener
            </h2>

            <div className="flex flex-col gap-8">
                {/* Top Filters & Controls */}
                <div className="bg-white border border-slate-100 shadow-sm rounded-2xl p-6 md:p-8 space-y-8">
                    {/* Metrics Selection */}
                    <div>
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">Metrics Selection</h3>

                        </div>
                        <div className="flex flex-wrap gap-2">
                            {METRICS.filter(m => m.id !== 'mlRankingScore').map(m => (
                                <button
                                    key={m.id}
                                    onClick={() => toggleMetric(m.id)}
                                    title={m.tooltip}
                                    className={`px-4 py-2 text-sm rounded-lg font-medium transition-all ${
                                        selectedMetrics.includes(m.id) 
                                        ? 'bg-finance-primary text-white shadow-md shadow-finance-primary/20' 
                                        : 'bg-slate-50 border border-slate-200 text-slate-600 hover:bg-slate-100'
                                    }`}
                                >
                                    {m.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8 pt-8 border-t border-slate-100">
                        {/* Metric Weightages */}
                        <div>
                            <div className="flex items-center gap-2 mb-6">
                                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">Scoring Weights</h3>
                                <div className="relative group">
                                    <div className="w-4 h-4 rounded-full bg-slate-200 text-slate-500 flex items-center justify-center cursor-help text-[10px] font-bold select-none">?</div>
                                    <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-72 bg-slate-900 text-white text-xs rounded-xl p-3.5 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
                                        <p className="font-bold text-emerald-400 mb-1">📊 Ranking & Scoring</p>
                                        <p className="leading-relaxed text-slate-300">Funds that pass the strict filters are <span className="text-white font-semibold">ranked and scored</span> against each other using these weights. A higher weight means that metric matters more to your final score (0–100).</p>
                                        <p className="mt-2 text-slate-400 italic">e.g. Weight 9/10 on 5Y Return → funds with higher returns score higher.</p>
                                        <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-slate-900"></div>
                                    </div>
                                </div>
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-6">
                                {METRICS.filter(m => selectedMetrics.includes(m.id)).map(metric => (
                                    <div key={`weight-${metric.id}`}>
                                        <label className="flex justify-between items-center text-sm font-medium text-slate-700 mb-2">
                                            <span>{metric.label}</span>
                                            <span className="text-finance-primary font-bold bg-finance-primary/10 px-2 py-0.5 rounded text-xs">{metricWeights[metric.id]}/10</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="0" max="10" step="1"
                                            value={metricWeights[metric.id]}
                                            onChange={(e) => handleWeightChange(metric.id, e.target.value)}
                                            className="w-full accent-finance-primary"
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Filter Criteria */}
                        <div>
                            <div className="flex items-center gap-2 mb-6">
                                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">Strict Filter Criteria</h3>
                                <div className="relative group">
                                    <div className="w-4 h-4 rounded-full bg-slate-200 text-slate-500 flex items-center justify-center cursor-help text-[10px] font-bold select-none">?</div>
                                    <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-72 bg-slate-900 text-white text-xs rounded-xl p-3.5 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50">
                                        <p className="font-bold text-rose-400 mb-1">🚫 Hard Knockout Rules</p>
                                        <p className="leading-relaxed text-slate-300">These are <span className="text-white font-semibold">mandatory pass/fail gates</span>. A fund must meet every threshold to appear in results at all. Failing even one filter eliminates the fund entirely.</p>
                                        <p className="mt-2 text-slate-400 italic">e.g. Max Beta = 1.2 → any fund with Beta &gt; 1.2 is excluded, no matter how good its returns are.</p>
                                        <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-slate-900"></div>
                                    </div>
                                </div>
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-6">
                                {METRICS.filter(m => selectedMetrics.includes(m.id)).map(metric => {
                                    const filterKey = getFilterKey(metric);
                                    const isMin = metric.type === 'min';
                                    const colorClass = isMin ? 'text-emerald-700 bg-emerald-50' : 'text-rose-700 bg-rose-50';

                                    return (
                                        <div key={metric.id}>
                                            <label className="flex justify-between items-center text-sm font-medium text-slate-700 mb-2">
                                                <span className="truncate pr-2">{metric.filterLabel}</span>
                                                <span className={`font-semibold px-2 py-0.5 rounded text-xs shrink-0 ${colorClass}`}>{filters[filterKey]}{metric.suffix}</span>
                                            </label>
                                            <input
                                                type="range"
                                                name={filterKey}
                                                min={metric.min} max={metric.max} step={metric.step}
                                                value={filters[filterKey]}
                                                onChange={handleFilterChange}
                                                className="w-full accent-finance-primary"
                                            />
                                        </div>
                                    );
                                })}

                                <div className="col-span-1 sm:col-span-2 mt-2">
                                    <label className="block text-sm font-medium text-slate-700 mb-2">Category Filter</label>
                                    <select 
                                        name="category"
                                        value={filters.category}
                                        onChange={handleFilterChange}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-sm outline-none focus:border-finance-primary/50 focus:ring-4 focus:ring-finance-primary/10 transition-all"
                                    >
                                        {categories.map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="pt-6 border-t border-slate-100 flex justify-end">
                        <button 
                            onClick={handleScreen}
                            disabled={loading}
                            className="bg-finance-primary hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-xl transition-all shadow-[0_4px_12px_rgba(37,99,235,0.2)] hover:shadow-[0_6px_16px_rgba(37,99,235,0.3)] disabled:opacity-70 disabled:cursor-not-allowed flex items-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    Applying...
                                </>
                            ) : 'Apply Advanced Filters'}
                        </button>
                    </div>
                </div>

                {/* Results Area */}
                <div className="w-full">
                    <div className="card !p-0 overflow-hidden flex flex-col min-h-[500px]">
                        <div className="p-4 border-b border-finance-border bg-finance-surface">
                            <h3 className="text-base font-semibold text-finance-text-primary">Screener Results ({scoredResults.length})</h3>
                        </div>
                        
                        {loading ? (
                            <div className="flex-1 flex justify-center items-center text-finance-primary">Loading...</div>
                        ) : scoredResults.length === 0 ? (
                            <div className="flex-1 flex flex-col justify-center items-center text-finance-text-secondary">
                                <p>No funds match these criteria.</p>
                                <p className="text-sm mt-1">Try relaxing your risk or return expectations.</p>
                            </div>
                        ) : (
                            <div className="overflow-x-auto">
                                <table className="w-full text-left border-collapse text-sm">
                                    <thead>
                                        <tr>
                                            <th className="font-medium text-finance-text-primary min-w-[250px]">Fund Name</th>
                                            <th className="font-medium text-finance-text-primary text-center leading-tight">
                                                Score<br/><span className="text-[10px] text-finance-text-secondary font-normal">(0-100)</span>
                                            </th>
                                            {METRICS.filter(m => selectedMetrics.includes(m.id)).map(metric => (
                                                <th key={metric.id} className="font-medium text-finance-text-primary text-right whitespace-nowrap">
                                                    {metric.label}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {scoredResults.map((fund, idx) => {
                                            const medal = getMedal(idx);
                                            const isWinner = idx < 3;
                                            
                                            return (
                                                <tr key={fund.scheme_code} 
                                                    className={`cursor-pointer hover:bg-[#F8FAFD] ${isWinner ? 'border-l-[3px] border-l-finance-primary' : 'border-l-[3px] border-l-transparent'}`}
                                                    onClick={() => onSelectFund(fund.scheme_code)}>
                                                    <td className="p-4">
                                                        <div className="flex gap-2">
                                                            <div className="w-6 shrink-0 text-center">{medal && <span className="text-lg">{medal}</span>}</div>
                                                            <div>
                                                                <div className="font-semibold text-finance-text-primary">{fund.scheme_name}</div>
                                                                <div className="text-[13px] text-finance-text-secondary mt-0.5">{fund.category}</div>
                                                            </div>
                                                        </div>
                                                    </td>
                                                    
                                                    <td className="p-4 text-center">
                                                        <span className="inline-block bg-finance-primary-soft px-2.5 py-1 rounded-md text-finance-primary font-medium border border-finance-primary/30">
                                                            {fund.compositeScore.toFixed(1)}
                                                        </span>
                                                    </td>

                                                    {METRICS.filter(m => selectedMetrics.includes(m.id)).map(metric => {
                                                        const val = fund[metric.dbKey];
                                                        const displayVal = val !== null && val !== undefined ? `${Number(val.toFixed(2))}${metric.suffix}` : '--';
                                                        
                                                        let colorClass = 'text-finance-text-primary';
                                                        if (metric.id.includes('cagr') && val > 0) colorClass = 'text-finance-positive font-medium';
                                                        else if (metric.id.includes('cagr') && val < 0) colorClass = 'text-finance-negative font-medium';
                                                        else if (metric.id === 'alpha' && val > 0) colorClass = 'text-finance-positive font-medium';
                                                        else if (metric.id === 'alpha' && val < 0) colorClass = 'text-finance-negative font-medium';
                                                        else if (metric.id === 'upCap' && val > 100) colorClass = 'text-finance-positive font-medium';
                                                        else if (metric.id === 'downCap' && val > 100) colorClass = 'text-finance-negative font-medium';

                                                        return (
                                                            <td key={metric.id} className={`p-4 text-right ${colorClass}`}>
                                                                {displayVal}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FundScreener;
