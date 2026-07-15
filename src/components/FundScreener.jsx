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
    const [showExperimental, setShowExperimental] = useState(false);

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
            const activeFilters = { category: filters.category, includeExperimental: showExperimental ? 'true' : 'false' };
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
    }, [selectedMetrics, showExperimental]);

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

            <div className="grid lg:grid-cols-[260px_1fr] gap-8">
                {/* Sidebar Filters */}
                <div className="lg:sticky lg:top-0 lg:h-[calc(100vh-8rem)]">
                    <div className="card h-full flex flex-col p-0 overflow-hidden">
                        <div className="flex-1 overflow-y-auto p-6 space-y-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-xs font-bold text-finance-text-secondary uppercase tracking-wider m-0">Metrics</h3>
                                <label className="flex items-center space-x-2 text-xs font-medium text-finance-text-secondary cursor-pointer">
                                    <input type="checkbox" checked={showExperimental} onChange={(e) => setShowExperimental(e.target.checked)} className="form-checkbox h-3 w-3 text-finance-primary rounded border-finance-border bg-finance-surface focus:ring-finance-primary" />
                                    <span>Show Experimental</span>
                                </label>
                            </div>
                            <div className="flex flex-wrap gap-2 mb-6 border-b border-finance-border pb-6">
                                {METRICS.filter(m => showExperimental || m.id !== 'mlRankingScore').map(m => (
                                    <button
                                        key={m.id}
                                        onClick={() => toggleMetric(m.id)}
                                        title={m.tooltip}
                                        className={`px-3 py-1 text-xs rounded font-medium transition-all ${
                                            selectedMetrics.includes(m.id) 
                                            ? 'chip-selected' 
                                            : 'chip-unselected'
                                        }`}
                                    >
                                        {m.label}
                                    </button>
                                ))}
                            </div>

                            <h3 className="text-xs font-bold text-finance-text-secondary uppercase tracking-wider mb-4">Metric Weightages</h3>
                            <div className="space-y-5 mb-6 border-b border-finance-border pb-6">
                                {METRICS.filter(m => selectedMetrics.includes(m.id)).map(metric => (
                                    <div key={`weight-${metric.id}`}>
                                        <label className="block text-xs font-medium text-finance-text-primary mb-1 flex justify-between">
                                            <span>{metric.label} Importance</span>
                                            <span className="text-finance-primary font-bold">{metricWeights[metric.id]}/10</span>
                                        </label>
                                        <input
                                            type="range"
                                            min="0" max="10" step="1"
                                            value={metricWeights[metric.id]}
                                            onChange={(e) => handleWeightChange(metric.id, e.target.value)}
                                            className="w-full"
                                        />
                                    </div>
                                ))}
                            </div>

                            <h3 className="text-xs font-bold text-finance-text-secondary uppercase tracking-wider mb-4">Filter Criteria</h3>
                            <div className="space-y-5">
                                {METRICS.filter(m => selectedMetrics.includes(m.id)).map(metric => {
                                    const filterKey = getFilterKey(metric);
                                    const isMin = metric.type === 'min';
                                    const colorClass = isMin ? 'text-finance-positive' : 'text-finance-negative';

                                    return (
                                        <div key={metric.id}>
                                            <label className="block text-xs font-medium text-finance-text-primary mb-1 flex justify-between">
                                                <span>{metric.filterLabel}</span>
                                                <span className={`font-semibold ${colorClass}`}>{filters[filterKey]}{metric.suffix}</span>
                                            </label>
                                            <input
                                                type="range"
                                                name={filterKey}
                                                min={metric.min} max={metric.max} step={metric.step}
                                                value={filters[filterKey]}
                                                onChange={handleFilterChange}
                                                className="w-full"
                                            />
                                        </div>
                                    );
                                })}

                                <div className="pt-4 border-t border-finance-border">
                                    <label className="block text-xs font-medium text-finance-text-primary mb-2">Category</label>
                                    <select 
                                        name="category"
                                        value={filters.category}
                                        onChange={handleFilterChange}
                                    >
                                        {categories.map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* Sticky Apply Button */}
                        <div className="p-4 bg-finance-surface border-t border-finance-border shrink-0">
                            <button 
                                onClick={handleScreen}
                                disabled={loading}
                                className="w-full bg-finance-primary hover:bg-blue-700 text-white font-medium py-2 rounded-lg transition-all shadow-lg shadow-finance-primary/20"
                            >
                                {loading ? 'Applying...' : 'Apply Filters'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Results Area */}
                <div className="">
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
