import React, { useState, useEffect } from 'react';
import { fetchComparison } from '../api';

const ComparisonView = ({ funds, onBack }) => {
    const [comparisonData, setComparisonData] = useState(null);
    const [loading, setLoading] = useState(true);

    const [riskPeriod, setRiskPeriod] = useState('3Y'); // 3Y or 5Y

    useEffect(() => {
        const fetchAllData = async () => {
            setLoading(true);
            try {
                const schemeCodes = funds.map(f => f.schemeCode);
                const results = await fetchComparison(schemeCodes);
                setComparisonData(results);
            } catch (err) {
                console.error("Comparison fetch error:", err);
            } finally {
                setLoading(false);
            }
        };

        if (funds.length > 0) {
            fetchAllData();
        }
    }, [funds]);

    if (loading) return <div className="text-center p-8 text-finance-primary">Loading comparison data...</div>;

    return (
        <div className="animate-fade-in">
            <button onClick={onBack} className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2">
                ← Back to List
            </button>
            <div className="card overflow-x-auto !p-0">
                <div className="p-4 border-b border-finance-border bg-finance-surface">
                    <h2 className="text-lg font-semibold text-finance-text-primary">Fund Comparison</h2>
                </div>

                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr>
                            <th className="px-6 py-4 font-medium">Metric</th>
                            {comparisonData.map(fund => (
                                <th key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary font-semibold align-top min-w-[200px]">
                                    {fund.schemeName}
                                    <div className="text-xs text-finance-text-secondary font-normal mt-1">{fund.schemeCode}</div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="text-sm">
                        {/* Returns Section */}
                        <tr className="bg-finance-bg">
                            <td colSpan={funds.length + 1} className="p-2 pl-4 text-xs font-bold text-finance-primary uppercase tracking-wider">
                                Trailing Returns (CAGR)
                            </td>
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">6 Months (Abs)</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.returns['6M'] ? `${fund.returns['6M']}%` : 'N/A'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">1 Year</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.returns['1Y'] ? `${fund.returns['1Y']}%` : 'N/A'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">3 Years</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.returns['3Y'] ? `${fund.returns['3Y']}%` : 'N/A'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">5 Years</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.returns['5Y'] ? `${fund.returns['5Y']}%` : 'N/A'}</td>
                            ))}
                        </tr>

                        {/* Risk Section */}
                        <tr className="bg-finance-bg">
                            <td className="p-2 pl-4 text-xs font-bold text-finance-primary uppercase tracking-wider flex items-center gap-4">
                                <span>Risk Measures vs Nifty 50</span>
                                <div className="flex gap-1 bg-finance-bg p-0.5 rounded">
                                    <button
                                        onClick={() => setRiskPeriod('3Y')}
                                        className={`px-3 py-1 text-xs rounded font-medium transition-all ${riskPeriod === '3Y' ? 'chip-selected' : 'chip-unselected'}`}
                                    >
                                        3Y
                                    </button>
                                    <button
                                        onClick={() => setRiskPeriod('5Y')}
                                        className={`px-3 py-1 text-xs rounded font-medium transition-all ${riskPeriod === '5Y' ? 'chip-selected' : 'chip-unselected'}`}
                                    >
                                        5Y
                                    </button>
                                </div>
                            </td>
                            <td colSpan={funds.length} className="bg-finance-bg border-b border-finance-border"></td>
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">Alpha</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-success">{fund.risk?.[riskPeriod]?.alpha ?? '--'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">Beta</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.risk?.[riskPeriod]?.beta ?? '--'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">Sharpe Ratio</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.risk?.[riskPeriod]?.sharpe ?? '--'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">Sortino Ratio</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.risk?.[riskPeriod]?.sortino ?? '--'}</td>
                            ))}
                        </tr>
                        <tr>
                            <td className="p-4 border-b border-finance-border text-finance-text-primary">Std. Deviation</td>
                            {comparisonData.map(fund => (
                                <td key={fund.schemeCode} className="p-4 border-b border-finance-border text-finance-text-primary">{fund.risk?.[riskPeriod]?.stdDev ? `${fund.risk[riskPeriod].stdDev} %` : '--'}</td>
                            ))}
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default ComparisonView;
