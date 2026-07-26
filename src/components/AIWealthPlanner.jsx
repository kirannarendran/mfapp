import React, { useState, useRef, useEffect } from 'react';
import { calculateSIPFutureValue } from '../utils/financialPlannerUtils';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';

const AIWealthPlanner = ({ onBack }) => {
    // ── STATE ───────────────────────────────────────────────────────────────
    const [inputs, setInputs] = useState({
        monthlyInvestment: 5000,
        expectedReturnRate: 12,
        years: 20,
        fundCategory: 'Any Equity',
        maxDrawdownTolerated: 20,
        numberOfFunds: 4,
    });

    const [isThinking, setIsThinking] = useState(false);
    const [aiSteps, setAiSteps] = useState([]);
    const [portfolioData, setPortfolioData] = useState(null);
    const [error, setError] = useState(null);
    const endOfMessagesRef = useRef(null);

    // ── DERIVED METRICS ─────────────────────────────────────────────────────
    const { investedAmount, estimatedReturns, totalValue } = calculateSIPFutureValue(
        inputs.monthlyInvestment, 
        inputs.years, 
        inputs.expectedReturnRate
    );

    // Derive risk profile based on expected return
    let riskProfile = 'Moderate';
    let maxDrawdownPct = 20;
    if (inputs.expectedReturnRate <= 9) {
        riskProfile = 'Conservative';
        maxDrawdownPct = 10;
    } else if (inputs.expectedReturnRate >= 14) {
        riskProfile = 'Aggressive';
        maxDrawdownPct = 35;
    }

    const formatCurrency = (val) => '₹' + val.toLocaleString('en-IN', { maximumFractionDigits: 0 });

    // ── EVENT HANDLERS ──────────────────────────────────────────────────────
    const handleSliderChange = (e) => setInputs({ ...inputs, [e.target.name]: Number(e.target.value) });
    const handleInputChange = (e) => {
        let value = e.target.name === 'fundCategory' ? e.target.value : Number(e.target.value);
        if (e.target.name === 'expectedReturnRate' && value > 30) value = 30;
        if (e.target.name === 'years' && value > 40) value = 40;
        if (e.target.name === 'monthlyInvestment' && value > 1000000) value = 1000000;
        setInputs({ ...inputs, [e.target.name]: value });
    };

    const handleGetRecommendations = async () => {
        if (!totalValue) return;
        
        setIsThinking(true);
        setAiSteps([]);
        setPortfolioData(null);
        setError(null);

        try {
            const response = await fetch('/api/advisor/plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    params: {
                        goal: 'Wealth creation',
                        horizonYears: inputs.years,
                        monthlySIP: inputs.monthlyInvestment,
                        lumpSum: 0,
                        targetCorpus: totalValue,
                        maxDrawdownPct: inputs.maxDrawdownTolerated,
                        riskProfile: riskProfile.toLowerCase(),
                        expectedCAGR: inputs.expectedReturnRate,
                        fundCategory: inputs.fundCategory,
                        numberOfFunds: inputs.numberOfFunds
                    }
                }),
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Failed to connect to AI server');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;
            let rawJsonBuffer = '';

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;
                if (value) {
                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6).trim();
                            if (!dataStr) continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'step') {
                                    setAiSteps(prev => {
                                        const exists = prev.findIndex(s => s.title === data.title);
                                        if (exists !== -1) {
                                            const updated = [...prev];
                                            updated[exists] = data;
                                            return updated;
                                        }
                                        return [...prev, data];
                                    });
                                } else if (data.type === 'result') {
                                    setPortfolioData(data.recommendation);
                                } else if (data.type === 'error') {
                                    setError(data.message);
                                }
                            } catch (e) {
                                console.error('Error parsing SSE json:', e, dataStr);
                            }
                        }
                    }
                }
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsThinking(false);
        }
    };

    useEffect(() => {
        if (endOfMessagesRef.current) {
            endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [aiSteps, portfolioData]);

    // ── COMPONENTS ──────────────────────────────────────────────────────────

    const AllocationChart = ({ funds }) => {
        const CHART_COLORS = ['#0ea5e9', '#6366f1', '#10b981', '#f59e0b', '#8b5cf6'];
        const data = funds.map(f => ({ name: f.name, value: Number(f.allocation_percentage) }));
        
        return (
            <div className="h-64 w-full flex flex-col md:flex-row items-center gap-6">
                <div className="h-full w-full md:w-1/2 relative shrink-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%" cy="50%"
                                innerRadius={60} outerRadius={85}
                                paddingAngle={2}
                                dataKey="value"
                                stroke="none"
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                                ))}
                            </Pie>
                            <RechartsTooltip formatter={(value) => `${value}%`} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <div className="w-full md:w-1/2 flex flex-col gap-3 justify-center">
                    {funds.map((f, i) => (
                        <div key={i} className="flex justify-between items-center text-sm">
                            <div className="flex items-center gap-2 overflow-hidden">
                                <div className="w-3 h-3 rounded-full shrink-0" style={{backgroundColor: CHART_COLORS[i % CHART_COLORS.length]}}></div>
                                <span className="text-slate-600 truncate" title={f.name}>{f.name}</span>
                            </div>
                            <span className="font-bold text-slate-800 ml-2">{f.allocation_percentage}%</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    const FundCard = ({ fund }) => {
        const [expanded, setExpanded] = useState(false);
        
        return (
            <div className="bg-white rounded-2xl p-5 md:p-6 shadow-[0_4px_16px_rgba(15,23,42,0.04)] border border-slate-100 flex flex-col">
                <div className="flex justify-between items-start mb-2 gap-4">
                    <h4 className="font-bold text-slate-900 text-lg leading-tight">{fund.name}</h4>
                    <span className="text-xl font-bold text-finance-primary shrink-0">{fund.allocation_percentage}%</span>
                </div>
                
                <div className="flex flex-wrap gap-2 mb-4">
                    <span className="px-2.5 py-1 bg-slate-100 text-slate-600 text-xs font-medium rounded-md">{fund.category}</span>
                    <span className="px-2.5 py-1 bg-amber-50 text-amber-700 text-xs font-medium rounded-md border border-amber-200/50">{fund.risk_level} Risk</span>
                </div>
                
                <p className="text-sm text-slate-700 mb-6">{fund.reason_short}</p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 pt-4 border-t border-slate-100">
                    <div>
                        <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold mb-1">5Y CAGR</div>
                        <div className="font-bold text-slate-900">{fund.metrics?.cagr_5y_percentage ? `${fund.metrics.cagr_5y_percentage}%` : 'N/A'}</div>
                    </div>
                    <div>
                        <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold mb-1">Alpha</div>
                        <div className="font-bold text-slate-900">{fund.metrics?.alpha ?? 'N/A'}</div>
                    </div>
                    <div>
                        <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold mb-1">Beta</div>
                        <div className="font-bold text-slate-900">{fund.metrics?.beta ?? 'N/A'}</div>
                    </div>
                    <div>
                        <div className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold mb-1">Sharpe</div>
                        <div className="font-bold text-slate-900">{fund.metrics?.sharpe_ratio ?? 'N/A'}</div>
                    </div>
                </div>

                <div className="mt-auto pt-2">
                    <button 
                        onClick={() => setExpanded(!expanded)}
                        aria-expanded={expanded}
                        className="text-finance-primary text-sm font-semibold hover:text-emerald-700 transition-colors flex items-center gap-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-finance-primary rounded"
                    >
                        {expanded ? 'Hide detailed reasoning' : 'Why this fund?'}
                        <svg className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                    </button>
                    
                    {expanded && (
                        <div className="mt-4 p-4 bg-slate-50 rounded-xl text-sm text-slate-600 leading-relaxed animate-fade-in border border-slate-100">
                            {fund.reason_detailed}
                        </div>
                    )}
                </div>
            </div>
        );
    };

    // Calculate Estimated Corpus based on Weighted CAGR
    let estimatedCorpus = 0;
    if (portfolioData?.portfolio_summary?.portfolio_metrics?.weighted_cagr_percentage) {
        const r = portfolioData.portfolio_summary.portfolio_metrics.weighted_cagr_percentage / 100;
        const t = inputs.years;
        const lumpSumFV = (inputs.lumpSum || 0) * Math.pow(1 + r, t);
        const monthlyRate = r / 12;
        const sipFV = inputs.monthlyInvestment > 0 
            ? inputs.monthlyInvestment * ((Math.pow(1 + monthlyRate, t * 12) - 1) / monthlyRate) * (1 + monthlyRate)
            : 0;
        estimatedCorpus = Math.round(lumpSumFV + sipFV);
    }

    // ── RENDER ──────────────────────────────────────────────────────────────
    return (
        <div className="pb-24 space-y-8 max-w-5xl mx-auto animate-fade-in">
            
            {/* Page Header */}
            <header className="border-b border-slate-200/60 pb-6 mb-8">
                <h1 className="text-3xl font-bold text-slate-900 mb-2">AI Wealth Planner</h1>
                <p className="text-slate-500 text-base">A personalized mutual fund portfolio based on your risk profile and investment horizon.</p>
            </header>

            {/* Input Configurator (Kept for functionality, styled beautifully) */}
            <div className="bg-white rounded-2xl p-6 md:p-8 shadow-[0_4px_16px_rgba(15,23,42,0.03)] border border-slate-100">
                <div className="grid md:grid-cols-3 gap-8 mb-8">
                    {/* Monthly Investment */}
                    <div>
                        <label className="block text-sm font-semibold text-slate-700 mb-3">Monthly investment</label>
                        <div className="relative group">
                            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 font-medium">₹</span>
                            <input 
                                type="number" name="monthlyInvestment" value={inputs.monthlyInvestment} onChange={handleInputChange}
                                className="w-full bg-slate-50 border border-transparent hover:border-slate-200 focus:bg-white focus:border-finance-primary/40 focus:ring-4 focus:ring-finance-primary/10 rounded-xl pl-10 pr-4 py-3 text-slate-900 font-bold transition-all outline-none"
                            />
                        </div>
                        <input type="range" name="monthlyInvestment" min="500" max="100000" step="500" value={inputs.monthlyInvestment} onChange={handleSliderChange} className="w-full mt-4 accent-finance-primary" />
                    </div>

                    {/* Expected Return Rate */}
                    <div>
                        <label className="block text-sm font-semibold text-slate-700 mb-3">Expected return (p.a)</label>
                        <div className="relative group">
                            <input 
                                type="number" name="expectedReturnRate" value={inputs.expectedReturnRate} onChange={handleInputChange}
                                className="w-full bg-slate-50 border border-transparent hover:border-slate-200 focus:bg-white focus:border-finance-primary/40 focus:ring-4 focus:ring-finance-primary/10 rounded-xl pl-4 pr-10 py-3 text-slate-900 font-bold transition-all outline-none"
                            />
                            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 font-medium">%</span>
                        </div>
                        <input type="range" name="expectedReturnRate" min="1" max="30" step="0.1" value={inputs.expectedReturnRate} onChange={handleSliderChange} className="w-full mt-4 accent-finance-primary" />
                    </div>

                    {/* Time Period */}
                    <div>
                        <label className="block text-sm font-semibold text-slate-700 mb-3">Time period</label>
                        <div className="relative group">
                            <input 
                                type="number" name="years" value={inputs.years} onChange={handleInputChange}
                                className="w-full bg-slate-50 border border-transparent hover:border-slate-200 focus:bg-white focus:border-finance-primary/40 focus:ring-4 focus:ring-finance-primary/10 rounded-xl pl-4 pr-12 py-3 text-slate-900 font-bold transition-all outline-none"
                            />
                            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 font-medium">Yr</span>
                        </div>
                        <input type="range" name="years" min="1" max="40" step="1" value={inputs.years} onChange={handleSliderChange} className="w-full mt-4 accent-finance-primary" />
                    </div>
                </div>

                {/* Advanced Settings */}
                <div className="mb-8 pt-6 border-t border-slate-100">
                    <h3 className="text-sm font-bold text-slate-900 mb-4 uppercase tracking-wider">Advanced Controls</h3>
                    <div className="grid md:grid-cols-3 gap-8">
                        {/* Fund Category */}
                        <div>
                            <label className="block text-xs font-semibold text-slate-500 mb-2">Fund Category Preference</label>
                            <select 
                                name="fundCategory" value={inputs.fundCategory} onChange={handleInputChange}
                                className="w-full bg-slate-50 border border-slate-200 focus:bg-white focus:border-finance-primary/40 focus:ring-4 focus:ring-finance-primary/10 rounded-xl px-4 py-2.5 text-slate-700 font-medium transition-all outline-none"
                            >
                                <option value="Any Equity">Any Equity (Diversified)</option>
                                <option value="Large Cap">Large Cap Bias</option>
                                <option value="Mid & Small Cap">Mid & Small Cap Bias</option>
                                <option value="Flexi/Multi Cap">Flexi/Multi Cap Bias</option>
                                <option value="Sectoral/Thematic">Sectoral/Thematic Bias</option>
                                <option value="Index Funds">Index Funds (Passive)</option>
                                <option value="Debt/Hybrid">Debt & Hybrid Bias</option>
                            </select>
                        </div>
                        
                        {/* Max Drawdown */}
                        <div>
                            <label className="block text-xs font-semibold text-slate-500 mb-2">Max Drawdown Tolerated: {inputs.maxDrawdownTolerated}%</label>
                            <input 
                                type="range" name="maxDrawdownTolerated" min="5" max="50" step="5" 
                                value={inputs.maxDrawdownTolerated} onChange={handleSliderChange} 
                                className="w-full mt-2 accent-finance-primary" 
                            />
                            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                                <span>Conservative (5%)</span>
                                <span>Aggressive (50%)</span>
                            </div>
                        </div>

                        {/* Number of Funds */}
                        <div>
                            <label className="block text-xs font-semibold text-slate-500 mb-2">Number of Funds ({inputs.numberOfFunds})</label>
                            <input 
                                type="range" name="numberOfFunds" min="1" max="7" step="1" 
                                value={inputs.numberOfFunds} onChange={handleSliderChange} 
                                className="w-full mt-2 accent-finance-primary" 
                            />
                            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                                <span>Concentrated (1)</span>
                                <span>Diversified (7)</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex flex-col md:flex-row justify-between items-center gap-6 pt-6 border-t border-slate-100">
                    <div className="text-center md:text-left">
                        <div className="text-sm text-slate-500 font-medium mb-1">Target Corpus Value</div>
                        <div className="text-2xl font-bold text-slate-900">{formatCurrency(totalValue)}</div>
                    </div>
                    <button
                        onClick={handleGetRecommendations}
                        disabled={isThinking}
                        className="w-full md:w-auto bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-8 py-3.5 rounded-xl transition-all shadow-[0_4px_12px_rgba(16,185,129,0.2)] hover:shadow-[0_6px_16px_rgba(16,185,129,0.3)] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {isThinking ? (
                            <>
                                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                Analyzing funds...
                            </>
                        ) : '✨ Generate AI Portfolio'}
                    </button>
                </div>
            </div>

            {/* AI Analysis Progress Timeline */}
            {(aiSteps.length > 0) && (
                <div className="bg-slate-50 rounded-2xl p-6 md:p-8 border border-slate-100">
                    <h3 className="text-lg font-bold text-slate-900 mb-6 flex items-center gap-2">
                        <div className="w-2 h-2 bg-finance-primary rounded-full animate-pulse"></div>
                        AI Analysis Progress
                    </h3>
                    
                    {error && (
                        <div className="bg-red-50 text-red-700 p-4 rounded-xl border border-red-100 mb-6 flex items-start gap-3">
                            <svg className="w-5 h-5 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            <div>
                                <h4 className="font-semibold">Analysis Failed</h4>
                                <p className="text-sm mt-1">{error}</p>
                            </div>
                        </div>
                    )}

                    <div className="space-y-4 relative before:absolute before:inset-0 before:ml-[11px] before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-slate-200">
                        {aiSteps.map((step, idx) => {
                            const isDone = step.status === 'done';
                            const isLoading = step.status === 'loading';
                            return (
                                <div key={idx} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                                    <div className="flex items-center justify-center w-6 h-6 rounded-full border-2 border-white bg-slate-200 text-slate-500 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10" style={isDone ? {backgroundColor: '#10b981', color: 'white'} : isLoading ? {backgroundColor: '#3b82f6', color: 'white'} : {}}>
                                        {isDone ? (
                                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                                        ) : isLoading ? (
                                            <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                        ) : <div className="w-1.5 h-1.5 bg-current rounded-full"></div>}
                                    </div>
                                    <div className="w-[calc(100%-3rem)] md:w-[calc(50%-1.5rem)] bg-white p-4 rounded-xl shadow-sm border border-slate-100">
                                        <h4 className={`font-semibold text-sm ${isDone ? 'text-slate-900' : 'text-slate-700'}`}>{step.title}</h4>
                                        {step.detail && <p className="text-xs text-slate-500 mt-1">{step.detail}</p>}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Structured Portfolio Presentation */}
            {portfolioData && (
                <div className="space-y-8 animate-fade-in">
                    
                    {/* Portfolio Summary Card */}
                    <div className="grid md:grid-cols-2 gap-6 md:gap-8">
                        <div className="bg-white rounded-3xl p-8 shadow-[0_8px_30px_rgba(15,23,42,0.04)] border border-slate-100 flex flex-col justify-center">
                            <h2 className="text-2xl font-bold text-slate-900 mb-2">{portfolioData.portfolio_summary.title}</h2>
                            <p className="text-slate-500 mb-8 leading-relaxed">{portfolioData.portfolio_summary.description}</p>
                            
                            <div className="grid grid-cols-2 gap-y-6 gap-x-4 mb-6">
                                <div>
                                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Risk Level</div>
                                    <div className="font-semibold text-slate-800">{portfolioData.portfolio_summary.risk_level}</div>
                                </div>
                                <div>
                                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Horizon</div>
                                    <div className="font-semibold text-slate-800">{portfolioData.portfolio_summary.investment_horizon_years} Years</div>
                                </div>
                                <div>
                                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Objective</div>
                                    <div className="font-semibold text-slate-800">{portfolioData.portfolio_summary.objective}</div>
                                </div>
                                <div>
                                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Funds</div>
                                    <div className="font-semibold text-slate-800">{portfolioData.funds.length} Recommended</div>
                                </div>
                            </div>
                            
                            {portfolioData.portfolio_summary.portfolio_metrics && (
                                <div className="mt-auto pt-6 border-t border-slate-100">
                                    <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <svg className="w-4 h-4 text-finance-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                                        Estimated Portfolio Metrics
                                    </h4>
                                    <div className="grid grid-cols-3 gap-4 bg-slate-50 p-4 rounded-t-xl border border-slate-100 border-b-0">
                                        <div>
                                            <div className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider mb-1">W. Beta</div>
                                            <div className="font-bold text-slate-900">{portfolioData.portfolio_summary.portfolio_metrics.weighted_beta ?? 'N/A'}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider mb-1">Max DD</div>
                                            <div className="font-bold text-slate-900">{portfolioData.portfolio_summary.portfolio_metrics.estimated_drawdown_percentage ? `-${portfolioData.portfolio_summary.portfolio_metrics.estimated_drawdown_percentage}%` : 'N/A'}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider mb-1">W. CAGR</div>
                                            <div className="font-bold text-emerald-600">{portfolioData.portfolio_summary.portfolio_metrics.weighted_cagr_percentage ? `${portfolioData.portfolio_summary.portfolio_metrics.weighted_cagr_percentage}%` : 'N/A'}</div>
                                        </div>
                                    </div>
                                    <div className="bg-emerald-50 p-4 rounded-b-xl border border-emerald-100">
                                        <div className="flex justify-between items-center mb-2">
                                            <div className="text-[10px] text-emerald-800 font-semibold uppercase tracking-wider">Estimated Total Corpus</div>
                                            <div className="font-bold text-emerald-700 text-lg">₹{estimatedCorpus.toLocaleString('en-IN')}</div>
                                        </div>
                                        <p className="text-[10px] text-emerald-700/80 leading-relaxed">
                                            * This projection is based on the historical {portfolioData.portfolio_summary.portfolio_metrics.weighted_cagr_percentage}% Weighted CAGR of the AI-selected funds, compared to your baseline expectation of {inputs.expectedReturnRate}%.
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Allocation Chart */}
                        <div className="bg-white rounded-3xl p-8 shadow-[0_8px_30px_rgba(15,23,42,0.04)] border border-slate-100">
                            <h3 className="text-lg font-bold text-slate-900 mb-6">Asset Allocation</h3>
                            <AllocationChart funds={portfolioData.funds} />
                        </div>
                    </div>

                    {/* Recommended Funds */}
                    <div>
                        <h3 className="text-xl font-bold text-slate-900 mb-4 px-2">Recommended Funds</h3>
                        <div className="grid gap-4">
                            {portfolioData.funds.map((fund, idx) => (
                                <FundCard key={idx} fund={fund} />
                            ))}
                        </div>
                    </div>

                    {/* Strategy & Risks Grid */}
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-slate-50 rounded-2xl p-6 md:p-8 border border-slate-200/60">
                            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <svg className="w-5 h-5 text-finance-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                                Portfolio Strategy
                            </h3>
                            <ul className="space-y-3">
                                {portfolioData.strategy.map((item, idx) => (
                                    <li key={idx} className="flex items-start gap-3 text-slate-700 text-sm leading-relaxed">
                                        <div className="w-1.5 h-1.5 rounded-full bg-slate-400 mt-1.5 shrink-0"></div>
                                        <span>{item}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                        
                        <div className="bg-rose-50/50 rounded-2xl p-6 md:p-8 border border-rose-100">
                            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <svg className="w-5 h-5 text-rose-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                                Key Risks
                            </h3>
                            <div className="space-y-4">
                                {portfolioData.risks.map((risk, idx) => (
                                    <div key={idx}>
                                        <h4 className="font-semibold text-slate-900 text-sm flex items-center gap-2">
                                            {risk.title}
                                            {risk.severity === 'High' && <span className="px-2 py-0.5 bg-rose-100 text-rose-700 text-[10px] uppercase font-bold rounded">High Risk</span>}
                                        </h4>
                                        <p className="text-sm text-slate-600 mt-1 leading-relaxed">{risk.description}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <p className="text-center text-xs text-slate-400 px-4 mt-8">
                        {portfolioData.disclaimer}
                    </p>

                    <div ref={endOfMessagesRef} />
                </div>
            )}
        </div>
    );
};

export default AIWealthPlanner;
