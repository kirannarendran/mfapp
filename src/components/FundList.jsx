import React, { useState, useEffect } from 'react';
import { fetchFundList } from '../api';

const QUICK_SEARCH_CHIPS = [
    { label: '⭐ Top Sharpe', query: '' },
    { label: 'Parag Parikh', query: 'Parag Parikh' },
    { label: 'Quant', query: 'Quant' },
    { label: 'HDFC', query: 'HDFC' },
    { label: 'Mirae', query: 'Mirae' },
    { label: 'SBI', query: 'SBI' },
    { label: 'Nifty 50 Index', query: 'Index' },
];

const FundList = ({ onSelectFund, comparisonList = [], onToggleCompare, onStartCompare, onClearCompare }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [funds, setFunds] = useState([]);
    const [isCurated, setIsCurated] = useState(true);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Initial load for curated spotlight funds + search debouncer
    useEffect(() => {
        let isMounted = true;
        setLoading(true);

        const timer = setTimeout(async () => {
            try {
                const data = await fetchFundList(searchTerm.trim());
                if (isMounted) {
                    setFunds(data);
                    setIsCurated(!searchTerm.trim() || searchTerm.trim().length < 2);
                    setError(null);
                }
            } catch (err) {
                if (isMounted) {
                    setError('Failed to load funds');
                    setFunds([]);
                }
            } finally {
                if (isMounted) setLoading(false);
            }
        }, searchTerm ? 300 : 0);

        return () => {
            isMounted = false;
            clearTimeout(timer);
        };
    }, [searchTerm]);

    return (
        <div className="fund-list pb-24 max-w-5xl mx-auto w-full">
            {/* Header & Search */}
            <div className="search-container mb-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
                    <div>
                        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
                            Fund Universe
                        </h2>
                        <p className="text-sm text-slate-500 mt-0.5">
                            Explore 5,000+ Direct Growth schemes with live NAV &amp; risk ratios
                        </p>
                    </div>
                    <span className="self-start sm:self-auto text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200/60 flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                        Live AMFI Feeds
                    </span>
                </div>

                <div className="relative mb-3">
                    <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                    </svg>
                    <input
                        type="text"
                        placeholder="Search by fund name or AMC (e.g. Parag Parikh Flexi, Quant, Mirae Large Cap)..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        style={{ paddingLeft: '3.25rem' }}
                        className="w-full text-base !pl-13 pr-10 py-3.5 shadow-sm border border-slate-200 rounded-2xl bg-white focus:outline-none focus:ring-2 focus:ring-finance-primary/20 focus:border-finance-primary transition-all text-slate-900 placeholder:text-slate-400"
                    />
                    {searchTerm && (
                        <button
                            onClick={() => setSearchTerm('')}
                            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 p-1"
                            title="Clear search"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    )}
                </div>

                {/* Quick Filter Chips */}
                <div className="flex items-center gap-2 overflow-x-auto pb-1.5 pt-1 text-xs">
                    <span className="text-slate-400 font-medium whitespace-nowrap mr-1">Quick:</span>
                    {QUICK_SEARCH_CHIPS.map((chip, idx) => {
                        const isActive = searchTerm === chip.query;
                        return (
                            <button
                                key={idx}
                                onClick={() => setSearchTerm(chip.query)}
                                className={`px-3 py-1.5 rounded-xl font-medium transition-all whitespace-nowrap border ${
                                    isActive
                                        ? 'bg-finance-primary text-white border-finance-primary shadow-sm'
                                        : 'bg-white text-slate-600 border-slate-200/80 hover:bg-slate-50 hover:border-slate-300'
                                }`}
                            >
                                {chip.label}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="p-4 mb-6 bg-rose-50 border border-rose-200 rounded-2xl text-xs text-rose-700 font-medium">
                    {error}
                </div>
            )}

            {/* Section Heading Badge */}
            <div className="flex items-center justify-between mb-3 px-1">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    {isCurated ? '⭐ Top Risk-Adjusted Spotlight' : `Search Results (${funds.length})`}
                </span>
                {isCurated && (
                    <span className="text-[11px] text-slate-400 font-medium">
                        Ranked by risk-adjusted Sharpe ratio
                    </span>
                )}
            </div>

            {/* Skeleton Loaders */}
            {loading && funds.length === 0 && (
                <div className="grid gap-3">
                    {[1, 2, 3, 4, 5].map((i) => (
                        <div key={i} className="bg-white p-5 rounded-2xl border border-slate-100 shadow-sm animate-pulse flex flex-col sm:flex-row justify-between gap-4">
                            <div className="space-y-2 flex-1">
                                <div className="h-4 bg-slate-200 rounded w-3/4"></div>
                                <div className="h-3 bg-slate-100 rounded w-1/3"></div>
                            </div>
                            <div className="h-8 bg-slate-200 rounded-xl w-24"></div>
                        </div>
                    ))}
                </div>
            )}

            {/* Funds List */}
            <div className="grid gap-3 animate-fade-in">
                {funds.map((fund) => {
                    const isSelected = comparisonList.some(f => f.schemeCode === fund.schemeCode);
                    const hasMetrics = fund.cagr3y != null || fund.cagr5y != null || fund.sharpe != null;

                    return (
                        <div
                            key={fund.schemeCode}
                            className={`bg-white rounded-2xl p-4 sm:p-5 border transition-all duration-150 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-sm hover:shadow-md ${
                                isSelected
                                    ? 'border-finance-primary bg-finance-primary/5 ring-1 ring-finance-primary'
                                    : 'border-slate-200/80 hover:border-slate-300'
                            }`}
                        >
                            <div
                                className="flex-grow cursor-pointer group"
                                onClick={() => onSelectFund(fund.schemeCode)}
                            >
                                <div className="flex items-center gap-2 flex-wrap mb-1.5">
                                    {fund.category && (
                                        <span className="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-slate-100 text-slate-600 border border-slate-200/60">
                                            {fund.category}
                                        </span>
                                    )}
                                    {fund.fundHouse && (
                                        <span className="text-xs text-slate-400 font-medium">
                                            {fund.fundHouse}
                                        </span>
                                    )}
                                </div>

                                <h3 className={`text-base font-bold transition-colors leading-snug ${
                                    isSelected
                                        ? 'text-finance-primary'
                                        : 'text-slate-900 group-hover:text-finance-primary'
                                }`}>
                                    {fund.schemeName}
                                </h3>

                                <div className="flex items-center gap-3 mt-2 text-xs flex-wrap">
                                    <span className="text-slate-400 font-medium">
                                        Code: <strong className="text-slate-600">{fund.schemeCode}</strong>
                                    </span>
                                    {fund.lastNav && (
                                        <span className="text-slate-600 font-medium">
                                            NAV: <strong className="text-slate-900">₹{Number(fund.lastNav).toFixed(2)}</strong>
                                        </span>
                                    )}
                                    {fund.cagr3y != null && (
                                        <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                                            fund.cagr3y >= 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                                        }`}>
                                            3Y CAGR: {fund.cagr3y > 0 ? '+' : ''}{fund.cagr3y.toFixed(1)}%
                                        </span>
                                    )}
                                    {fund.sharpe != null && (
                                        <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-50 text-blue-700">
                                            Sharpe: {fund.sharpe.toFixed(2)}
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                                <button
                                    onClick={() => onSelectFund(fund.schemeCode)}
                                    className="px-3.5 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors"
                                >
                                    Details →
                                </button>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onToggleCompare(fund);
                                    }}
                                    className={`px-4 py-2 rounded-xl text-xs font-bold transition-all border ${
                                        isSelected
                                            ? 'bg-finance-primary text-white border-finance-primary shadow-sm'
                                            : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                                    }`}
                                >
                                    {isSelected ? '✓ In Compare' : '+ Compare'}
                                </button>
                            </div>
                        </div>
                    );
                })}

                {funds.length === 0 && !loading && (
                    <div className="bg-white rounded-3xl border border-slate-200 p-12 text-center shadow-sm">
                        <div className="w-12 h-12 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-3 text-slate-400">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                            </svg>
                        </div>
                        <h4 className="text-base font-bold text-slate-800 mb-1">No matching funds found</h4>
                        <p className="text-xs text-slate-500 max-w-sm mx-auto mb-4">
                            Try searching for another keyword or AMC name, or click one of the quick filter chips above.
                        </p>
                        <button
                            onClick={() => setSearchTerm('')}
                            className="px-4 py-2 bg-finance-primary text-white text-xs font-semibold rounded-xl hover:bg-finance-primary-dark transition-colors shadow-sm"
                        >
                            Reset to Top Funds
                        </button>
                    </div>
                )}
            </div>

            {/* Floating Comparison Tray */}
            {comparisonList.length > 0 && (
                <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-slate-900 text-white shadow-2xl rounded-2xl px-5 py-3.5 flex items-center gap-4 z-50 animate-fade-in w-[92%] sm:w-auto justify-between border border-slate-800">
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span className="text-sm font-semibold whitespace-nowrap">
                            {comparisonList.length} / 3 funds selected
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={onClearCompare}
                            className="text-xs text-slate-400 hover:text-rose-400 px-2 py-1 rounded transition-colors"
                        >
                            Clear
                        </button>
                        <button
                            onClick={onStartCompare}
                            className="bg-finance-primary hover:bg-blue-600 text-white font-bold px-4 py-2 rounded-xl text-xs transition-colors whitespace-nowrap shadow-md flex items-center gap-1.5"
                        >
                            <span>Compare Head-to-Head</span>
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                            </svg>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default FundList;
