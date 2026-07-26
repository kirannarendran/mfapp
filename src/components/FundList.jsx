import React, { useState, useEffect } from 'react';
import { fetchFundList } from '../api';

const FundList = ({ onSelectFund, comparisonList = [], onToggleCompare, onStartCompare, onClearCompare }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [funds, setFunds] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!searchTerm || searchTerm.length < 2) {
            setFunds([]);
            return;
        }

        setLoading(true);
        const timer = setTimeout(async () => {
            try {
                const data = await fetchFundList(searchTerm);
                setFunds(data);
                setError(null);
            } catch (err) {
                setError('Failed to search funds');
                setFunds([]);
            } finally {
                setLoading(false);
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [searchTerm]);

    if (error) return <div className="text-finance-danger p-4">{error}</div>;

    const isSearching = searchTerm.length > 0;

    return (
        <div className="fund-list pb-20 max-w-4xl">
            <div className="search-container mb-8">
                <h2 className="text-2xl font-bold mb-4 text-finance-text-primary">
                    Fund Universe
                </h2>
                <div className="relative">
                    <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-finance-text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                    <input
                        type="text"
                        placeholder="Search for a mutual fund (e.g., Parag Parikh, SBI)..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full text-base pl-12 pr-4 py-3 shadow-sm border border-finance-border rounded-lg bg-finance-surface focus:border-finance-primary focus:ring-1 focus:ring-finance-primary"
                        style={{ paddingLeft: '3rem' }}
                        autoFocus
                    />
                </div>
            </div>

            {isSearching && (
                <div className="grid gap-3 animate-fade-in">
                    {funds.map((fund) => {
                        const isSelected = comparisonList.some(f => f.schemeCode === fund.schemeCode);
                        return (
                            <div
                                key={fund.schemeCode}
                                className={`card !p-4 flex justify-between items-center transition-all ${isSelected ? 'border-finance-primary bg-finance-primary-soft/30' : 'hover:border-finance-primary/50 hover:bg-finance-surface'}`}
                            >
                                <div
                                    className="flex-grow cursor-pointer"
                                    onClick={() => onSelectFund(fund.schemeCode)}
                                >
                                    <h3 className={`text-base font-semibold mb-1 ${isSelected ? 'text-finance-primary-dark' : 'text-finance-text-primary hover:text-finance-primary'}`}>
                                        {fund.schemeName}
                                    </h3>
                                    <p className="text-sm text-finance-text-secondary">Code: {fund.schemeCode}</p>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onToggleCompare(fund);
                                    }}
                                    className={`ml-4 px-4 py-1.5 rounded-md text-sm font-medium transition-colors border ${isSelected
                                        ? 'bg-finance-primary-soft text-finance-primary border-finance-primary/30 hover:bg-finance-primary-soft/80'
                                        : 'bg-finance-surface text-finance-text-secondary border-finance-border hover:bg-finance-bg hover:text-finance-text-primary'
                                        }`}
                                >
                                    {isSelected ? '✓ Added' : '+ Compare'}
                                </button>
                            </div>
                        );
                    })}
                    {funds.length === 0 && !loading && (
                        <p className="text-finance-text-secondary mt-4 p-8 text-center bg-finance-surface rounded-lg border border-finance-border">No funds found matching "{searchTerm}"</p>
                    )}
                    {loading && (
                        <p className="text-finance-primary mt-4 p-8 text-center">Searching...</p>
                    )}
                </div>
            )}

            {!isSearching && (
                <div className="bg-finance-surface border border-finance-border rounded-lg p-12 text-center text-finance-text-secondary mt-4">
                    <svg className="w-12 h-12 mx-auto text-finance-border mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                    <p className="text-lg">Start typing to search for mutual funds across the market.</p>
                </div>
            )}

            {comparisonList.length > 0 && (
                <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 bg-finance-surface border border-finance-border shadow-lg rounded-full px-6 py-3 flex items-center gap-4 z-50 animate-fade-in">
                    <span className="text-finance-text-primary text-sm font-medium">
                        {comparisonList.length} fund{comparisonList.length !== 1 && 's'} selected
                    </span>
                    <button
                        onClick={onClearCompare}
                        className="text-finance-text-secondary hover:text-finance-negative text-sm font-medium transition-colors px-2"
                    >
                        Clear
                    </button>
                    <button
                        onClick={onStartCompare}
                        className="bg-finance-primary hover:bg-finance-primary-dark text-white px-4 py-1.5 rounded-full text-sm font-semibold transition-colors shadow-sm"
                    >
                        Compare Now &rarr;
                    </button>
                </div>
            )}
        </div>
    );
};

export default FundList;
