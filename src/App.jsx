import React, { useState, useEffect } from 'react';
import FundList from './components/FundList';
import FundDetail from './components/FundDetail';
import ComparisonView from './components/ComparisonView';
import FinancialPlanner from './components/FinancialPlanner';
import FundScreener from './components/FundScreener';
import { fetchSyncStatus } from './api';

function App() {
  const [selectedSchemeCode, setSelectedSchemeCode] = useState(null);
  const [comparisonList, setComparisonList] = useState([]); // Array of { schemeCode, schemeName }
  const [isComparing, setIsComparing] = useState(false);
  const [isPlanning, setIsPlanning] = useState(false);
  const [isScreening, setIsScreening] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const [syncStatus, setSyncStatus] = useState(null);

  useEffect(() => {
    let intervalId;
    
    const checkSyncStatus = async () => {
      try {
        const status = await fetchSyncStatus();
        setSyncStatus(status);
        
        if (!status.isSyncing && intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      } catch (err) {
        console.error("Failed to fetch sync status", err);
      }
    };

    checkSyncStatus();
    intervalId = setInterval(checkSyncStatus, 10000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const handleToggleCompare = (fund) => {
    setComparisonList(prev => {
      const exists = prev.find(f => f.schemeCode === fund.schemeCode);
      if (exists) {
        return prev.filter(f => f.schemeCode !== fund.schemeCode);
      }
      if (prev.length >= 3) {
        alert("You can compare up to 3 funds at a time.");
        return prev;
      }
      return [...prev, fund];
    });
  };

  const handleStartCompare = () => {
    setIsComparing(true);
    setSelectedSchemeCode(null);
  };

  const handleBackToList = () => {
    setSelectedSchemeCode(null);
    setIsComparing(false);
    setIsPlanning(false);
    setIsScreening(false);
  };

  const formatSyncTime = (isoString) => {
    if (!isoString) return 'Never';
    const date = new Date(isoString);
    return date.toLocaleString();
  };

  return (
    <div className="flex h-screen overflow-hidden bg-finance-bg font-sans">
      <aside className={`${isSidebarOpen ? 'w-[232px]' : 'w-[72px]'} bg-finance-surface border-r border-finance-border flex flex-col z-20 shrink-0 transition-all duration-300`}>
        <div className="h-16 px-4 border-b border-finance-border flex items-center justify-center gap-3 overflow-hidden">
            <div className="w-8 h-8 rounded bg-finance-primary flex shrink-0 items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
            </div>
            {isSidebarOpen && (
                <h1 className="text-lg font-bold text-finance-text-primary tracking-tight whitespace-nowrap animate-fade-in">
                    MF Tracker
                </h1>
            )}
        </div>

        <nav className="flex-1 px-3 py-6 space-y-2 overflow-x-hidden">
            <button 
                onClick={handleBackToList} 
                title="Fund List"
                className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-center ${isSidebarOpen ? 'gap-3' : 'justify-center'} text-sm font-medium ${!isScreening && !isPlanning && !selectedSchemeCode && !isComparing ? 'bg-finance-primary-soft text-finance-primary' : 'text-finance-text-secondary hover:bg-finance-bg hover:text-finance-text-primary'}`}>
                <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 6h16M4 10h16M4 14h16M4 18h16"></path></svg>
                {isSidebarOpen && <span className="whitespace-nowrap animate-fade-in">Fund List</span>}
            </button>
            <button 
                onClick={() => { setIsScreening(true); setIsPlanning(false); setIsComparing(false); setSelectedSchemeCode(null); }} 
                title="Fund Screener"
                className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-center ${isSidebarOpen ? 'gap-3' : 'justify-center'} text-sm font-medium ${isScreening ? 'bg-finance-primary-soft text-finance-primary' : 'text-finance-text-secondary hover:bg-finance-bg hover:text-finance-text-primary'}`}>
                <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"></path></svg>
                {isSidebarOpen && <span className="whitespace-nowrap animate-fade-in">Fund Screener</span>}
            </button>
            <button 
                onClick={() => { setIsPlanning(true); setIsScreening(false); setIsComparing(false); setSelectedSchemeCode(null); }} 
                title="Financial Planner"
                className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-center ${isSidebarOpen ? 'gap-3' : 'justify-center'} text-sm font-medium ${isPlanning ? 'bg-finance-primary-soft text-finance-primary' : 'text-finance-text-secondary hover:bg-finance-bg hover:text-finance-text-primary'}`}>
                <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path></svg>
                {isSidebarOpen && <span className="whitespace-nowrap animate-fade-in">Financial Planner</span>}
            </button>
        </nav>

        <div className="p-4 border-t border-finance-border overflow-x-hidden">
            {syncStatus && (
                <div className={`text-xs flex flex-col gap-2 rounded-lg ${isSidebarOpen ? 'p-3 border bg-finance-bg border-finance-border' : 'p-1 items-center justify-center'}`}>
                    {syncStatus.isSyncing ? (
                        <div className={`flex items-center text-finance-primary font-medium ${isSidebarOpen ? 'gap-2' : ''}`} title="Syncing data...">
                            <div className="w-4 h-4 shrink-0 border-2 border-finance-primary border-t-transparent rounded-full animate-spin"></div>
                            {isSidebarOpen && <span className="animate-fade-in">Syncing data...</span>}
                        </div>
                    ) : (
                        <>
                            <div className={`flex items-center text-finance-positive font-medium ${isSidebarOpen ? 'gap-1.5' : ''}`} title={`System Ready\nLast synced: ${formatSyncTime(syncStatus.lastSyncDate)}`}>
                                <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                                {isSidebarOpen && <span className="animate-fade-in whitespace-nowrap">System Ready</span>}
                            </div>
                            {isSidebarOpen && <span className="text-finance-text-secondary leading-relaxed whitespace-nowrap animate-fade-in">Last synced:<br/>{formatSyncTime(syncStatus.lastSyncDate)}</span>}
                        </>
                    )}
                </div>
            )}
        </div>
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="h-16 bg-finance-surface border-b border-finance-border flex items-center px-6 shrink-0 z-10 gap-4">
            <button 
                onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
                className="p-2 -ml-2 rounded-lg text-finance-text-secondary hover:bg-finance-bg hover:text-finance-text-primary transition-colors focus:outline-none focus:ring-2 focus:ring-finance-primary/20"
                title={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
                </svg>
            </button>
            <div className="flex items-center text-sm font-medium text-finance-text-primary">
                {isScreening ? 'Fund Screener' : isPlanning ? 'Financial Planner' : isComparing ? 'Fund Comparison' : selectedSchemeCode ? 'Fund Details' : 'Fund List'}
            </div>
        </header>

        <main className="flex-1 overflow-y-auto bg-finance-bg p-6">
            <div className="max-w-6xl mx-auto">
            {isScreening ? (
                <FundScreener 
                    onBack={handleBackToList} 
                    onSelectFund={(code) => {
                        setSelectedSchemeCode(code);
                        setIsScreening(false);
                    }} 
                />
            ) : isPlanning ? (
                <FinancialPlanner onBack={handleBackToList} />
            ) : isComparing ? (
                <ComparisonView
                    funds={comparisonList}
                    onBack={handleBackToList}
                />
            ) : selectedSchemeCode ? (
                <FundDetail
                    schemeCode={selectedSchemeCode}
                    onBack={handleBackToList}
                />
            ) : (
                <FundList
                    onSelectFund={setSelectedSchemeCode}
                    comparisonList={comparisonList}
                    onToggleCompare={handleToggleCompare}
                    onStartCompare={handleStartCompare}
                />
            )}
            </div>
        </main>
      </div>
    </div>
  );
}

export default App;
