import React, { useState, useEffect, useRef } from 'react';
import FundList from './components/FundList';
import FundDetail from './components/FundDetail';
import ComparisonView from './components/ComparisonView';
import AIWealthPlanner from './components/AIWealthPlanner';
import AIPortfolioAnalyzer from './components/AIPortfolioAnalyzer';
import FundScreener from './components/FundScreener';
import AboutPage from './components/AboutPage';
import { fetchSyncStatus } from './api';

function App() {
  const [selectedSchemeCode, setSelectedSchemeCode] = useState(null);
  const [comparisonList, setComparisonList] = useState([]);
  const [isComparing, setIsComparing] = useState(false);
  const [isPlanning, setIsPlanning] = useState(false);
  const [isAnalyzer, setIsAnalyzer] = useState(false);
  const [isScreening, setIsScreening] = useState(false);
  const [isAbout, setIsAbout] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Mobile drawer state
  
  const [syncStatus, setSyncStatus] = useState(null);
  
  // Ref for focus restoration
  const previousFocusRef = useRef(null);
  // Ref for the drawer itself
  const drawerRef = useRef(null);

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

  // Handle escape key, focus trapping, and body scroll lock
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isSidebarOpen) return;
      
      if (e.key === 'Escape') {
        setIsSidebarOpen(false);
        return;
      }

      // Basic focus trap
      if (e.key === 'Tab' && drawerRef.current) {
        const focusableElements = drawerRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            lastElement.focus();
            e.preventDefault();
          }
        } else {
          if (document.activeElement === lastElement) {
            firstElement.focus();
            e.preventDefault();
          }
        }
      }
    };

    if (isSidebarOpen) {
      previousFocusRef.current = document.activeElement;
      document.body.style.overflow = 'hidden';
      // Auto-focus the drawer container or first element
      if (drawerRef.current) {
         const firstFocusable = drawerRef.current.querySelector('button');
         if (firstFocusable) firstFocusable.focus();
      }
    } else {
      document.body.style.overflow = 'unset';
      if (previousFocusRef.current) {
        previousFocusRef.current.focus();
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isSidebarOpen]);

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
    setIsAnalyzer(false);
    setIsScreening(false);
    setIsAbout(false);
  };

  const formatSyncTime = (isoString) => {
    if (!isoString) return 'Never';
    const date = new Date(isoString);
    return date.toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true });
  };

  const currentViewTitle = isAbout ? 'About FundSense.AI' : isAnalyzer ? 'Portfolio X-Ray' : isScreening ? 'Fund Screener' : isPlanning ? 'AI Wealth Planner' : isComparing ? 'Fund Comparison' : selectedSchemeCode ? 'Fund Details' : 'Fund List';

  const NavButton = ({ title, isActive, onClick, iconPath }) => (
    <button 
      onClick={() => { onClick(); setIsSidebarOpen(false); }} 
      title={title}
      aria-current={isActive ? 'page' : undefined}
      className={`w-full text-left px-4 py-3 md:py-2.5 rounded-xl transition-colors flex items-center gap-3 text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-finance-primary focus-visible:ring-offset-2
        ${isActive 
          ? 'bg-finance-primary/10 text-finance-primary' 
          : 'text-finance-text-secondary hover:bg-slate-200/50 hover:text-finance-text-primary'
        }`}>
      <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={isActive ? "2" : "1.5"} d={iconPath}></path>
      </svg>
      <span className="whitespace-nowrap">{title}</span>
    </button>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 font-sans text-finance-text-primary">
      
      {/* Mobile Drawer Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/40 z-30 md:hidden transition-opacity"
          onClick={() => setIsSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Navigation */}
      <aside 
        ref={drawerRef}
        className={`fixed inset-y-0 left-0 z-40 w-[260px] bg-slate-50 border-r border-slate-200/60 flex flex-col transition-transform duration-300 md:relative md:translate-x-0
          ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="h-16 px-6 flex items-center gap-3 shrink-0">
          <div className="w-8 h-8 rounded-xl bg-finance-primary flex shrink-0 items-center justify-center shadow-sm">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
            </svg>
          </div>
          <h1 className="text-lg font-bold text-slate-900 tracking-tight whitespace-nowrap">
            FundSense.AI
          </h1>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
          <NavButton 
            title="Fund List" 
            isActive={!isScreening && !isPlanning && !selectedSchemeCode && !isComparing && !isAnalyzer && !isAbout}
            onClick={handleBackToList}
            iconPath="M4 6h16M4 10h16M4 14h16M4 18h16"
          />
          <NavButton 
            title="Fund Screener" 
            isActive={isScreening}
            onClick={() => { setIsScreening(true); setIsPlanning(false); setIsComparing(false); setIsAnalyzer(false); setIsAbout(false); setSelectedSchemeCode(null); }}
            iconPath="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
          />
          <NavButton 
            title="AI Wealth Planner" 
            isActive={isPlanning}
            onClick={() => { setIsPlanning(true); setIsScreening(false); setIsComparing(false); setIsAnalyzer(false); setIsAbout(false); setSelectedSchemeCode(null); }}
            iconPath="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
          />
          <NavButton 
            title="Portfolio X-Ray" 
            isActive={isAnalyzer} 
            onClick={() => { setIsAnalyzer(true); setIsPlanning(false); setIsScreening(false); setIsComparing(false); setIsAbout(false); setSelectedSchemeCode(null); }}
            iconPath="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" 
          />
          <NavButton 
            title="About FundSense.AI" 
            isActive={isAbout} 
            onClick={() => { setIsAbout(true); setIsAnalyzer(false); setIsPlanning(false); setIsScreening(false); setIsComparing(false); setSelectedSchemeCode(null); }}
            iconPath="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
          />
        </nav>

        {/* Compact Footer Status Area */}
        <div className="p-5 mt-auto">
          {syncStatus && (
            <div className="flex flex-col gap-1 text-sm">
              <div className="flex items-center gap-2">
                {syncStatus.isSyncing ? (
                  <div className="w-2.5 h-2.5 rounded-full bg-finance-warning animate-pulse" />
                ) : (
                  <div className="w-2.5 h-2.5 rounded-full bg-finance-primary" />
                )}
                <span className="font-medium text-slate-700">
                  {syncStatus.isSyncing ? 'Syncing data...' : 'System ready'}
                </span>
              </div>
              {!syncStatus.isSyncing && (
                <span className="text-xs text-slate-500 pl-4.5 ml-0.5 block">
                  Last synced: {formatSyncTime(syncStatus.lastSyncDate)}
                </span>
              )}
            </div>
          )}
        </div>
      </aside>

      {/* Main Content Column */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        
        {/* Application Header */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200/60 flex items-center px-4 md:px-8 shrink-0 z-20">
          <button 
            onClick={() => setIsSidebarOpen(true)} 
            className="md:hidden p-2 -ml-2 mr-3 rounded-lg text-slate-600 hover:bg-slate-100 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-finance-primary"
            aria-label="Open navigation menu">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
            </svg>
          </button>
          
          <h2 className="text-[15px] font-semibold text-slate-800">
            {currentViewTitle}
          </h2>
        </header>

        {/* Main Scrollable Area */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden relative">
          <div className="max-w-[1152px] w-full mx-auto px-4 md:px-6 lg:px-8 py-6 md:py-8 min-h-full flex flex-col">
            {isAbout ? (
              <AboutPage />
            ) : isAnalyzer ? (
              <AIPortfolioAnalyzer onBack={handleBackToList} />
            ) : isScreening ? (
              <FundScreener 
                onBack={handleBackToList} 
                onSelectFund={(code) => {
                  setSelectedSchemeCode(code);
                  setIsScreening(false);
                }} 
              />
            ) : isPlanning ? (
              <AIWealthPlanner onBack={handleBackToList} />
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
                onClearCompare={() => setComparisonList([])}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
