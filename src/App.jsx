import React, { useState, useEffect, useRef } from 'react';
import FundList from './components/FundList';
import FundDetail from './components/FundDetail';
import ComparisonView from './components/ComparisonView';
import AIWealthPlanner from './components/AIWealthPlanner';
import AIPortfolioAnalyzer from './components/AIPortfolioAnalyzer';
import FundScreener from './components/FundScreener';
import AboutPage from './components/AboutPage';
import UserNav from './components/UserNav';
import LandingPage from './components/LandingPage';
import ProfileModal from './components/ProfileModal';
import ErrorBoundary from './components/ErrorBoundary';
import { useAuth } from './context/AuthContext';
import { fetchSyncStatus, triggerManualSync } from './api';
import { trackPageView, trackEvent } from './utils/analytics';

function App() {
  const { user, isAuthenticated, isLoading: isAuthLoading } = useAuth();
  const [isGuestMode, setIsGuestMode] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  // Automatically prompt new users to complete their profile after initial login (only if never saved)
  useEffect(() => {
    if (user && !user.profileCompleted) {
      const promptKey = `fundsense_profile_prompted_${user.id}`;
      const completedKey = `fundsense_profile_completed_${user.id}`;
      if (!sessionStorage.getItem(promptKey) && !localStorage.getItem(completedKey)) {
        setIsProfileOpen(true);
        sessionStorage.setItem(promptKey, 'true');
      }
    }
  }, [user?.id, user?.profileCompleted]);

  const [selectedSchemeCode, setSelectedSchemeCode] = useState(null);
  const [comparisonList, setComparisonList] = useState([]);
  const [isComparing, setIsComparing] = useState(false);
  const [isPlanning, setIsPlanning] = useState(false);
  const [isAnalyzer, setIsAnalyzer] = useState(false);
  const [isScreening, setIsScreening] = useState(false);
  const [isAbout, setIsAbout] = useState(false);
  const [isProfile, setIsProfile] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Mobile drawer state
  
  const [syncStatus, setSyncStatus] = useState(null);
  const [syncTrigger, setSyncTrigger] = useState(0);
  
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
  }, [syncTrigger]);

  const handleManualSync = async () => {
    try {
      setSyncStatus(prev => ({ ...prev, isSyncing: true }));
      await triggerManualSync();
      setSyncTrigger(prev => prev + 1);
    } catch (err) {
      console.error("Manual sync failed", err);
      setSyncTrigger(prev => prev + 1);
    }
  };

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
    setIsProfile(false);
  };

  const formatSyncDate = (status) => {
    if (!status) return null;
    let d = null;
    if (status.lastSyncTime) {
      d = new Date(status.lastSyncTime);
    } else if (status.lastSyncDate) {
      const raw = status.lastSyncDate;
      d = new Date(raw.includes('T') ? raw : raw.replace(' ', 'T') + 'Z');
    }
    if (!d || isNaN(d.getTime())) return null;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}`;
  };

  const formatFullSyncTime = (status) => {
    if (!status) return null;
    let d = null;
    if (status.lastSyncTime) {
      d = new Date(status.lastSyncTime);
    } else if (status.lastSyncDate) {
      const raw = status.lastSyncDate;
      d = new Date(raw.includes('T') ? raw : raw.replace(' ', 'T') + 'Z');
    }
    if (!d || isNaN(d.getTime())) return null;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    let hours = d.getHours();
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    return `${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}, ${hours}:${minutes} ${ampm}`;
  };

  const currentViewTitle = isProfile ? 'My Profile' : isAbout ? 'About FundSense.AI' : isAnalyzer ? 'Portfolio X-Ray' : isScreening ? 'Fund Screener' : isPlanning ? 'AI Wealth Planner' : isComparing ? 'Fund Comparison' : selectedSchemeCode ? 'Fund Details' : 'Fund List';

  // Google Analytics Pageview Tracking for SPA
  useEffect(() => {
    if (!isAuthenticated && !isGuestMode) {
      trackPageView('Landing Page', '/');
    } else {
      trackPageView(currentViewTitle);
    }
  }, [isAuthenticated, isGuestMode, currentViewTitle]);

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

  const renderSyncProgress = () => {
    if (!syncStatus || !syncStatus.isSyncing || !syncStatus.state) return null;
    const state = syncStatus.state;
    
    let etaText = 'Calculating...';
    let percentage = 0;

    if (state.total > 0 && state.progress > 0) {
      percentage = Math.round((state.progress / state.total) * 100);
      // eslint-disable-next-line react-hooks/purity
      const elapsedMs = Date.now() - state.startTime;
      const msPerItem = elapsedMs / state.progress;
      const msRemaining = msPerItem * (state.total - state.progress);
      
      if (msRemaining < 60000) {
        etaText = `${Math.round(msRemaining / 1000)}s left`;
      } else {
        etaText = `${Math.round(msRemaining / 60000)}m left`;
      }
    }

    return (
      <div className="mt-2 pl-4.5 ml-0.5 w-full pr-4">
        <div className="flex justify-between text-[10.5px] text-slate-500 mb-1.5 font-medium">
          <span className="truncate max-w-[130px]">{state.currentStep || 'Starting...'}</span>
          {percentage > 0 && <span>{percentage}% ({etaText})</span>}
        </div>
        {state.total > 0 && (
          <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
            <div className="bg-finance-primary h-1.5 rounded-full transition-all duration-500 ease-out" style={{ width: `${percentage}%` }}></div>
          </div>
        )}
      </div>
    );
  };

  if (isAuthLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-50">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-finance-primary border-t-transparent animate-spin" />
          <p className="text-xs text-slate-500 font-medium">Loading FundSense.AI...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated && !isGuestMode) {
    return (
      <LandingPage 
        onExploreGuest={() => {
          trackEvent('guest_explore_click');
          setIsGuestMode(true);
        }}
        onSelectFeature={(featureId) => {
          trackEvent('landing_feature_click', { feature: featureId });
          setIsGuestMode(true);
          setSelectedSchemeCode(null);
          setIsComparing(false);
          setIsAbout(false);
          if (featureId === 'screener') {
            setIsScreening(true);
            setIsPlanning(false);
            setIsAnalyzer(false);
          } else if (featureId === 'planner') {
            setIsPlanning(true);
            setIsScreening(false);
            setIsAnalyzer(false);
          } else if (featureId === 'xray') {
            setIsAnalyzer(true);
            setIsPlanning(false);
            setIsScreening(false);
          }
        }}
        onAbout={() => {
          setIsGuestMode(true);
          setIsAbout(true);
          setIsScreening(false);
          setIsPlanning(false);
          setIsAnalyzer(false);
          setIsComparing(false);
          setSelectedSchemeCode(null);
        }}
      />
    );
  }

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
        <button 
          onClick={() => {
            setIsGuestMode(false);
            handleBackToList();
            setIsSidebarOpen(false);
          }}
          className="h-16 px-6 flex items-center gap-3 shrink-0 text-left hover:opacity-85 transition-opacity focus:outline-none group cursor-pointer"
          title="Return to Home & Overview"
        >
          <div className="w-8 h-8 rounded-xl bg-finance-primary flex shrink-0 items-center justify-center shadow-sm group-hover:scale-105 transition-transform">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
            </svg>
          </div>
          <h1 className="text-lg font-bold text-slate-900 tracking-tight whitespace-nowrap">
            FundSense.AI
          </h1>
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200/80 leading-none">
            Beta
          </span>
        </button>

        <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
          {!isAuthenticated && (
            <NavButton 
              title="Home & Overview" 
              isActive={false}
              onClick={() => setIsGuestMode(false)}
              iconPath="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
            />
          )}
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
          {isAuthenticated && (
            <NavButton 
              title="My Profile" 
              isActive={isProfile} 
              onClick={() => {
                setIsProfile(true);
                setIsAbout(false);
                setIsAnalyzer(false);
                setIsPlanning(false);
                setIsScreening(false);
                setIsComparing(false);
                setSelectedSchemeCode(null);
              }} 
              iconPath="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" 
            />
          )}
          <NavButton 
            title="About FundSense.AI" 
            isActive={isAbout} 
            onClick={() => { setIsAbout(true); setIsAnalyzer(false); setIsPlanning(false); setIsScreening(false); setIsComparing(false); setIsProfile(false); setSelectedSchemeCode(null); }}
            iconPath="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
          />
        </nav>
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
          
          <button 
            onClick={() => {
              setIsGuestMode(false);
              handleBackToList();
            }}
            className="flex items-center gap-2 min-w-0 text-left hover:opacity-80 transition-opacity focus:outline-none cursor-pointer"
            title="Return to Home & Overview"
          >
            <h2 className="text-[15px] font-semibold text-slate-800 truncate">
              {currentViewTitle}
            </h2>
            <span className="inline-flex md:hidden items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200/80 leading-none shrink-0">
              Beta
            </span>
          </button>

          <div className="ml-auto flex items-center gap-3">
            {!isAuthenticated && isGuestMode && (
              <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-700 border border-amber-200/80">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse"></span>
                Guest Demo Mode
              </span>
            )}
            <UserNav onOpenProfile={() => {
              setIsProfile(true);
              setIsAbout(false);
              setIsAnalyzer(false);
              setIsPlanning(false);
              setIsScreening(false);
              setIsComparing(false);
              setSelectedSchemeCode(null);
            }} />
          </div>
        </header>

        {/* Main Scrollable Area */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden relative flex flex-col justify-between">
          <div className="max-w-[1152px] w-full mx-auto px-4 md:px-6 lg:px-8 py-6 md:py-8 flex flex-col flex-1">
            <ErrorBoundary onReset={handleBackToList}>
              {isProfile ? (
                <ProfileModal mode="page" onBack={handleBackToList} />
              ) : isAbout ? (
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
            </ErrorBoundary>
          </div>

          {/* In-App Responsive Educational & Compliance Footer */}
          <footer className="mt-auto border-t border-slate-200/70 bg-white/70 backdrop-blur-sm py-4 px-4 sm:px-8 shrink-0">
            <div className="max-w-[1152px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 text-center sm:text-left">
              <div className="flex items-center justify-center sm:justify-start gap-2">
                <button 
                  onClick={() => {
                    setIsGuestMode(false);
                    handleBackToList();
                  }}
                  className="flex items-center gap-2 hover:opacity-80 transition-opacity cursor-pointer text-left focus:outline-none"
                  title="Return to Home & Overview"
                >
                  <span className="font-semibold text-slate-800">FundSense.AI</span>
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200/80 leading-none">
                    Beta
                  </span>
                </button>
                <span className="text-slate-300 hidden sm:inline">•</span>
                <span 
                  className="text-slate-400 hidden sm:inline"
                  title={formatFullSyncTime(syncStatus) ? `Last synchronized on: ${formatFullSyncTime(syncStatus)}` : undefined}
                >
                  {formatSyncDate(syncStatus) ? `Last sync: ${formatSyncDate(syncStatus)}` : 'AMFI Daily NAV Verified'}
                </span>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-slate-400">
                <span>Educational Tool • Not SEBI Advice</span>
                <button 
                  onClick={() => {
                    setIsAbout(true);
                    setIsAnalyzer(false);
                    setIsPlanning(false);
                    setIsScreening(false);
                    setIsComparing(false);
                    setSelectedSchemeCode(null);
                  }}
                  className="text-finance-primary hover:underline font-medium"
                >
                  About & Methodology
                </button>
              </div>
            </div>
          </footer>
        </main>
      </div>

      <ProfileModal
        mode="modal"
        isOpen={isProfileOpen && !isProfile}
        onClose={() => setIsProfileOpen(false)}
      />
    </div>
  );
}

export default App;
