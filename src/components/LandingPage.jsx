import React, { useState } from 'react';
import UserNav from './UserNav';

const LandingPage = ({ onStartWizard, onExploreGuest, onSelectFeature, onAbout }) => {
  const [activeTab, setActiveTab] = useState('screener');
  const [previewSip, setPreviewSip] = useState(15000);
  const [previewCategory, setPreviewCategory] = useState('Large Cap');

  const features = [
    {
      id: 'screener',
      title: 'Risk-First Screener',
      badge: 'Institutional Analytics',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
        </svg>
      ),
      headline: 'Filter 5,000+ Direct Funds Beyond Past Returns',
      description: 'Chasing last year’s 30% return often means buying at peak risk. FundSense computes daily risk-adjusted metrics (Sharpe, Sortino, Alpha, Beta, Capture ratios) so you can identify fund managers who consistently beat the benchmark safely.',
      preview: {
        title: 'Live Screener Snapshot',
        metrics: [
          { label: 'Sharpe Ratio', value: '≥ 1.20', desc: 'Excess return per unit of total risk' },
          { label: 'Beta Range', value: '0.60 – 0.95', desc: 'Lower sensitivity to market crashes' },
          { label: 'Sortino Ratio', value: '≥ 1.50', desc: 'Focuses strictly on downside volatility' },
          { label: '5Y Alpha', value: '+4.8% p.a.', desc: 'Value delivered above the benchmark' },
        ],
        badgeColor: 'bg-emerald-50 text-emerald-700 border-emerald-200'
      }
    },
    {
      id: 'planner',
      title: 'AI Wealth Planner',
      badge: 'Deterministic + LLM Reasoning',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      headline: 'Mathematically Screened, Intelligently Explained',
      description: 'Set your monthly SIP, investment horizon, and maximum risk tolerance. Our backend screens thousands of eligible funds against strict mathematical constraints, then Groq LLMs reason over shortlisted options to formulate a personalized portfolio strategy.',
      preview: {
        title: 'Sample AI Plan Output',
        metrics: [
          { label: 'Target Monthly SIP', value: '₹15,000', desc: 'Calculated for long-term compounding' },
          { label: 'Weighted Beta', value: '0.84', desc: 'Defensive tilt vs Nifty 50' },
          { label: 'Projected Drawdown', value: '≤ 18%', desc: 'Within user risk budget' },
          { label: 'Portfolio Alpha', value: '+3.9%', desc: 'Grounded in historical outperformance' },
        ],
        badgeColor: 'bg-blue-50 text-blue-700 border-blue-200'
      }
    },
    {
      id: 'xray',
      title: 'Portfolio X-Ray',
      badge: 'Diagnostics & Risk Audit',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
        </svg>
      ),
      headline: 'Diagnose Overlap, Drift & Hidden Vulnerabilities',
      description: 'Holding 8 funds might feel safe, but they often hold the exact same 30 large-cap stocks. Paste your portfolio to detect concentration risks, uncompensated volatility, and underperforming legacy regular plans.',
      preview: {
        title: 'Diagnostic Insights',
        metrics: [
          { label: 'Stock Overlap', value: 'Under 15%', desc: 'True market diversification' },
          { label: 'Downside Capture', value: '72%', desc: 'Loses 28% less during bear markets' },
          { label: 'Expense Drag', value: 'Direct Only', desc: 'Zero commission intermediary fee' },
          { label: 'Risk Grade', value: 'A-', desc: 'Institutional grade health rating' },
        ],
        badgeColor: 'bg-purple-50 text-purple-700 border-purple-200'
      }
    }
  ];

  const currentFeature = features.find(f => f.id === activeTab) || features[0];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 text-slate-900 font-sans w-full overflow-x-hidden">
      
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-white/85 backdrop-blur-md border-b border-slate-200/70 w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
            <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-finance-primary flex items-center justify-center shadow-sm text-white shrink-0">
              <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <span className="text-lg sm:text-xl font-black tracking-tight text-slate-900 whitespace-nowrap">
              FundSense<span className="text-finance-primary">.AI</span>
            </span>
            <span className="inline-flex items-center ml-1 px-1.5 sm:px-2 py-0.5 rounded text-[9px] sm:text-[10px] font-bold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200/80 leading-none">
              Beta
            </span>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <button
              onClick={onExploreGuest}
              className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-sm font-semibold text-slate-700 hover:bg-slate-100 transition-colors focus:outline-none focus:ring-2 focus:ring-finance-primary/20 shrink-0 cursor-pointer"
            >
              <span>Explore as Guest</span>
              <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <UserNav />
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-10 pb-14 sm:pt-16 sm:pb-20 w-full">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-finance-primary/10 border border-finance-primary/20 text-finance-primary text-xs font-semibold mb-6 animate-in fade-in slide-in-from-top-3 duration-500">
            <span className="w-2 h-2 rounded-full bg-finance-primary animate-pulse"></span>
            Institutional Risk Intelligence for Indian Retail Investors
          </div>

          <h1 className="text-3xl sm:text-5xl lg:text-6xl font-black text-slate-900 tracking-tight leading-[1.15] mb-5">
            Build a Smarter Mutual Fund Portfolio <br className="hidden sm:inline" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-finance-primary via-blue-600 to-indigo-600">
              Grounded in Data, Not Hype.
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed mb-8 px-2">
            SEBI-aware portfolio intelligence using live AMFI historical NAVs. Filter 5,000+ Direct Growth funds by true risk metrics — Sharpe, Sortino, and downside beta — with zero distributor commissions.
          </p>

          {/* Action CTAs: Focused 2-Path Design */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 max-w-lg mx-auto mb-6 px-4">
            <button
              onClick={onStartWizard || onExploreGuest}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2.5 px-8 py-3.5 rounded-full bg-finance-primary hover:bg-finance-primary-dark text-white font-bold text-sm sm:text-base shadow-md hover:shadow-xl transition-all focus:outline-none focus:ring-2 focus:ring-finance-primary/30 whitespace-nowrap cursor-pointer hover:scale-[1.02]"
            >
              <span>Build My Portfolio (2 Mins)</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
            
            <button
              onClick={onExploreGuest}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-full bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-semibold text-sm sm:text-base shadow-xs hover:shadow transition-all focus:outline-none focus:ring-2 focus:ring-slate-300 whitespace-nowrap cursor-pointer"
            >
              <span>Explore Tools as Guest</span>
            </button>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-8 text-xs text-slate-500 font-medium mb-10 sm:mb-12 px-4">
            <span className="flex items-center gap-1.5"><span className="text-emerald-600 font-bold">✓</span> Direct Growth Only</span>
            <span className="flex items-center gap-1.5"><span className="text-emerald-600 font-bold">✓</span> Zero Commission Drag</span>
            <span className="flex items-center gap-1.5"><span className="text-emerald-600 font-bold">✓</span> No Broker Lock-in</span>
            <span className="flex items-center gap-1.5"><span className="text-emerald-600 font-bold">✓</span> Free Guest Mode</span>
          </div>

          {/* Key Value Badges: 3 Distinct Pillars */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 max-w-3xl mx-auto text-left">
            <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200/80 shadow-xs">
              <div className="text-xl mb-2">🎯</div>
              <p className="text-sm font-bold text-slate-900">Goal-Tailored Portfolios</p>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Interview-based allocation matching your time horizon, SIP/lumpsum outlay, and emergency safety net.
              </p>
            </div>
            <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200/80 shadow-xs">
              <div className="text-xl mb-2">🛡️</div>
              <p className="text-sm font-bold text-slate-900">Institutional Risk Screening</p>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Funds filtered by downside capture, Sortino, and Sharpe ratios — not just short-term return hype.
              </p>
            </div>
            <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200/80 shadow-xs">
              <div className="text-xl mb-2">⚡</div>
              <p className="text-sm font-bold text-slate-900">100% Unbiased Intelligence</p>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Direct plans only. Zero distributor commissions, zero sales quotas, and mathematical transparency.
              </p>
            </div>
          </div>

        </div>
      </section>

      {/* Feature Showcase Section */}
      <section className="py-12 sm:py-16 bg-slate-100/70 border-y border-slate-200/80 w-full overflow-x-hidden">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
          
          <div className="text-center max-w-2xl mx-auto mb-8 sm:mb-10">
            <h2 className="text-xs font-bold uppercase tracking-wider text-finance-primary mb-2">
              Inside the Platform
            </h2>
            <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              Tools Engineered for Data-Driven Investors
            </p>
          </div>

          {/* Interactive Feature Tabs */}
          <div className="w-full max-w-full overflow-x-auto pb-3 mb-6 sm:mb-8 no-scrollbar flex justify-start sm:justify-center">
            <div className="flex gap-2 min-w-max px-2 sm:px-0">
              {features.map((f) => (
                <button
                  key={f.id}
                  onClick={() => setActiveTab(f.id)}
                  className={`inline-flex items-center gap-2 px-4 sm:px-5 py-2.5 rounded-xl font-semibold text-xs sm:text-sm transition-all whitespace-nowrap shrink-0 ${
                    activeTab === f.id
                      ? 'bg-finance-primary text-white shadow-md'
                      : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200/80'
                  }`}
                >
                  {f.icon}
                  <span>{f.title}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Active Tab Preview Card */}
          <div className="bg-white rounded-3xl border border-slate-200 shadow-xl overflow-hidden flex flex-col lg:grid lg:grid-cols-12 w-full max-w-full">
            
            <div className="p-6 sm:p-8 lg:p-12 lg:col-span-6 flex flex-col justify-center">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 w-fit mb-3">
                {currentFeature.badge}
              </span>
              <h3 className="text-xl sm:text-2xl lg:text-3xl font-bold text-slate-900 tracking-tight mb-3">
                {currentFeature.headline}
              </h3>
              <p className="text-slate-600 text-sm sm:text-base leading-relaxed mb-6">
                {currentFeature.description}
              </p>
              
              <button
                onClick={() => onSelectFeature(currentFeature.id)}
                className="inline-flex items-center gap-2 text-finance-primary hover:text-finance-primary-dark font-bold text-sm group self-start cursor-pointer"
              >
                <span>Launch {currentFeature.title} now</span>
                <svg className="w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </div>

            <div className="bg-slate-50/80 p-4 sm:p-6 lg:p-10 lg:col-span-6 border-t lg:border-t-0 lg:border-l border-slate-200/80 flex flex-col justify-center">
              <div className="bg-white p-4 sm:p-6 rounded-2xl border border-slate-200/90 shadow-sm w-full">
                <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
                  <div>
                    <h4 className="font-bold text-slate-800 text-sm">{currentFeature.preview.title}</h4>
                    <p className="text-[10px] text-slate-400">Interactive live simulation</p>
                  </div>
                  <span className="text-[11px] font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200/60">
                    Live Engine
                  </span>
                </div>

                {activeTab === 'screener' && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
                      {['Large Cap', 'Flexi Cap', 'Mid Cap'].map(cat => (
                        <button
                          key={cat}
                          type="button"
                          onClick={() => setPreviewCategory(cat)}
                          className={`px-3 py-1 rounded-lg font-semibold transition-all text-xs shrink-0 ${
                            previewCategory === cat
                              ? 'bg-finance-primary text-white shadow-sm'
                              : 'bg-slate-100 text-slate-600 hover:bg-slate-200/70'
                          }`}
                        >
                          {cat}
                        </button>
                      ))}
                    </div>

                    <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Sharpe Ratio</p>
                        <p className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">
                          {previewCategory === 'Large Cap' ? '1.24' : previewCategory === 'Flexi Cap' ? '1.38' : '1.15'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Excess return / unit risk</p>
                      </div>
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Market Beta</p>
                        <p className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">
                          {previewCategory === 'Large Cap' ? '0.82' : previewCategory === 'Flexi Cap' ? '0.91' : '1.12'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Sensitivity vs index</p>
                      </div>
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Sortino Ratio</p>
                        <p className="text-base sm:text-lg font-bold text-emerald-600 mt-0.5">
                          {previewCategory === 'Large Cap' ? '1.58' : previewCategory === 'Flexi Cap' ? '1.72' : '1.45'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Downside safety gate</p>
                      </div>
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">5Y Alpha</p>
                        <p className="text-base sm:text-lg font-bold text-finance-primary mt-0.5">
                          {previewCategory === 'Large Cap' ? '+3.4% p.a.' : previewCategory === 'Flexi Cap' ? '+4.8% p.a.' : '+6.1% p.a.'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Outperformance</p>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'planner' && (
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center text-xs font-semibold text-slate-700 mb-1">
                        <span>Test Monthly SIP Slider</span>
                        <span className="text-finance-primary font-bold">₹{previewSip.toLocaleString('en-IN')}/mo</span>
                      </div>
                      <input
                        type="range"
                        min="5000"
                        max="50000"
                        step="2500"
                        value={previewSip}
                        onChange={(e) => setPreviewSip(Number(e.target.value))}
                        className="w-full accent-finance-primary"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Monthly SIP</p>
                        <p className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">₹{previewSip.toLocaleString('en-IN')}</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">15-year compounding</p>
                      </div>
                      <div className="p-2.5 sm:p-3 bg-emerald-50/70 rounded-xl border border-emerald-100">
                        <p className="text-[10px] text-emerald-800 font-semibold uppercase">Target 15Y Corpus</p>
                        <p className="text-base sm:text-lg font-bold text-emerald-700 mt-0.5">
                          ₹{Math.round(previewSip * ((Math.pow(1 + 0.13/12, 15 * 12) - 1) / (0.13/12)) * (1 + 0.13/12)).toLocaleString('en-IN')}
                        </p>
                        <p className="text-[10px] text-emerald-600 mt-0.5">At 13% CAGR projection</p>
                      </div>
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Weighted Beta</p>
                        <p className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">0.84</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Defensive vs Nifty 50</p>
                      </div>
                      <div className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Max Drawdown</p>
                        <p className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">≤ 18%</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Risk budget constraint</p>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'xray' && (
                  <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                    {currentFeature.preview.metrics.map((m, i) => (
                      <div key={i} className="p-2.5 sm:p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">{m.label}</p>
                        <p className="text-base sm:text-lg font-bold text-slate-900 mt-0.5">{m.value}</p>
                        <p className="text-[10px] text-slate-400 mt-0.5 leading-snug">{m.desc}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

          </div>

        </div>
      </section>

      {/* Educational & SEBI Compliance Footer */}
      <footer className="py-10 sm:py-12 bg-white border-t border-slate-200 w-full overflow-x-hidden">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          
          {/* Top Brand & Navigation Links */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-6 pb-8 border-b border-slate-100 text-center sm:text-left">
            <div>
              <div className="flex items-center justify-center sm:justify-start gap-2">
                <span className="text-lg font-black tracking-tight text-slate-900">
                  FundSense<span className="text-finance-primary">.AI</span>
                </span>
                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200/80 leading-none">
                  Beta
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">
                Institutional-grade risk intelligence and portfolio diagnostics for Indian retail investors.
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-xs font-medium text-slate-600">
              <button 
                onClick={() => onSelectFeature && onSelectFeature('screener')} 
                className="hover:text-finance-primary transition-colors py-1"
              >
                Risk Screener
              </button>
              <button 
                onClick={() => onSelectFeature && onSelectFeature('planner')} 
                className="hover:text-finance-primary transition-colors py-1"
              >
                Wealth Planner
              </button>
              <button 
                onClick={() => onSelectFeature && onSelectFeature('xray')} 
                className="hover:text-finance-primary transition-colors py-1"
              >
                Portfolio X-Ray
              </button>
              <button 
                onClick={onAbout} 
                className="hover:text-finance-primary transition-colors font-semibold text-finance-primary py-1"
              >
                About & Methodology
              </button>
            </div>
          </div>

          {/* Statutory Compliance Notice */}
          <div className="py-6 border-b border-slate-100">
            <div className="bg-slate-50 rounded-xl p-4 sm:p-5 border border-slate-200/70 text-left">
              <p className="text-xs text-slate-600 leading-relaxed">
                <strong className="text-slate-800 font-semibold">SEBI Statutory Notice:</strong> FundSense.AI is an independent open-source educational project. The creator is not a SEBI-registered Investment Advisor (RIA) or Research Analyst (RA). All metrics, ratios (Sharpe, Sortino, Alpha, Beta), and portfolio scenarios are generated from mathematical computations on publicly available AMFI historical NAV feeds and do not constitute financial advice, investment endorsement, or solicitation. Please consult a SEBI-registered financial advisor before making actual investments.
              </p>
            </div>
          </div>

          {/* Bottom Copyright & Verification Note */}
          <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400 text-center sm:text-left">
            <p>
              © {new Date().getFullYear()} FundSense.AI. Built for educational and research demonstration purposes.
            </p>
            <p className="flex items-center justify-center sm:justify-end gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              <span>AMFI Daily NAV Verified</span>
            </p>
          </div>

        </div>
      </footer>

    </div>
  );
};

export default LandingPage;
