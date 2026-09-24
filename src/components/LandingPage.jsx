import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import UserNav from './UserNav';

const LandingPage = ({ onExploreGuest, onSelectFeature }) => {
  const { isAuthenticated, user } = useAuth();
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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 text-slate-900 font-sans">
      
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-white/85 backdrop-blur-md border-b border-slate-200/70">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-finance-primary flex items-center justify-center shadow-sm text-white">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <span className="text-xl font-black tracking-tight text-slate-900">
              FundSense<span className="text-finance-primary">.AI</span>
            </span>
            <span className="hidden sm:inline-flex items-center ml-2 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200/80">
              Beta
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onExploreGuest}
              className="inline-flex items-center gap-1 px-2.5 sm:px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold text-slate-700 hover:bg-slate-100 transition-colors focus:outline-none focus:ring-2 focus:ring-finance-primary/20 shrink-0"
            >
              <span><span className="hidden sm:inline">Explore as </span>Guest</span>
              <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <UserNav />
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-12 pb-20 sm:pt-16 sm:pb-28">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-finance-primary/10 border border-finance-primary/20 text-finance-primary text-xs font-semibold mb-6 animate-in fade-in slide-in-from-top-3 duration-500">
            <span className="w-2 h-2 rounded-full bg-finance-primary animate-pulse"></span>
            Institutional Risk Intelligence for Indian Retail Investors
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black text-slate-900 tracking-tight leading-[1.15] mb-6">
            Evaluate Mutual Funds <br className="hidden sm:inline" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-finance-primary to-blue-600">
              Beyond Raw Returns.
            </span>
          </h1>

          <p className="text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed mb-10">
            A fund returning 25% with severe drawdowns is completely different from one making 22% with half the volatility. Explore 5,000+ Direct Growth funds using live NAV data and risk-adjusted metrics.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 max-w-md mx-auto mb-4">
            <div className="w-full sm:w-auto flex justify-center">
              <UserNav />
            </div>
            
            <button
              onClick={onExploreGuest}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-slate-900 hover:bg-slate-800 text-white font-semibold text-sm shadow-md hover:shadow-lg transition-all focus:outline-none focus:ring-2 focus:ring-slate-900/30"
            >
              <span>Explore Live Demo as Guest</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </button>
          </div>

          <p className="text-xs text-slate-400 mb-12 font-medium flex items-center justify-center gap-2 flex-wrap">
            <span>⚡ Instant Free Access</span>
            <span className="hidden sm:inline">•</span>
            <span>No Broker Login Required</span>
            <span className="hidden sm:inline">•</span>
            <span>Zero Spam</span>
          </p>

          {/* Key Value Badges */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto text-left">
            <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm">
              <p className="text-2xl font-black text-slate-900">5,000+</p>
              <p className="text-xs text-slate-500 font-medium mt-0.5">Direct Growth Funds</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm">
              <p className="text-2xl font-black text-slate-900">10M+</p>
              <p className="text-xs text-slate-500 font-medium mt-0.5">Historical NAV Records</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm">
              <p className="text-2xl font-black text-finance-primary">Daily</p>
              <p className="text-xs text-slate-500 font-medium mt-0.5">AMFI Sync &amp; Ratios</p>
            </div>
            <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-sm">
              <p className="text-2xl font-black text-emerald-600">SEBI-Aware</p>
              <p className="text-xs text-slate-500 font-medium mt-0.5">Educational Purpose</p>
            </div>
          </div>

        </div>
      </section>

      {/* Feature Showcase Section */}
      <section className="py-16 bg-slate-100/70 border-y border-slate-200/80">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-xs font-bold uppercase tracking-wider text-finance-primary mb-2">
              Inside the Platform
            </h2>
            <p className="text-3xl font-extrabold text-slate-900 tracking-tight">
              Tools Engineered for Data-Driven Investors
            </p>
          </div>

          {/* Interactive Feature Tabs */}
          <div className="flex justify-center gap-2 mb-10 overflow-x-auto pb-2">
            {features.map((f) => (
              <button
                key={f.id}
                onClick={() => setActiveTab(f.id)}
                className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all whitespace-nowrap ${
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

          {/* Active Tab Preview Card */}
          <div className="bg-white rounded-3xl border border-slate-200 shadow-xl overflow-hidden grid lg:grid-cols-12">
            
            <div className="p-8 sm:p-12 lg:col-span-6 flex flex-col justify-center">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 w-fit mb-4">
                {currentFeature.badge}
              </span>
              <h3 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mb-4">
                {currentFeature.headline}
              </h3>
              <p className="text-slate-600 text-base leading-relaxed mb-8">
                {currentFeature.description}
              </p>
              
              <button
                onClick={() => {
                  onSelectFeature(currentFeature.id);
                  onExploreGuest();
                }}
                className="inline-flex items-center gap-2 text-finance-primary hover:text-finance-primary-dark font-bold text-sm group"
              >
                <span>Launch {currentFeature.title} now</span>
                <svg className="w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </div>

            <div className="bg-slate-50/80 p-6 sm:p-10 lg:col-span-6 border-t lg:border-t-0 lg:border-l border-slate-200/80 flex flex-col justify-center">
              <div className="bg-white p-6 rounded-2xl border border-slate-200/90 shadow-sm">
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
                          className={`px-3 py-1 rounded-lg font-semibold transition-all text-xs ${
                            previewCategory === cat
                              ? 'bg-finance-primary text-white shadow-sm'
                              : 'bg-slate-100 text-slate-600 hover:bg-slate-200/70'
                          }`}
                        >
                          {cat}
                        </button>
                      ))}
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Sharpe Ratio</p>
                        <p className="text-lg font-bold text-slate-900 mt-0.5">
                          {previewCategory === 'Large Cap' ? '1.24' : previewCategory === 'Flexi Cap' ? '1.38' : '1.15'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Excess return / unit risk</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Market Beta</p>
                        <p className="text-lg font-bold text-slate-900 mt-0.5">
                          {previewCategory === 'Large Cap' ? '0.82' : previewCategory === 'Flexi Cap' ? '0.91' : '1.12'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Sensitivity vs benchmark</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Sortino Ratio</p>
                        <p className="text-lg font-bold text-emerald-600 mt-0.5">
                          {previewCategory === 'Large Cap' ? '1.58' : previewCategory === 'Flexi Cap' ? '1.72' : '1.45'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Downside safety gate</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">5Y Alpha</p>
                        <p className="text-lg font-bold text-finance-primary mt-0.5">
                          {previewCategory === 'Large Cap' ? '+3.4% p.a.' : previewCategory === 'Flexi Cap' ? '+4.8% p.a.' : '+6.1% p.a.'}
                        </p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Historical outperformance</p>
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

                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Monthly SIP</p>
                        <p className="text-lg font-bold text-slate-900 mt-0.5">₹{previewSip.toLocaleString('en-IN')}</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">15-year compounding</p>
                      </div>
                      <div className="p-3 bg-emerald-50/70 rounded-xl border border-emerald-100">
                        <p className="text-[10px] text-emerald-800 font-semibold uppercase">Target 15Y Corpus</p>
                        <p className="text-lg font-bold text-emerald-700 mt-0.5">
                          ₹{Math.round(previewSip * ((Math.pow(1 + 0.13/12, 15 * 12) - 1) / (0.13/12)) * (1 + 0.13/12)).toLocaleString('en-IN')}
                        </p>
                        <p className="text-[10px] text-emerald-600 mt-0.5">At 13% CAGR projection</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Weighted Beta</p>
                        <p className="text-lg font-bold text-slate-900 mt-0.5">0.84</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Defensive vs Nifty 50</p>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">Max Drawdown</p>
                        <p className="text-lg font-bold text-slate-900 mt-0.5">≤ 18%</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Risk budget constraint</p>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'xray' && (
                  <div className="grid grid-cols-2 gap-3">
                    {currentFeature.preview.metrics.map((m, i) => (
                      <div key={i} className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <p className="text-[10px] text-slate-500 font-semibold uppercase">{m.label}</p>
                        <p className="text-lg font-bold text-slate-900 mt-0.5">{m.value}</p>
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
      <footer className="py-14 bg-white border-t border-slate-200">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          
          <div className="inline-flex items-center gap-2 mb-4">
            <span className="text-lg font-bold text-slate-900">FundSense.AI</span>
            <span className="text-slate-300">|</span>
            <span className="text-sm text-slate-500">Free Open-Source Educational Tool</span>
          </div>

          <p className="text-xs text-slate-500 max-w-3xl mx-auto leading-relaxed mb-6">
            <strong className="text-slate-700">Statutory Notice:</strong> FundSense.AI is a personal educational and demonstration project. The developer is not a SEBI-registered Investment Advisor (RIA) or Research Analyst (RA). All metrics, ratios, and AI-generated suggestions are based on historical mathematical computations from AMFI daily NAV feeds and do not constitute financial advice, solicitation, or performance guarantees. Please consult a SEBI-registered Investment Advisor before making actual investment decisions.
          </p>

          <p className="text-xs text-slate-400">
            Sign-in stores only your name and email for personalization — no financial data is collected. See our About page for full privacy details. Built for educational and engineering demonstration purposes.
          </p>
        </div>
      </footer>

    </div>
  );
};

export default LandingPage;
