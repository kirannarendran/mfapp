import React from 'react';

const FeatureCard = ({ icon, title, description }) => (
  <div className="bg-white p-6 rounded-2xl border border-slate-200/60 shadow-sm hover:shadow-md transition-shadow">
    <div className="w-12 h-12 bg-finance-primary/10 rounded-xl flex items-center justify-center mb-4 text-finance-primary">
      {icon}
    </div>
    <h3 className="text-lg font-bold text-slate-800 mb-2">{title}</h3>
    <p className="text-slate-600 text-sm leading-relaxed">{description}</p>
  </div>
);

const AboutPage = () => {
  return (
    <div className="max-w-4xl mx-auto pb-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
      
      {/* Hero Section */}
      <div className="text-center mb-16 pt-8">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-finance-primary text-white shadow-lg mb-6">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
          </svg>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
          Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-finance-primary to-blue-600">FundSense.AI</span>
        </h1>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          An advanced intelligence platform for mutual fund investors. We provide the analytics and risk-adjusted metrics you need to evaluate funds beyond just their raw returns, empowering you to make more informed investment decisions.
        </p>
      </div>

      {/* Grid Features */}
      <div className="grid md:grid-cols-2 gap-6 mb-16">
        <FeatureCard 
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
          }
          title="Risk-First Screener"
          description="Filter funds by Sharpe ratio, Sortino ratio, Alpha, and Beta. Identify the managers actually delivering value over the benchmark."
        />
        <FeatureCard 
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
          }
          title="AI Wealth Planner"
          description="Tell our AI your goals, horizon, and risk tolerance. We'll construct a personalized, well-diversified portfolio using top-tier funds."
        />
        <FeatureCard 
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 21h7a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v11m0 5l4-4m-4 4l-4-4"></path>
            </svg>
          }
          title="Portfolio X-Ray"
          description="Upload your current holdings. Our AI diagnoses laggards, identifies hidden risks, and suggests tax-efficient ways to rebalance."
        />
        <FeatureCard 
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"></path>
            </svg>
          }
          title="Head-to-Head Comparisons"
          description="Stack up to 3 funds side-by-side to compare historical performance, expense ratios, and risk metrics dynamically."
        />
      </div>

      {/* Mission statement */}
      <div className="bg-slate-900 rounded-3xl p-8 md:p-12 text-center text-white shadow-xl">
        <h2 className="text-2xl font-bold mb-4">The Philosophy</h2>
        <p className="text-slate-300 max-w-2xl mx-auto mb-6 text-sm md:text-base leading-relaxed">
          Too many investors chase last year's highest returns, ignoring the volatility and risk taken to achieve them. FundSense.AI was built to bridge the gap between institutional-grade risk analysis and everyday retail investing. We don't just show you how much a fund grew; we show you how safely it got there.
        </p>
      </div>

      {/* Legal & SEBI Disclaimer */}
      <div className="mt-12 pt-8 border-t border-slate-200/60 text-center">
        <p className="text-xs text-slate-500 max-w-4xl mx-auto leading-relaxed">
          <strong className="text-slate-700">Legal Disclaimer:</strong> FundSense.AI is a technology and analytics platform, not a SEBI-registered Investment Advisor or Research Analyst. Mutual fund investments are subject to market risks; please read all scheme-related documents carefully before investing. AI-generated insights, rankings, and portfolio analyses are derived from historical data and mathematical models. They are provided for educational and informational purposes only and do not constitute financial advice, recommendations, or guarantees of future returns. Always consult with a qualified financial advisor before making investment decisions.
        </p>
      </div>
      
    </div>
  );
};

export default AboutPage;
