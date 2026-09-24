import React from 'react';

const OnboardingChoiceModal = ({ isOpen, onStartWizard, onExploreManual, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-md flex items-center justify-center p-4 sm:p-6 animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-white rounded-3xl shadow-2xl border border-slate-200/90 overflow-hidden">
        {/* Top Header */}
        <div className="px-6 sm:px-8 pt-8 pb-4 text-center">
          <div className="w-12 h-12 rounded-2xl bg-emerald-50 text-emerald-600 mx-auto flex items-center justify-center mb-3 text-2xl shadow-sm">
            ✨
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            Welcome to FundSense.AI
          </h2>
          <p className="text-sm text-slate-500 mt-1 max-w-md mx-auto">
            Your profile has been saved. How would you like to build your mutual fund investment strategy?
          </p>
        </div>

        {/* Choice Options Grid */}
        <div className="p-6 sm:p-8 pt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Option 1: Guided Interview (TurboTax style) */}
          <div 
            onClick={onStartWizard}
            className="group relative p-6 rounded-2xl border-2 border-finance-primary/30 hover:border-finance-primary bg-gradient-to-b from-blue-50/40 to-white hover:bg-blue-50/60 transition-all duration-200 cursor-pointer shadow-sm hover:shadow-md flex flex-col justify-between"
          >
            <div className="absolute top-4 right-4">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-finance-primary text-white shadow-xs">
                Recommended
              </span>
            </div>

            <div>
              <div className="w-10 h-10 rounded-xl bg-finance-primary/10 text-finance-primary flex items-center justify-center text-xl mb-4 group-hover:scale-110 transition-transform">
                🧭
              </div>
              <h3 className="text-base font-bold text-slate-900 mb-1.5 group-hover:text-finance-primary transition-colors">
                Guided Portfolio Builder
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed mb-4">
                Answer 5 simple life questions. No finance jargon required. Our institutional algorithm builds your personalized asset-allocated portfolio.
              </p>

              <ul className="text-[11px] text-slate-500 space-y-1.5 mb-6">
                <li className="flex items-center gap-1.5">
                  <span className="text-emerald-500 font-bold">✓</span> 2-Minute interactive interview
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="text-emerald-500 font-bold">✓</span> Real-world market crash stress test
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="text-emerald-500 font-bold">✓</span> Curated Direct Growth funds &amp; SIP plan
                </li>
              </ul>
            </div>

            <button
              type="button"
              onClick={onStartWizard}
              className="w-full py-2.5 px-4 rounded-xl bg-finance-primary group-hover:bg-blue-700 text-white font-semibold text-xs transition-colors shadow-sm text-center"
            >
              Start 2-Min Interview →
            </button>
          </div>

          {/* Option 2: Manual DIY Explorer */}
          <div 
            onClick={onExploreManual}
            className="group p-6 rounded-2xl border-2 border-slate-200/80 hover:border-slate-300 bg-white hover:bg-slate-50/70 transition-all duration-200 cursor-pointer shadow-sm hover:shadow-md flex flex-col justify-between"
          >
            <div>
              <div className="w-10 h-10 rounded-xl bg-slate-100 text-slate-700 flex items-center justify-center text-xl mb-4 group-hover:scale-110 transition-transform">
                🔍
              </div>
              <h3 className="text-base font-bold text-slate-900 mb-1.5 group-hover:text-slate-700 transition-colors">
                Explore On My Own
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed mb-4">
                For experienced and DIY investors. Manually research schemes, configure multi-metric filters, and compare side-by-side.
              </p>

              <ul className="text-[11px] text-slate-500 space-y-1.5 mb-6">
                <li className="flex items-center gap-1.5">
                  <span className="text-slate-400 font-bold">•</span> Browse all 5,144 Direct Growth schemes
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="text-slate-400 font-bold">•</span> Institutional Fund Screener with sorting
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="text-slate-400 font-bold">•</span> Head-to-head scheme comparison tool
                </li>
              </ul>
            </div>

            <button
              type="button"
              onClick={onExploreManual}
              className="w-full py-2.5 px-4 rounded-xl bg-slate-100 group-hover:bg-slate-200 text-slate-700 font-semibold text-xs transition-colors text-center"
            >
              Browse Fund Universe →
            </button>
          </div>
        </div>

        {/* Footer Note */}
        <div className="px-6 py-3.5 bg-slate-50/80 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
          <span>You can switch between guided and manual modes at any time.</span>
          <button
            type="button"
            onClick={onClose || onExploreManual}
            className="text-slate-400 hover:text-slate-600 font-medium cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
};

export default OnboardingChoiceModal;
