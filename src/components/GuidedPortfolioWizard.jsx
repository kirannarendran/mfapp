import React, { useState, useEffect } from 'react';
import { calculateSIPFutureValue } from '../utils/financialPlannerUtils';

const GOALS = [
  {
    id: 'wealth_creation',
    emoji: '🏖️',
    title: 'Wealth Creation & Compounding',
    subtitle: 'Maximize long-term capital growth and beat inflation',
    defaultHorizon: 15,
    defaultRisk: 'moderate',
  },
  {
    id: 'home',
    emoji: '🏠',
    title: 'Home Purchase / Down Payment',
    subtitle: 'Capital protection with steady multi-year appreciation',
    defaultHorizon: 5,
    defaultRisk: 'moderate',
  },
  {
    id: 'fire',
    emoji: '🚀',
    title: 'Early Financial Freedom (FIRE)',
    subtitle: 'Aggressive alpha engine to retire 10–15 years earlier',
    defaultHorizon: 10,
    defaultRisk: 'aggressive',
  },
  {
    id: 'education',
    emoji: '🎓',
    title: "Children's Education / Future",
    subtitle: 'Disciplined compounding for a non-negotiable life milestone',
    defaultHorizon: 12,
    defaultRisk: 'moderate',
  },
  {
    id: 'capital_preservation',
    emoji: '🛡️',
    title: 'Capital Shield & Steady Income',
    subtitle: 'Low volatility, crash-resilience & sovereign fixed income',
    defaultHorizon: 3,
    defaultRisk: 'conservative',
  },
  {
    id: 'short_term',
    emoji: '✈️',
    title: 'Short-Term Dream / Big Purchase',
    subtitle: 'Safe parking with better returns than a traditional savings account',
    defaultHorizon: 2,
    defaultRisk: 'conservative',
  },
];

const HORIZONS = [
  { years: 2, label: 'Under 3 Years', desc: 'Short-Term · Strict Capital Safety · Sovereign Debt & Liquid focus' },
  { years: 5, label: '3 to 7 Years', desc: 'Medium-Term · Balanced Hybrid & Large Cap · Controlled Drawdowns' },
  { years: 10, label: '7 to 15 Years', desc: 'Long-Term · Flexi Cap & Multi Cap · Compounding Alpha' },
  { years: 20, label: '15+ Years', desc: 'Multi-Decade · Aggressive Equity · Maximum Wealth Multiplier' },
];

const STRESS_SCENARIOS = [
  {
    id: 'panic',
    emoji: '🔴',
    reaction: "I'd panic and want to cash out immediately",
    description: "You prioritize sleep-at-night security over returns. We will enforce strict downside shields, capping drawdowns with sovereign debt.",
    riskProfile: 'conservative',
    maxDrawdown: 10,
  },
  {
    id: 'hold',
    emoji: '🟡',
    reaction: "I'd feel uneasy, but stay invested and wait it out",
    description: "You understand markets have cyclical fluctuations. We will balance equity upside with downside cushions.",
    riskProfile: 'moderate',
    maxDrawdown: 18,
  },
  {
    id: 'buy_more',
    emoji: '🟢',
    reaction: "I'd see it as a huge discount and invest more!",
    description: "You have strong emotional conviction. We will tilt heavily towards high-alpha equity compounding engines.",
    riskProfile: 'aggressive',
    maxDrawdown: 28,
  },
];

const GuidedPortfolioWizard = ({ onBack, onOpenAnalyzer, onSelectFund }) => {
  const [step, setStep] = useState(1);
  const totalSteps = 5;

  // User Responses State
  const [selectedGoal, setSelectedGoal] = useState('wealth_creation');
  const [horizonYears, setHorizonYears] = useState(15);
  const [contributionMode, setContributionMode] = useState('sip'); // 'sip', 'lump', 'both'
  const [monthlySIP, setMonthlySIP] = useState(10000);
  const [lumpSum, setLumpSum] = useState(50000);
  const [stressReaction, setStressReaction] = useState('hold');
  const [hasEmergencyFund, setHasEmergencyFund] = useState('yes');

  // Generation & Results State
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamSteps, setStreamSteps] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [error, setError] = useState(null);
  const [savedSuccess, setSavedSuccess] = useState(false);

  // Sync default horizon when goal changes
  const handleSelectGoal = (goalId) => {
    setSelectedGoal(goalId);
    const g = GOALS.find(item => item.id === goalId);
    if (g) {
      setHorizonYears(g.defaultHorizon);
      if (g.defaultRisk === 'conservative') setStressReaction('panic');
      else if (g.defaultRisk === 'aggressive') setStressReaction('buy_more');
      else setStressReaction('hold');
    }
  };

  const handleNext = () => {
    if (step < totalSteps) {
      setStep(step + 1);
    } else {
      generatePortfolio();
    }
  };

  const handlePrev = () => {
    if (step > 1) {
      setStep(step - 1);
    } else if (onBack) {
      onBack();
    }
  };

  // Portfolio Generation Call
  const generatePortfolio = async () => {
    setStep(6); // Step 6 = Generation & Results
    setIsGenerating(true);
    setStreamSteps([]);
    setPortfolio(null);
    setError(null);
    setSavedSuccess(false);

    const stress = STRESS_SCENARIOS.find(s => s.id === stressReaction) || STRESS_SCENARIOS[1];
    let riskProfile = stress.riskProfile;
    let maxDrawdown = stress.maxDrawdown;

    // Emergency fund safety adjustment
    if (hasEmergencyFund === 'not_yet' && riskProfile === 'aggressive') {
      riskProfile = 'moderate';
      maxDrawdown = 18;
    }

    const goalObj = GOALS.find(g => g.id === selectedGoal);

    try {
      const response = await fetch('/api/advisor/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          params: {
            goal: goalObj?.title || 'Wealth creation',
            horizonYears,
            monthlySIP: contributionMode === 'lump' ? 0 : monthlySIP,
            lumpSum: contributionMode === 'sip' ? 0 : lumpSum,
            riskProfile,
            maxDrawdownPct: maxDrawdown,
            hasEmergencyFund,
            fundCategory: horizonYears <= 3 ? 'Debt/Hybrid' : 'Any Equity',
            numberOfFunds: 4,
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to connect to recommendation engine.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = '';

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep trailing incomplete line in buffer

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              const dataStr = trimmed.slice(6).trim();
              if (!dataStr) continue;
              try {
                const data = JSON.parse(dataStr);
                if (data.type === 'step') {
                  setStreamSteps(prev => {
                    const idx = prev.findIndex(s => s.title === data.title);
                    if (idx !== -1) {
                      const copy = [...prev];
                      copy[idx] = data;
                      return copy;
                    }
                    return [...prev, data];
                  });
                } else if (data.type === 'result') {
                  setPortfolio(data.recommendation);
                } else if (data.type === 'error') {
                  setError(data.message);
                }
              } catch (e) {
                // Ignore parse errors on raw tokens
              }
            }
          }
        }
      }

      // Process any remaining data in buffer
      if (buffer.trim().startsWith('data: ')) {
        try {
          const data = JSON.parse(buffer.trim().slice(6).trim());
          if (data.type === 'result') {
            setPortfolio(data.recommendation);
          } else if (data.type === 'error') {
            setError(data.message);
          }
        } catch (_) {}
      }
    } catch (err) {
      console.error('[GuidedWizard] Error:', err);
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSavePortfolio = () => {
    if (!portfolio) return;
    try {
      localStorage.setItem('fundsense_target_portfolio', JSON.stringify({
        portfolio,
        answers: {
          selectedGoal,
          horizonYears,
          monthlySIP,
          lumpSum,
          contributionMode,
          stressReaction,
          hasEmergencyFund
        },
        savedAt: new Date().toISOString()
      }));
      setSavedSuccess(true);
      setTimeout(() => setSavedSuccess(false), 4000);
    } catch (e) {
      console.error('Failed to save target portfolio:', e);
    }
  };

  // Projected Value Calculation
  const estimatedReturnRate = portfolio?.portfolio_metrics?.weighted_cagr_percentage || (stressReaction === 'aggressive' ? 14 : stressReaction === 'moderate' ? 12 : 8.5);
  const effectiveMonthly = contributionMode === 'lump' ? 0 : monthlySIP;
  const { investedAmount, estimatedReturns, totalValue } = calculateSIPFutureValue(
    effectiveMonthly,
    horizonYears,
    estimatedReturnRate
  );

  return (
    <div className="max-w-4xl mx-auto w-full pb-20 animate-fade-in">
      {/* Top Header & Navigation Bar */}
      <div className="flex items-center justify-between gap-4 mb-6">
        {step === 6 ? (
          <button
            type="button"
            onClick={onBack}
            className="text-xs sm:text-sm font-bold text-finance-primary hover:text-blue-700 flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            ← Back to Fund Universe
          </button>
        ) : (
          <button
            type="button"
            onClick={handlePrev}
            className="text-xs sm:text-sm font-semibold text-finance-primary hover:text-blue-700 flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            ← {step === 1 ? 'Exit to Universe' : 'Previous Step'}
          </button>
        )}

        {step <= totalSteps ? (
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-500">
              Step {step} of {totalSteps}
            </span>
            <div className="flex gap-1.5">
              {[1, 2, 3, 4, 5].map((s) => (
                <div
                  key={s}
                  className={`h-2 rounded-full transition-all duration-300 ${
                    s === step
                      ? 'w-7 bg-finance-primary'
                      : s < step
                      ? 'w-2 bg-emerald-500'
                      : 'w-2 bg-slate-200'
                  }`}
                />
              ))}
            </div>
          </div>
        ) : (
          onBack && (
            <button
              type="button"
              onClick={onBack}
              className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 transition-colors cursor-pointer flex items-center gap-1"
            >
              <span>Back to Universe</span>
              <span>✕</span>
            </button>
          )
        )}

        {onBack && step <= totalSteps && (
          <button
            type="button"
            onClick={onBack}
            className="text-xs text-slate-400 hover:text-slate-600 font-medium cursor-pointer"
          >
            Skip Interview ✕
          </button>
        )}
      </div>

      {/* ──────────────── STEP 1: GOAL & MILESTONE ──────────────── */}
      {step === 1 && (
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-10 animate-fade-in">
          <div className="text-center max-w-lg mx-auto mb-8">
            <span className="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-blue-50 text-finance-primary border border-blue-100">
              Milestone Discovery
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
              What are you investing for?
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              Select your primary objective. Every portfolio should begin with a clear life outcome.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 mb-8">
            {GOALS.map((g) => (
              <button
                type="button"
                key={g.id}
                onClick={() => handleSelectGoal(g.id)}
                className={`p-5 rounded-2xl border text-left transition-all duration-150 cursor-pointer flex flex-col justify-between ${
                  selectedGoal === g.id
                    ? 'border-finance-primary bg-finance-primary/5 ring-2 ring-finance-primary/30 shadow-sm'
                    : 'border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/50'
                }`}
              >
                <div>
                  <div className="text-2xl mb-3">{g.emoji}</div>
                  <h3 className="text-sm font-bold text-slate-900 mb-1">{g.title}</h3>
                  <p className="text-xs text-slate-500 leading-relaxed">{g.subtitle}</p>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400 font-medium">
                  <span>Suggested Horizon</span>
                  <span className="font-semibold text-slate-700">{g.defaultHorizon} Years</span>
                </div>
              </button>
            ))}
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleNext}
              className="w-full sm:w-auto px-8 py-3 rounded-xl bg-finance-primary hover:bg-blue-700 text-white font-bold text-sm shadow-md shadow-finance-primary/20 transition-all cursor-pointer"
            >
              Continue to Time Horizon →
            </button>
          </div>
        </div>
      )}

      {/* ──────────────── STEP 2: TIME HORIZON ──────────────── */}
      {step === 2 && (
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-10 animate-fade-in">
          <div className="text-center max-w-lg mx-auto mb-8">
            <span className="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-emerald-50 text-emerald-700 border border-emerald-100">
              Duration &amp; Compounding
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
              When will you need this money?
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              Time is your greatest asset. Longer horizons allow us to harness high-conviction equity compounding.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 mb-8">
            {HORIZONS.map((h) => (
              <button
                type="button"
                key={h.years}
                onClick={() => setHorizonYears(h.years)}
                className={`p-5 rounded-2xl border text-left transition-all duration-150 cursor-pointer ${
                  horizonYears === h.years
                    ? 'border-finance-primary bg-finance-primary/5 ring-2 ring-finance-primary/30 shadow-sm'
                    : 'border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <h3 className="text-base font-bold text-slate-900">{h.label}</h3>
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700">
                    {h.years} Years
                  </span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">{h.desc}</p>
              </button>
            ))}
          </div>

          {/* Custom Horizon Slider */}
          <div className="p-5 rounded-2xl bg-slate-50 border border-slate-200/80 mb-8">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                Or Adjust Exact Duration
              </label>
              <span className="text-base font-bold text-finance-primary bg-white px-3 py-0.5 rounded-lg border border-slate-200 shadow-2xs">
                {horizonYears} Years
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="35"
              step="1"
              value={horizonYears}
              onChange={(e) => setHorizonYears(parseInt(e.target.value, 10))}
              className="w-full accent-finance-primary cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-medium">
              <span>1 Year (Immediate)</span>
              <span>15 Years (Long)</span>
              <span>35 Years (Multi-Decade)</span>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={handlePrev}
              className="px-5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
            >
              ← Back
            </button>
            <button
              type="button"
              onClick={handleNext}
              className="px-8 py-3 rounded-xl bg-finance-primary hover:bg-blue-700 text-white font-bold text-sm shadow-md shadow-finance-primary/20 transition-all cursor-pointer"
            >
              Continue to Contributions →
            </button>
          </div>
        </div>
      )}

      {/* ──────────────── STEP 3: CONTRIBUTION / CASH FLOW ──────────────── */}
      {step === 3 && (
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-10 animate-fade-in">
          <div className="text-center max-w-lg mx-auto mb-8">
            <span className="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-indigo-50 text-indigo-700 border border-indigo-100">
              Contribution Strategy
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
              How do you plan to invest?
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              SIPs enforce rupee-cost averaging, shielding you from having to time the market.
            </p>
          </div>

          {/* Mode Switcher */}
          <div className="grid grid-cols-3 gap-2.5 max-w-md mx-auto mb-8 p-1 bg-slate-100 rounded-2xl">
            {[
              { id: 'sip', label: 'Monthly SIP' },
              { id: 'lump', label: 'One-Time Lump Sum' },
              { id: 'both', label: 'Both' },
            ].map((m) => (
              <button
                type="button"
                key={m.id}
                onClick={() => setContributionMode(m.id)}
                className={`py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                  contributionMode === m.id
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Monthly SIP Amount */}
          {(contributionMode === 'sip' || contributionMode === 'both') && (
            <div className="mb-6 p-5 rounded-2xl bg-slate-50 border border-slate-200/80">
              <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                Monthly SIP Amount (₹)
              </label>
              <div className="flex flex-wrap gap-2 mb-3">
                {[2500, 5000, 10000, 25000, 50000].map((amt) => (
                  <button
                    type="button"
                    key={amt}
                    onClick={() => setMonthlySIP(amt)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all cursor-pointer ${
                      monthlySIP === amt
                        ? 'border-finance-primary bg-finance-primary text-white shadow-xs'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    ₹{amt.toLocaleString('en-IN')}
                  </button>
                ))}
              </div>
              <input
                type="number"
                min="500"
                step="500"
                value={monthlySIP}
                onChange={(e) => setMonthlySIP(Math.max(parseInt(e.target.value, 10) || 0, 0))}
                className="w-full px-4 py-2.5 text-base font-bold bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-finance-primary/20 text-slate-900"
                placeholder="Enter custom SIP amount"
              />
            </div>
          )}

          {/* Lump Sum Amount */}
          {(contributionMode === 'lump' || contributionMode === 'both') && (
            <div className="mb-8 p-5 rounded-2xl bg-slate-50 border border-slate-200/80">
              <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                Initial Lump Sum Investment (₹)
              </label>
              <div className="flex flex-wrap gap-2 mb-3">
                {[25000, 50000, 100000, 250000, 500000].map((amt) => (
                  <button
                    type="button"
                    key={amt}
                    onClick={() => setLumpSum(amt)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all cursor-pointer ${
                      lumpSum === amt
                        ? 'border-finance-primary bg-finance-primary text-white shadow-xs'
                        : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    ₹{amt.toLocaleString('en-IN')}
                  </button>
                ))}
              </div>
              <input
                type="number"
                min="5000"
                step="5000"
                value={lumpSum}
                onChange={(e) => setLumpSum(Math.max(parseInt(e.target.value, 10) || 0, 0))}
                className="w-full px-4 py-2.5 text-base font-bold bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-finance-primary/20 text-slate-900"
                placeholder="Enter custom lump sum amount"
              />
            </div>
          )}

          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={handlePrev}
              className="px-5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
            >
              ← Back
            </button>
            <button
              type="button"
              onClick={handleNext}
              className="px-8 py-3 rounded-xl bg-finance-primary hover:bg-blue-700 text-white font-bold text-sm shadow-md shadow-finance-primary/20 transition-all cursor-pointer"
            >
              Continue to Stress Test →
            </button>
          </div>
        </div>
      )}

      {/* ──────────────── STEP 4: REAL-WORLD MARKET CRASH STRESS TEST ──────────────── */}
      {step === 4 && (
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-10 animate-fade-in">
          <div className="text-center max-w-lg mx-auto mb-8">
            <span className="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-rose-50 text-rose-700 border border-rose-100">
              Behavioral Risk Stress Test
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
              The Real-World Crash Scenario
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-2 leading-relaxed">
              Suppose you invest <strong>₹2,00,000</strong> today. Six months from now, a global crisis strikes and your portfolio temporarily drops to <strong>₹1,50,000 (-25%)</strong>.
              <br />
              What is your genuine gut reaction?
            </p>
          </div>

          <div className="space-y-3.5 mb-8">
            {STRESS_SCENARIOS.map((s) => (
              <button
                type="button"
                key={s.id}
                onClick={() => setStressReaction(s.id)}
                className={`w-full p-5 rounded-2xl border text-left transition-all duration-150 cursor-pointer flex items-start gap-4 ${
                  stressReaction === s.id
                    ? 'border-finance-primary bg-finance-primary/5 ring-2 ring-finance-primary/30 shadow-sm'
                    : 'border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/50'
                }`}
              >
                <span className="text-2xl shrink-0 mt-0.5">{s.emoji}</span>
                <div className="flex-1">
                  <h3 className="text-sm sm:text-base font-bold text-slate-900 mb-1">{s.reaction}</h3>
                  <p className="text-xs text-slate-500 leading-relaxed">{s.description}</p>
                </div>
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={handlePrev}
              className="px-5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
            >
              ← Back
            </button>
            <button
              type="button"
              onClick={handleNext}
              className="px-8 py-3 rounded-xl bg-finance-primary hover:bg-blue-700 text-white font-bold text-sm shadow-md shadow-finance-primary/20 transition-all cursor-pointer"
            >
              Continue to Safety Net Check →
            </button>
          </div>
        </div>
      )}

      {/* ──────────────── STEP 5: EMERGENCY SAFETY NET CHECK ──────────────── */}
      {step === 5 && (
        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-10 animate-fade-in">
          <div className="text-center max-w-lg mx-auto mb-8">
            <span className="px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-amber-50 text-amber-700 border border-amber-100">
              Prudential Safety Check
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-2 tracking-tight">
              Do you have an emergency fund?
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              Do you currently hold 3 to 6 months of mandatory living expenses safely parked in a savings bank or fixed deposit?
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
            <button
              type="button"
              onClick={() => setHasEmergencyFund('yes')}
              className={`p-6 rounded-2xl border text-left transition-all duration-150 cursor-pointer ${
                hasEmergencyFund === 'yes'
                  ? 'border-finance-primary bg-finance-primary/5 ring-2 ring-finance-primary/30 shadow-sm'
                  : 'border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              <div className="text-3xl mb-3">🛡️</div>
              <h3 className="text-base font-bold text-slate-900 mb-1">Yes, I'm covered</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                100% of your contributions can be allocated toward your selected milestone and equity compounding.
              </p>
            </button>

            <button
              type="button"
              onClick={() => setHasEmergencyFund('not_yet')}
              className={`p-6 rounded-2xl border text-left transition-all duration-150 cursor-pointer ${
                hasEmergencyFund === 'not_yet'
                  ? 'border-finance-primary bg-finance-primary/5 ring-2 ring-finance-primary/30 shadow-sm'
                  : 'border-slate-200/80 bg-white hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              <div className="text-3xl mb-3">⚠️</div>
              <h3 className="text-base font-bold text-slate-900 mb-1">Not yet / In progress</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                Our algorithm will automatically embed a 15–20% allocation in low-duration Sovereign G-Sec/Liquid funds to protect your emergency buffer.
              </p>
            </button>
          </div>

          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={handlePrev}
              className="px-5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
            >
              ← Back
            </button>
            <button
              type="button"
              onClick={generatePortfolio}
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-finance-primary to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-bold text-sm shadow-lg shadow-finance-primary/25 transition-all cursor-pointer flex items-center gap-2"
            >
              <span>Build My Custom Portfolio</span>
              <span>✨</span>
            </button>
          </div>
        </div>
      )}

      {/* ──────────────── STEP 6: CALCULATION & PORTFOLIO RESULTS ──────────────── */}
      {step === 6 && (
        <div className="space-y-6 animate-fade-in">
          {/* Loading Animation Card */}
          {isGenerating && (
            <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-8 sm:p-12 text-center max-w-xl mx-auto">
              <div className="w-16 h-16 rounded-2xl bg-finance-primary/10 text-finance-primary mx-auto flex items-center justify-center mb-6 text-3xl animate-bounce">
                ⚙️
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">
                Synthesizing Your Portfolio
              </h3>
              <p className="text-xs text-slate-500 mb-6">
                Filtering 5,144 Direct Growth mutual funds against your institutional risk constraints...
              </p>

              {/* Progress Steps */}
              <div className="space-y-3 text-left max-w-md mx-auto">
                {streamSteps.map((st, i) => (
                  <div key={i} className="flex items-start gap-2.5 text-xs">
                    <span className="shrink-0 text-base">{st.icon || '✓'}</span>
                    <div>
                      <p className="font-semibold text-slate-800">{st.title}</p>
                      {st.detail && <p className="text-[11px] text-slate-500">{st.detail}</p>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && !isGenerating && (
            <div className="bg-rose-50 border border-rose-200 rounded-2xl p-6 text-center max-w-lg mx-auto">
              <div className="text-2xl mb-2">⚠️</div>
              <h4 className="text-sm font-bold text-rose-800 mb-1">Portfolio Synthesis Note</h4>
              <p className="text-xs text-rose-600 mb-4">{error}</p>
              <button
                type="button"
                onClick={generatePortfolio}
                className="btn-primary text-xs px-4 py-2"
              >
                Retry Generation
              </button>
            </div>
          )}

          {/* Results Presentation */}
          {portfolio && !isGenerating && (
            <div className="space-y-6">
              {/* Portfolio Banner */}
              <div className="bg-gradient-to-br from-slate-900 via-slate-850 to-slate-900 text-white rounded-3xl p-6 sm:p-8 shadow-xl border border-slate-800">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                  <div>
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-400/20 text-emerald-300 border border-emerald-400/30">
                      {portfolio.portfolio_summary?.risk_level || 'Custom'} Risk Profile
                    </span>
                    <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight mt-2 text-white">
                      {portfolio.portfolio_summary?.title || 'Your Custom Portfolio'}
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-300 mt-1 max-w-xl leading-relaxed">
                      {portfolio.portfolio_summary?.description}
                    </p>
                  </div>

                  <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/10 shrink-0 text-right sm:text-center min-w-[140px]">
                    <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold block">
                      Target Horizon
                    </span>
                    <span className="text-2xl font-black text-white block mt-0.5">
                      {horizonYears} Years
                    </span>
                    <span className="text-[11px] text-emerald-400 font-medium">
                      Est. {portfolio.portfolio_summary?.portfolio_metrics?.expected_return_range || '12% – 14%'}
                    </span>
                  </div>
                </div>

                {/* Key Metrics Strip */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-white/10 text-xs">
                  <div>
                    <span className="text-slate-400 block text-[11px]">Monthly Commitment</span>
                    <span className="font-bold text-white text-sm">₹{monthlySIP.toLocaleString('en-IN')}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[11px]">Target Objective</span>
                    <span className="font-bold text-white text-sm truncate block">{portfolio.portfolio_summary?.objective || 'Wealth Growth'}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[11px]">Weighted Beta</span>
                    <span className="font-bold text-white text-sm">{portfolio.portfolio_summary?.portfolio_metrics?.weighted_beta?.toFixed(2) || '0.85'}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[11px]">Rebalance Frequency</span>
                    <span className="font-bold text-white text-sm">{portfolio.portfolio_summary?.review_frequency || 'Annual'}</span>
                  </div>
                </div>
              </div>

              {/* Fund Allocation Cards */}
              <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-6 sm:p-8">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-base font-bold text-slate-900">Recommended Fund Allocation</h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Top-ranked Direct Growth mutual funds selected for optimal risk-adjusted alpha
                    </p>
                  </div>
                  <span className="text-xs font-bold text-slate-600 bg-slate-100 px-2.5 py-1 rounded-lg">
                    {portfolio.funds?.length || 0} Funds
                  </span>
                </div>

                {/* Visual Allocation Bar */}
                <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden flex mb-6">
                  {portfolio.funds?.map((fund, i) => {
                    const colors = ['bg-finance-primary', 'bg-emerald-500', 'bg-indigo-500', 'bg-amber-500', 'bg-rose-500'];
                    return (
                      <div
                        key={i}
                        style={{ width: `${fund.allocation_percentage}%` }}
                        className={`${colors[i % colors.length]} transition-all`}
                        title={`${fund.name || fund.scheme_name}: ${fund.allocation_percentage}%`}
                      />
                    );
                  })}
                </div>

                {/* Fund Cards List */}
                <div className="space-y-3.5 mb-8">
                  {portfolio.funds?.map((fund, i) => {
                    const fundSip = Math.round((monthlySIP * (fund.allocation_percentage || 0)) / 100);
                    return (
                      <div
                        key={i}
                        className="p-5 rounded-2xl border border-slate-200/90 bg-slate-50/50 hover:bg-slate-50 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2 flex-wrap mb-1">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-finance-primary/10 text-finance-primary">
                              {fund.allocation_percentage}% Allocation
                            </span>
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-200/80 text-slate-700">
                              {fund.category}
                            </span>
                            {fund.scheme_code && onSelectFund && (
                              <button
                                type="button"
                                onClick={() => onSelectFund(fund.scheme_code)}
                                className="text-[10px] text-finance-primary hover:underline font-semibold cursor-pointer"
                              >
                                View Fund Details ↗
                              </button>
                            )}
                          </div>
                          <h4 className="text-sm font-bold text-slate-900 mb-1">
                            {fund.name || fund.scheme_name}
                          </h4>
                          <p className="text-xs text-slate-600 leading-relaxed">
                            {fund.reason_detailed || fund.reason_short || fund.reason}
                          </p>
                        </div>

                        <div className="sm:text-right shrink-0 bg-white p-3 rounded-xl border border-slate-200/80 sm:border-0 sm:bg-transparent sm:p-0">
                          <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold block">
                            Suggested SIP
                          </span>
                          <span className="text-base font-extrabold text-slate-900 block mt-0.5">
                            ₹{fundSip.toLocaleString('en-IN')}/mo
                          </span>
                          {fund.metrics?.cagr_5y_percentage && (
                            <span className="text-[11px] font-semibold text-emerald-600">
                              {fund.metrics.cagr_5y_percentage}% 5Y CAGR
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Wealth Compounding Projection Card */}
                <div className="p-6 rounded-2xl bg-gradient-to-r from-blue-50 to-indigo-50/60 border border-blue-100 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div>
                    <span className="text-[11px] font-bold uppercase tracking-wider text-finance-primary">
                      {horizonYears}-Year Compounding Forecast
                    </span>
                    <h4 className="text-base font-bold text-slate-900 mt-0.5">
                      Estimated Wealth: <span className="text-finance-primary">₹{Math.round(totalValue).toLocaleString('en-IN')}</span>
                    </h4>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Based on regular monthly SIP of ₹{monthlySIP.toLocaleString('en-IN')} (Total Outlay: ₹{Math.round(investedAmount).toLocaleString('en-IN')})
                    </p>
                  </div>

                  <div className="text-left sm:text-right">
                    <span className="text-xs font-semibold text-emerald-700 bg-emerald-100 px-3 py-1 rounded-full inline-block">
                      +₹{Math.round(estimatedReturns).toLocaleString('en-IN')} Estimated Gains
                    </span>
                  </div>
                </div>
              </div>

              {/* Save Success Banner */}
              {savedSuccess && (
                <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-emerald-800 animate-fade-in shadow-xs">
                  <div className="flex items-center gap-2.5">
                    <span className="text-xl">🎉</span>
                    <div>
                      <p className="font-bold text-emerald-900">Target Portfolio Saved Successfully!</p>
                      <p className="text-emerald-700 text-[11px] mt-0.5">
                        Your target asset allocation is now stored under "My Profile". You can also test its resilience in Portfolio X-Ray.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {onBack && (
                      <button
                        type="button"
                        onClick={onBack}
                        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white font-bold rounded-xl transition-all shadow-xs cursor-pointer"
                      >
                        Go to Universe →
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setStep(1)}
                    className="px-4 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
                  >
                    ↺ Retake Interview
                  </button>
                  {onBack && (
                    <button
                      type="button"
                      onClick={onBack}
                      className="px-4 py-2.5 text-xs font-bold text-finance-primary hover:bg-finance-primary/10 rounded-xl transition-colors cursor-pointer"
                    >
                      ← Back to Fund Universe
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  {onOpenAnalyzer && (
                    <button
                      type="button"
                      onClick={() => {
                        const totalBasis = contributionMode === 'lump' ? lumpSum : (monthlySIP * 12);
                        const holdings = portfolio.funds?.map(f => ({
                          fundName: f.name || f.scheme_name,
                          value: Math.round((totalBasis * (f.allocation_percentage || 25)) / 100) || 50000
                        })) || [];
                        onOpenAnalyzer(holdings);
                      }}
                      className="px-5 py-2.5 rounded-xl border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5"
                    >
                      <span>🔬 Inspect in Portfolio X-Ray</span>
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={handleSavePortfolio}
                    className={`px-6 py-2.5 rounded-xl text-white text-xs font-bold shadow-md transition-all cursor-pointer flex items-center gap-1.5 ${
                      savedSuccess
                        ? 'bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/20'
                        : 'bg-finance-primary hover:bg-blue-700 shadow-finance-primary/20'
                    }`}
                  >
                    <span>{savedSuccess ? '✓ Portfolio Saved!' : '💾 Save Target Portfolio'}</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Fallback to prevent blank page when generator finishes without portfolio or error */}
          {!portfolio && !isGenerating && !error && (
            <div className="bg-white rounded-3xl border border-slate-200 shadow-sm p-8 sm:p-12 text-center max-w-lg mx-auto">
              <div className="w-12 h-12 rounded-2xl bg-amber-50 text-amber-600 mx-auto flex items-center justify-center mb-4 text-2xl">
                ⚙️
              </div>
              <h3 className="text-base font-bold text-slate-900 mb-1">Portfolio Synthesis Ready</h3>
              <p className="text-xs text-slate-500 mb-6">
                Click below to synthesize a tailored portfolio matching your milestone, horizon, and emergency fund buffer.
              </p>
              <div className="flex items-center justify-center gap-3">
                {onBack && (
                  <button
                    type="button"
                    onClick={onBack}
                    className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors"
                  >
                    Cancel
                  </button>
                )}
                <button
                  type="button"
                  onClick={generatePortfolio}
                  className="px-6 py-2.5 bg-finance-primary hover:bg-blue-700 text-white text-xs font-bold rounded-xl transition-all shadow-md shadow-finance-primary/20 cursor-pointer"
                >
                  Generate Portfolio Now
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default GuidedPortfolioWizard;
