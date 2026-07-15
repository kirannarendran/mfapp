import React, { useState } from 'react';
import { calculateSIP } from '../utils/financialPlannerUtils';

const FinancialPlanner = ({ onBack }) => {
    const [inputs, setInputs] = useState({
        goalName: '',
        targetAmount: '',
        years: '',
        expectedReturn: '12' // Default 12%
    });

    const [result, setResult] = useState(null);

    const handleChange = (e) => {
        setInputs({ ...inputs, [e.target.name]: e.target.value });
    };

    const handleCalculate = () => {
        const { targetAmount, years, expectedReturn } = inputs;
        if (!targetAmount || !years || !expectedReturn) return;

        const rate = Number(expectedReturn);
        const monthlySIP = calculateSIP(Number(targetAmount), Number(years), rate);

        setResult({
            monthlySIP,
            projections: { total: targetAmount, years, expectedRate: rate }
        });
    };

    return (
        <div className="animate-fade-in pb-20">
            <button onClick={onBack} className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2">
                ← Back to Dashboard
            </button>

            <h2 className="text-3xl font-bold mb-8">
                Financial Goal Planner
            </h2>

            <div className="grid md:grid-cols-2 gap-8">
                {/* Input Section */}
                <div className="card">
                    <h3 className="text-xl font-semibold text-finance-text-primary mb-6">Plan Your Goal</h3>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-finance-text-primary mb-1">Goal Name</label>
                            <input
                                type="text"
                                name="goalName"
                                placeholder="e.g. Dream Car, Retirement"
                                value={inputs.goalName}
                                onChange={handleChange}
                                className="w-full bg-finance-surface border border-finance-border rounded-lg px-4 py-3 text-finance-text-primary focus:outline-none focus:border-finance-primary transition-colors"
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-finance-text-primary mb-1">Target Amount (₹)</label>
                                <input
                                    type="number"
                                    name="targetAmount"
                                    placeholder="e.g. 500000"
                                    value={inputs.targetAmount}
                                    onChange={handleChange}
                                    className="w-full bg-finance-surface border border-finance-border rounded-lg px-4 py-3 text-finance-text-primary focus:outline-none focus:border-finance-primary transition-colors"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-finance-text-primary mb-1">Time Horizon (Years)</label>
                                <input
                                    type="number"
                                    name="years"
                                    placeholder="e.g. 5"
                                    value={inputs.years}
                                    onChange={handleChange}
                                    className="w-full bg-finance-surface border border-finance-border rounded-lg px-4 py-3 text-finance-text-primary focus:outline-none focus:border-finance-primary transition-colors"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-finance-text-primary mb-1 flex justify-between">
                                <span>Expected Annual Return (%)</span>
                                <span className="text-finance-primary">{inputs.expectedReturn}%</span>
                            </label>
                            <input
                                type="range"
                                name="expectedReturn"
                                min="1" max="30" step="0.5"
                                value={inputs.expectedReturn}
                                onChange={handleChange}
                                className="w-full mt-2"
                            />
                        </div>

                        <button
                            onClick={handleCalculate}
                            disabled={!inputs.targetAmount || !inputs.years || !inputs.expectedReturn}
                            className="w-full mt-4 bg-finance-primary hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            Calculate Plan
                        </button>
                    </div>
                </div>

                {/* Results Section */}
                <div className="card flex flex-col items-center justify-center">
                    {!result ? (
                        <div className="h-full flex flex-col items-center justify-center text-finance-text-secondary opacity-60">
                            <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path></svg>
                            <p>Enter your details to see the plan</p>
                        </div>
                    ) : (
                        <div className="space-y-6 animate-fade-in w-full">
                            <div className="text-center p-6 bg-finance-score-bg rounded-lg border border-finance-border">
                                <p className="text-finance-text-secondary text-sm mb-1">Required Monthly SIP</p>
                                <h3 className="text-4xl font-bold text-finance-positive">₹{result.monthlySIP.toLocaleString()}</h3>
                                <p className="text-xs text-finance-text-secondary mt-2">
                                    To reach ₹{Number(inputs.targetAmount).toLocaleString()} in {inputs.years} years
                                    (@ {result.projections.expectedRate}%)
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default FinancialPlanner;
