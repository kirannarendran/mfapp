import React, { useState, useRef, useEffect } from 'react';

const AIPortfolioAnalyzer = ({ onBack }) => {
    const [holdings, setHoldings] = useState([]);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [aiSteps, setAiSteps] = useState([]);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [error, setError] = useState(null);
    const endOfMessagesRef = useRef(null);

    // Simple CSV parser
    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const text = event.target.result;
            const lines = text.split('\n').map(l => l.trim()).filter(l => l);
            if (lines.length < 2) {
                setError("Invalid CSV format. Please ensure it has headers and data rows.");
                return;
            }

            // Assume standard comma separation. Let's find columns that might be Name and Value/Amount
            const headers = lines[0].toLowerCase().split(',');
            
            let nameIdx = headers.findIndex(h => h.includes('name') || h.includes('scheme') || h.includes('fund'));
            let valueIdx = headers.findIndex(h => h.includes('value') || h.includes('amount') || h.includes('current'));

            // Fallback if headers don't strictly match typical names
            if (nameIdx === -1) nameIdx = 0; // Usually first col
            if (valueIdx === -1) valueIdx = headers.length > 1 ? 1 : 0;

            const parsedHoldings = [];
            for (let i = 1; i < lines.length; i++) {
                // simple split by comma, ignoring quotes for this basic v1
                const cols = lines[i].split(','); 
                if (cols.length > Math.max(nameIdx, valueIdx)) {
                    const name = cols[nameIdx].replace(/['"]/g, '').trim();
                    const valueStr = cols[valueIdx].replace(/[^0-9.]/g, ''); // strip out currency symbols
                    const value = parseFloat(valueStr) || 0;
                    if (name && value > 0) {
                        parsedHoldings.push({ fundName: name, value });
                    }
                }
            }

            if (parsedHoldings.length === 0) {
                setError("Could not extract any valid holdings from the CSV.");
                return;
            }

            setHoldings(parsedHoldings);
            setError(null);
            setAnalysisResult(null);
            setAiSteps([]);
        };
        reader.onerror = () => setError("Failed to read file.");
        reader.readAsText(file);
    };

    const handleAnalyze = async () => {
        if (holdings.length === 0) return;

        setIsAnalyzing(true);
        setAiSteps([]);
        setAnalysisResult(null);
        setError(null);

        try {
            const response = await fetch('/api/advisor/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ holdings })
            });

            if (!response.ok) {
                const text = await response.text();
                throw new Error(`Server error: ${response.status} - ${text}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') continue;

                        try {
                            const data = JSON.parse(dataStr);
                            if (data.type === 'step') {
                                setAiSteps(prev => {
                                    const exists = prev.findIndex(s => s.title === data.title);
                                    if (exists !== -1) {
                                        const updated = [...prev];
                                        updated[exists] = data;
                                        return updated;
                                    }
                                    return [...prev, data];
                                });
                            } else if (data.type === 'result') {
                                setAnalysisResult(data.analysis);
                            } else if (data.type === 'error') {
                                setError(data.message);
                            }
                        } catch (e) {
                            console.error('Error parsing SSE json:', e, dataStr);
                        }
                    }
                }
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
        }
    };

    useEffect(() => {
        if (endOfMessagesRef.current) {
            endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [aiSteps, analysisResult]);

    return (
        <div className="max-w-4xl mx-auto pb-24">
            <div className="flex items-center gap-4 mb-8">
                {onBack && (
                    <button onClick={onBack} className="p-2 hover:bg-slate-100 rounded-full transition-colors">
                        <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                    </button>
                )}
                <div>
                    <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Portfolio X-Ray</h1>
                    <p className="text-slate-500 text-sm mt-1">Upload your CAS or broker CSV to get an AI health check.</p>
                </div>
            </div>

            {/* Input Section */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden mb-8">
                <div className="p-6 md:p-8">
                    <label className="block text-sm font-semibold text-slate-700 mb-4">Upload CSV File</label>
                    <div className="border-2 border-dashed border-slate-200 rounded-xl p-8 text-center hover:bg-slate-50 transition-colors">
                        <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" id="csv-upload" />
                        <label htmlFor="csv-upload" className="cursor-pointer flex flex-col items-center gap-3">
                            <div className="w-12 h-12 bg-finance-primary/10 rounded-full flex items-center justify-center text-finance-primary">
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                            </div>
                            <div>
                                <span className="font-semibold text-finance-primary">Click to upload</span>
                                <span className="text-slate-500"> or drag and drop</span>
                            </div>
                            <p className="text-xs text-slate-400">CSV file containing 'Fund Name' and 'Value' columns</p>
                        </label>
                    </div>

                    {holdings.length > 0 && (
                        <div className="mt-6 p-4 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="flex justify-between items-center mb-3">
                                <h3 className="text-sm font-semibold text-slate-700">Detected Holdings ({holdings.length})</h3>
                                <span className="text-xs font-bold text-finance-primary bg-finance-primary/10 px-2 py-1 rounded-md">
                                    Total: ₹{holdings.reduce((sum, h) => sum + h.value, 0).toLocaleString('en-IN')}
                                </span>
                            </div>
                            <div className="max-h-40 overflow-y-auto space-y-2">
                                {holdings.map((h, i) => (
                                    <div key={i} className="flex justify-between items-center text-sm py-1 border-b border-slate-100 last:border-0">
                                        <span className="text-slate-600 truncate mr-4">{h.fundName}</span>
                                        <span className="font-semibold text-slate-900 shrink-0">₹{h.value.toLocaleString('en-IN')}</span>
                                    </div>
                                ))}
                            </div>
                            <div className="mt-6 flex justify-end">
                                <button
                                    onClick={handleAnalyze}
                                    disabled={isAnalyzing}
                                    className="px-6 py-2.5 bg-emerald-600 text-white font-bold rounded-xl hover:bg-emerald-700 transition-all flex items-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed shadow-[0_4px_12px_rgba(16,185,129,0.2)] hover:shadow-[0_6px_16px_rgba(16,185,129,0.3)]"
                                >
                                    {isAnalyzing ? (
                                        <>
                                            <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                                            Generate AI Health Check
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="mb-8 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-start gap-3">
                    <svg className="w-5 h-5 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    <p className="text-sm font-medium">{error}</p>
                </div>
            )}

            {/* Progress Visualization */}
            {aiSteps.length > 0 && !analysisResult && (
                <div className="mb-8">
                    <h3 className="text-sm font-bold text-slate-900 mb-4 px-1">AI Analysis Progress</h3>
                    <div className="space-y-4">
                        {aiSteps.map((step, idx) => (
                            <div key={idx} className={`p-4 rounded-xl border flex gap-4 transition-all duration-500
                                ${step.status === 'in_progress' ? 'bg-white border-finance-primary/30 shadow-sm' : 'bg-slate-50 border-slate-100'}`}>
                                <div className="shrink-0 mt-1">
                                    {step.status === 'in_progress' ? (
                                        <svg className="animate-spin h-5 w-5 text-finance-primary" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    ) : (
                                        <div className="h-5 w-5 rounded-full bg-emerald-100 flex items-center justify-center">
                                            <svg className="w-3.5 h-3.5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg>
                                        </div>
                                    )}
                                </div>
                                <div>
                                    <h4 className={`text-sm font-bold ${step.status === 'in_progress' ? 'text-finance-primary' : 'text-slate-700'}`}>{step.title}</h4>
                                    <p className="text-sm text-slate-500 mt-1 leading-relaxed">{step.details}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Result Visualization */}
            {analysisResult && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    {/* Header Summary */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
                        <div className="p-6 md:p-8 flex flex-col md:flex-row items-center gap-8">
                            <div className="shrink-0 flex flex-col items-center">
                                <div className="relative w-32 h-32 flex items-center justify-center">
                                    <svg className="absolute inset-0 w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                                        <circle cx="50" cy="50" r="45" fill="none" stroke="#f1f5f9" strokeWidth="8" />
                                        <circle cx="50" cy="50" r="45" fill="none" stroke={analysisResult.overall_health_score >= 7 ? '#10b981' : analysisResult.overall_health_score >= 4 ? '#f59e0b' : '#ef4444'} strokeWidth="8" strokeDasharray={`${(analysisResult.overall_health_score / 10) * 283} 283`} strokeLinecap="round" />
                                    </svg>
                                    <div className="text-center">
                                        <span className="text-3xl font-black text-slate-900">{analysisResult.overall_health_score}</span>
                                        <span className="text-sm font-bold text-slate-400 block -mt-1">/ 10</span>
                                    </div>
                                </div>
                                <span className="mt-4 text-xs font-bold tracking-wider uppercase text-slate-500">Health Score</span>
                            </div>
                            <div className="flex-1">
                                <h3 className="text-xl font-bold text-slate-900 mb-4">Portfolio Diagnosis</h3>
                                <ul className="space-y-3">
                                    {analysisResult.key_observations.map((obs, idx) => (
                                        <li key={idx} className="flex gap-3 text-sm text-slate-700">
                                            <svg className="w-5 h-5 text-finance-primary shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                            <span className="leading-relaxed">{obs}</span>
                                        </li>
                                    ))}
                                </ul>
                                {analysisResult.risk_adjusted_summary && (
                                    <div className="mt-5 pt-4 border-t border-slate-100">
                                        <p className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Risk-Adjusted Overview</p>
                                        <p className="text-sm text-slate-600 leading-relaxed">{analysisResult.risk_adjusted_summary}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Laggards to Exit */}
                    {analysisResult.laggards_to_exit && analysisResult.laggards_to_exit.length > 0 && (
                        <div className="bg-red-50 rounded-2xl border border-red-100 overflow-hidden">
                            <div className="px-6 py-4 border-b border-red-100 bg-red-100/50">
                                <h4 className="text-sm font-bold text-red-900 flex items-center gap-2">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                                    Underperformers Identified
                                </h4>
                            </div>
                            <div className="p-6">
                                <div className="space-y-4">
                                    {analysisResult.laggards_to_exit.map((laggard, idx) => (
                                        <div key={idx} className="bg-white p-4 rounded-xl border border-red-100 shadow-sm">
                                            <h5 className="font-bold text-slate-900">{laggard.fund_name || laggard.scheme_name}</h5>
                                            <p className="text-sm text-red-700 mt-1">{laggard.risk_adjusted_reason || laggard.reason}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Proposed Action Plan */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
                        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
                            <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                            <h4 className="font-bold text-slate-900">AI Proposed Action Plan</h4>
                        </div>
                        <div className="p-6">
                            <ul className="space-y-4">
                                {analysisResult.proposed_action_plan.map((action, idx) => (
                                    <li key={idx} className="flex gap-4">
                                        <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center font-bold text-sm shrink-0">
                                            {idx + 1}
                                        </div>
                                        <div className="text-sm text-slate-700 pt-1.5 leading-relaxed">
                                            {action}
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    {/* Tax Advisory Note */}
                    <div className="rounded-2xl border border-amber-200 bg-amber-50 overflow-hidden">
                        <div className="px-6 py-4 border-b border-amber-200 bg-amber-100/60 flex items-center gap-2">
                            <svg className="w-5 h-5 text-amber-700 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            <h4 className="font-bold text-amber-900">Tax Impact Advisory</h4>
                            <span className="ml-auto text-xs font-semibold bg-amber-200 text-amber-800 px-2 py-0.5 rounded-full">India LTCG/STCG Rules</span>
                        </div>
                        <div className="p-6 space-y-4">
                            {analysisResult.tax_advisory_note && (
                                <p className="text-sm text-amber-900 leading-relaxed font-medium">{analysisResult.tax_advisory_note}</p>
                            )}
                            <div className="grid sm:grid-cols-2 gap-3 pt-2">
                                <div className="bg-white rounded-xl p-4 border border-amber-100">
                                    <p className="text-xs font-bold uppercase tracking-wider text-amber-600 mb-1">LTCG (Held &gt; 1 year)</p>
                                    <p className="text-sm text-slate-700">First <span className="font-bold text-slate-900">₹1,25,000</span> exempt per FY.<br/>Gains above taxed at <span className="font-bold text-amber-700">12.5%</span>.</p>
                                </div>
                                <div className="bg-white rounded-xl p-4 border border-amber-100">
                                    <p className="text-xs font-bold uppercase tracking-wider text-red-500 mb-1">STCG (Held &lt; 1 year)</p>
                                    <p className="text-sm text-slate-700">All gains taxed at <span className="font-bold text-red-600">20%</span> flat.<br/>Consider STP to defer tax.</p>
                                </div>
                            </div>
                            <p className="text-xs text-amber-700 italic pt-1">* This is AI-generated guidance. Consult a SEBI-registered tax advisor before making redemption decisions.</p>
                        </div>
                    </div>
                </div>
            )}
            <div ref={endOfMessagesRef} />
        </div>
    );
};

export default AIPortfolioAnalyzer;
