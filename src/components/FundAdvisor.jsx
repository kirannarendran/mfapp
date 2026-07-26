import React, { useState, useRef, useEffect } from 'react';

// Simple markdown-to-HTML converter for the recommendation output
function renderMarkdown(text) {
  return text
    .replace(/^### (.+)$/gm, '<h3 class="text-base font-semibold text-finance-text-primary mt-4 mb-1">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="text-lg font-bold text-finance-text-primary mt-5 mb-2">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="text-xl font-bold text-finance-text-primary mt-5 mb-2">$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-finance-text-primary">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^- (.+)$/gm, '<li class="ml-4 list-disc text-finance-text-secondary">$1</li>')
    .replace(/(<li.*<\/li>\n?)+/g, '<ul class="my-2 space-y-1">$&</ul>')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');
}

const SUGGESTIONS = [
  'I want to retire in 15 years with ₹2 crore. I can invest ₹20,000/month. I cannot tolerate more than a 20% drop.',
  'Help me build ₹50 lakh for my child\'s education in 10 years. SIP of ₹15,000/month. Moderate risk.',
  'I have ₹5 lakh lump sum and want to grow it aggressively over 7 years. I can handle high volatility.',
  'Conservative investor, 5-year horizon, ₹10,000 SIP, max 15% drawdown.',
];

function StepItem({ step }) {
  const isLoading = step.status === 'loading';
  return (
    <div className="flex items-start gap-3 py-2 animate-fade-in">
      <div className="text-lg w-6 shrink-0 mt-0.5">
        {isLoading ? (
          <div className="w-5 h-5 border-2 border-finance-primary border-t-transparent rounded-full animate-spin mt-0.5" />
        ) : (
          <span>{step.icon}</span>
        )}
      </div>
      <div>
        <p className={`text-sm font-medium ${isLoading ? 'text-finance-text-secondary' : 'text-finance-text-primary'}`}>
          {step.title}
        </p>
        {step.detail && (
          <p className="text-xs text-finance-text-secondary mt-0.5 leading-relaxed">{step.detail}</p>
        )}
      </div>
    </div>
  );
}

export default function FundAdvisor() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]); // { role: 'user'|'agent', content, steps, recommendation }
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (text) => {
    const userMsg = text || inputValue.trim();
    if (!userMsg || isStreaming) return;

    setInputValue('');
    setIsStreaming(true);

    const userEntry = { role: 'user', content: userMsg };
    const agentEntry = { role: 'agent', steps: [], recommendation: null, error: null };

    setMessages(prev => [...prev, userEntry, agentEntry]);

    try {
      const response = await fetch('/api/advisor/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg }),
      });

      if (!response.ok) {
        throw new Error('Server error. Please try again.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === 'step') {
              setMessages(prev => {
                const updated = [...prev];
                const last = { ...updated[updated.length - 1] };
                // Replace loading step with done, or add new step
                const existingLoadingIdx = last.steps.findIndex(s => s.status === 'loading');
                if (existingLoadingIdx >= 0 && event.status === 'done') {
                  last.steps = last.steps.map((s, i) =>
                    i === existingLoadingIdx ? event : s
                  );
                } else {
                  last.steps = [...last.steps, event];
                }
                updated[updated.length - 1] = last;
                return updated;
              });
            } else if (event.type === 'result') {
              setMessages(prev => {
                const updated = [...prev];
                const last = { ...updated[updated.length - 1], recommendation: event.recommendation };
                updated[updated.length - 1] = last;
                return updated;
              });
            } else if (event.type === 'error') {
              setMessages(prev => {
                const updated = [...prev];
                const last = { ...updated[updated.length - 1], error: event.message };
                updated[updated.length - 1] = last;
                return updated;
              });
            }
          } catch (_) { /* ignore parse errors */ }
        }
      }
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        const last = { ...updated[updated.length - 1], error: err.message };
        updated[updated.length - 1] = last;
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-finance-primary flex items-center justify-center shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold text-finance-text-primary">AI Fund Advisor</h2>
            <p className="text-xs text-finance-text-secondary">Powered by Groq · Screens 1,400+ funds by risk metrics</p>
          </div>
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto space-y-6 pb-4">
        {isEmpty && (
          <div className="space-y-4 animate-fade-in">
            <div className="bg-finance-surface border border-finance-border rounded-xl p-5">
              <p className="text-sm text-finance-text-secondary leading-relaxed">
                Tell me your <span className="text-finance-text-primary font-medium">investment goals</span> in plain language — your target corpus, timeline, monthly SIP, and how much of a portfolio dip you can tolerate. 
                I'll screen funds using risk-adjusted metrics and build a personalised recommendation for you.
              </p>
            </div>
            <p className="text-xs font-medium text-finance-text-secondary uppercase tracking-wider px-1">Try an example</p>
            <div className="grid gap-2">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(s)}
                  className="text-left text-sm text-finance-text-secondary bg-finance-surface border border-finance-border rounded-xl px-4 py-3 hover:border-finance-primary hover:text-finance-text-primary transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx}>
            {msg.role === 'user' && (
              <div className="flex justify-end">
                <div className="bg-finance-primary text-white text-sm px-4 py-3 rounded-2xl rounded-tr-sm max-w-[85%] leading-relaxed">
                  {msg.content}
                </div>
              </div>
            )}

            {msg.role === 'agent' && (
              <div className="flex flex-col gap-3">
                {/* Agent thinking trace */}
                {msg.steps.length > 0 && (
                  <div className="bg-finance-surface border border-finance-border rounded-xl px-4 py-3">
                    <p className="text-xs font-semibold text-finance-text-secondary uppercase tracking-wider mb-2">Agent Steps</p>
                    <div className="divide-y divide-finance-border">
                      {msg.steps.map((step, si) => (
                        <StepItem key={si} step={step} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Error */}
                {msg.error && (
                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 text-sm text-red-400">
                    ⚠️ {msg.error}
                  </div>
                )}

                {/* Final recommendation */}
                {msg.recommendation && (
                  <div className="bg-finance-surface border border-finance-border rounded-xl px-5 py-4">
                    <div className="flex items-center gap-2 mb-3 pb-3 border-b border-finance-border">
                      <span className="text-lg">🏆</span>
                      <p className="text-sm font-semibold text-finance-text-primary">Your Personalised Portfolio</p>
                    </div>
                    <div
                      className="text-sm text-finance-text-secondary leading-relaxed prose-sm"
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.recommendation) }}
                    />
                    <p className="text-xs text-finance-text-secondary mt-4 pt-3 border-t border-finance-border opacity-60">
                      ⚠️ This is not financial advice. Mutual fund investments are subject to market risk. Please read all scheme-related documents carefully before investing.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="pt-4 border-t border-finance-border">
        <div className="flex gap-3 items-end bg-finance-surface border border-finance-border rounded-xl px-4 py-3 focus-within:border-finance-primary transition-colors">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your investment goals… e.g. 'I want ₹1 crore in 12 years, ₹15k/month SIP, max 20% drawdown'"
            rows={2}
            disabled={isStreaming}
            className="flex-1 bg-transparent text-sm text-finance-text-primary placeholder-finance-text-secondary resize-none outline-none leading-relaxed disabled:opacity-50"
          />
          <button
            onClick={() => sendMessage()}
            disabled={!inputValue.trim() || isStreaming}
            className="shrink-0 w-9 h-9 rounded-lg bg-finance-primary text-white flex items-center justify-center hover:opacity-90 disabled:opacity-40 transition-opacity"
          >
            {isStreaming ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
              </svg>
            )}
          </button>
        </div>
        <p className="text-xs text-finance-text-secondary mt-2 text-center opacity-60">Press Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}
