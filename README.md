# FundSense

A full-stack web application for screening, tracking, analysing, and comparing Indian mutual funds — built with a focus on **risk-adjusted metrics** rather than raw returns.

---

## Features

### 📋 Fund Universe
Search across 5,000+ AMFI-registered Direct Growth mutual funds. Click any fund to view a detailed breakdown of its NAV history, risk metrics, and performance charts.

### ⚖️ Fund Comparison
Select up to 4 funds and compare them side-by-side across all key metrics — CAGR, Alpha, Beta, Sharpe Ratio, Sortino Ratio, Standard Deviation, and more.

### 🔍 AI Fund Screener
A dual-layer screening engine:
- **Strict Filter Criteria** — Hard knockout rules. A fund must meet every threshold (e.g. Min 5Y Return, Max Beta) to appear in results.
- **Scoring Weights** — Among qualifying funds, you set how much each metric contributes to the final score (0–100). Funds are ranked accordingly.

This approach ensures you never see a fund that fails your risk tolerance, while still being able to rank survivors by your personal priorities.

### 🤖 AI Wealth Planner
A conversational financial planning assistant powered by Groq (Llama). Describe your investment goals, risk appetite, and time horizon — it streams back a structured, step-by-step investment plan with specific fund category recommendations.

### 📊 Portfolio X-Ray
Upload a CSV of your current mutual fund holdings (Fund Name, Amount). The analyser:
1. Cross-references your funds against the live database to pull real risk metrics (Sharpe, Sortino, Alpha, Beta, 5Y CAGR).
2. Identifies underperformers using **risk-adjusted signals** — not CAGR alone. A fund with good returns but poor Sharpe/Alpha is still flagged.
3. Generates an AI health report with key observations, laggards to exit (with reasons), and a proposed action plan.
4. Includes a **Tax Impact Advisory** — all rebalancing suggestions account for Indian LTCG (₹1.25L exempt, 12.5% above) and STCG (20%) rules, recommending staggered exits or STPs where the tax cost of switching outweighs the benefit.

---

## Tech Stack

### Frontend
- **React 19** + **Vite**
- **Tailwind CSS v4**
- **Recharts** for NAV and performance charts

### Backend
- **Node.js** + **Express**
- **better-sqlite3** — local SQLite database for fund registry and NAV history
- **Groq API** (Llama model) — powers the AI Wealth Planner and Portfolio X-Ray analysis
- **mfapi.in** — source for live NAV data and fund registry

### Data & Metrics
All risk metrics are computed server-side from raw NAV history:
- **CAGR** (3Y, 5Y)
- **Alpha** — excess return over Nifty 50 benchmark
- **Beta** — volatility relative to the market
- **Sharpe Ratio** — return per unit of total risk
- **Sortino Ratio** — return per unit of downside risk
- **Standard Deviation**
- **Upside/Downside Capture Ratios**

---

## Getting Started

### Prerequisites
- Node.js 18+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
git clone https://github.com/kirannarendran/mfapp.git
cd mfapp
npm install
```

Create a `.env` file in the root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Running the App

Start the backend server (port 3001):
```bash
npm run server
```

In a separate terminal, start the frontend (port 5173):
```bash
npm run dev
```

The backend automatically syncs fund data and computes metrics on first run.

---

## Portfolio X-Ray — CSV Format

Upload a CSV with this format:

```csv
Fund Name,Amount
Parag Parikh Flexi Cap Fund - Direct Growth,150000
HDFC Mid-Cap Opportunities Fund - Direct Growth,75000
Axis Small Cap Fund - Direct Plan Growth,50000
```

---

## Disclaimer

This application is for **informational and educational purposes only**. It is not SEBI-registered financial advice. All analysis, scores, and recommendations are algorithmic outputs based on historical data. Past performance is not indicative of future results. Consult a qualified financial advisor before making investment decisions.
