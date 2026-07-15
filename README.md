# Mutual Fund Tracker & Analysis Suite

This project contains two major components:
1. **MfApp**: A React-based web application for screening, tracking, and comparing mutual funds. Features include detailed fund views, comparison tables, a financial planner, and a unified API service for backend data integration.
2. **5-Year Persistence Empirical Baseline Analysis**: A rigorous statistical and survivorship audit (`mf_5y_persistence/`) of the 5-year persistence base rate for Indian mutual funds (Direct Growth).

## Web Application

The frontend is built using:
- React + Vite
- TailwindCSS for styling
- Core components: `FundScreener`, `FundList`, `FundDetail`, `ComparisonView`, and `FinancialPlanner`.

The frontend interfaces with the local Node.js `server/` instance which syncs data from `mfapi.in` and calculates advanced risk metrics (CAGR, Alpha, Beta, Sharpe, Sortino).

## 5-Year Persistence Empirical Baseline

This rigorous empirical audit analyzes mutual funds with a past 5-year CAGR of 11% to 13%.
It implements exact identity matching, direct vs. regular unspooling, and handles unresolved survivorship edge cases.

### Key Audit Findings
- **Complete Observations**: 142
- **Complete-case Sample Rate**: 59.15% (84 successes)
- **Adverse-Case Bound**: 49.70%
- **Independent Market Blocks**: 1
- **Evidence Quality**: Low (due to overlapping cohorts)

> **Disclaimer**: This statistic describes a limited pooled historical sample. It is not a prediction, recommendation, expected return, or guarantee that a particular mutual fund will repeat its previous performance.
