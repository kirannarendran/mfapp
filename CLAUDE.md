# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FundSense.AI: a React 19 + Vite frontend and an Express 5 + better-sqlite3 backend for screening, comparing and planning with Indian mutual funds (Direct Growth plans). Risk metrics are computed server-side from raw NAV history pulled from mfapi.in. AI features call Groq's OpenAI-compatible API.

## Commands

```bash
npm run server        # backend on :3001 (node server/index.js)
npm run dev           # Vite frontend on :5173, proxies /api -> :3001
npm run build         # vite build -> dist/ (served by Express in production)
npm run lint          # eslint .

npm run sync:local    # run the full data sync pipeline once (mfapi.in -> SQLite -> metrics)
npm run snapshot      # copy server/data/mf_tracker.db into server/seed_snapshot.db
npm run sync:push     # snapshot + git commit + git push origin main (pushes! confirm before running)
```

Tests are plain Node scripts, not a test framework:

```bash
node test_functional_suite.js          # imports routers/services directly, simulates req/res; needs a populated DB
node test_regression.js                # same approach, covers metricsEngine + routes
node --test server/ml_demotion.test.js # node:test, hits http://localhost:3001 — server must be running
```

There is no single-test filter; each suite runs top to bottom and exits non-zero on any failure.

## Environment

`.env` at repo root (loaded via `dotenv/config` in `server/index.js`):
- `GROQ_API_KEY` (the code also falls back to `GROK_API_KEY`, which is the name currently used in `.env`; the endpoint is Groq regardless)
- `GOOGLE_CLIENT_ID` / `VITE_GOOGLE_CLIENT_ID`, `JWT_SECRET`: Google sign-in and session JWTs. If `JWT_SECRET` is missing, `server/index.js` generates a random per-process secret and logs a warning, so sessions reset on restart.
- `CORS_ORIGIN`: optional comma-separated allowed origins. Cross-origin access is off by default because the frontend is same-origin.
- `VITE_GA_MEASUREMENT_ID`: GA4 (see `src/utils/analytics.js`)
- `DISABLE_BACKGROUND_SYNC`, `SKIP_SYNC`, `NODE_ENV=production` / `RENDER=true`: control the sync scheduler (below)

## Architecture

### Data pipeline (server)

- `server/db.js` owns the schema. The runtime DB is `server/data/mf_tracker.db` (gitignored). On boot, if it is missing or the `funds` table is empty, it is copied from the committed `server/seed_snapshot.db`. Schema changes are done with `CREATE TABLE IF NOT EXISTS` plus try/catch `ALTER TABLE ADD COLUMN` lists. When adding a column, also add it to those lists and to the table definitions in `server/createSnapshot.js`, which rebuilds the snapshot schema by hand.
- `server/services/scheduler.js` `runFullSync()` runs five steps in order: risk-free rate (`rbiRateFetcher.js` scrapes the 10Y G-Sec yield into `config.risk_free_rate`), fund registry, benchmark NAVs, tracked fund NAVs (`dataSync.js`), then `metricsEngine.recomputeAllMetrics()` into `fund_metrics`. Cron runs it at 17:30 UTC on weekdays, and `checkMissedSync()` catches up on boot.
- **Production (Render) runs in "snapshot mode"**: when `NODE_ENV=production` or `RENDER=true`, cron and the catch-up sync are disabled and the app serves only the bundled snapshot. Fresh data reaches production by running the sync locally, then `npm run sync:push`. `.github/workflows/keepalive.yml` pings the Render URL to keep it awake.
- Benchmarks: equity funds use scheme 100484 (Nifty 50 index fund), debt funds use 120137 (10Y gilt fund), chosen by `isDebtCategory()`. Those constants and `isDebtCategory` are **duplicated** in `server/services/metricsEngine.js` and `src/api.js`. Keep both copies in sync.
- `ml_predictions` is joined into `/api/funds/screener` only when `includeExperimental=true`. It is not created by `db.js`; it comes from outside. ML fields must never appear in the default screener response, and `minMlProb` is rejected with a 400 (`server/ml_demotion.test.js` enforces both).

### API

- `server/routes/funds.js`: fund list/detail/metrics/compare, `/funds/screener` (hard filter thresholds in SQL, then weighted scoring), `/benchmark`, `/config`, sync status and manual or admin sync triggers.
- `server/routes/advisor.js` -> `server/services/aiAdvisor.js`: `/advisor/chat`, `/advisor/plan` (structured guided-portfolio plan) and `/advisor/analyze` (Portfolio X-Ray). These stream responses to `res`. The model is set by `GROQ_MODEL` in `aiAdvisor.js`, and prompts include Indian LTCG/STCG tax rules.
- `server/routes/auth.js`: `POST /auth/google` verifies the Google ID token, upserts `users` and returns a JWT. `PUT /auth/profile` and `GET /auth/me` read the Bearer JWT.
- `server/index.js` applies `helmet` (CSP off) and per-IP rate limits: 600 requests/15 min on `/api`, 30/hour on `/api/advisor`. `/admin/sync` and `/sync/manual` are disabled in production.
- In production, Express also serves `dist/` with an SPA fallback for non-`/api` paths.

### Frontend

- There is no router. `src/App.jsx` switches views with boolean state flags (`isWizard`, `isAnalyzer`, `isScreening`, `isPlanning`, `isComparing`, `isAbout`, `isProfile`, `selectedSchemeCode`, …), and `currentViewTitle` derives from them (also used for GA page views). A new view needs a flag, a branch in that chain and a way to reset the other flags.
- `src/api.js` is the single fetch client (`/api` base) and also holds shared benchmark helpers.
- `src/context/AuthContext.jsx` handles Google sign-in and guest mode. **Saved portfolios live only in localStorage**: `fundsense_guest_portfolios` for guests, per-user keys plus `fundsense_target_portfolios_global` for signed-in users. Guest portfolios migrate into the user's key on login. Profile data is stored server-side in `users` and also mirrored under several legacy `fundsense_profile_*` localStorage keys that `App.jsx` and `AuthContext.jsx` both check.
- `GuidedPortfolioWizard.jsx` (the largest component) is the goal-based builder. It calls `/advisor/plan` and computes projections via `src/utils/financialPlannerUtils.js` and `financialMetrics.js`. The functional suite asserts on these projections (SIP/lumpsum wealth, weighted CAGR).
- Styling uses Tailwind v4 via `@tailwindcss/postcss`.

### Other directories

- `mf_5y_persistence/` and `legacy_6m_category_outperformance_model_archive/` are standalone Python research projects with their own `requirements.txt` and pytest tests. They are not used by the web app at runtime. The legacy one is archived and has a committed `.venv`.
- Root `*.py` scripts (`convert_to_light.py`, `fix_*.py`, `update_styles.py`) and `scratch/` are one-off migration helpers.
