cat > app.py << 'EOF'
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---
RF_ANNUAL = 0.06297          # 6.297% annual risk-free rate
TRADING_DAYS = 252
RF_DAILY = RF_ANNUAL / TRADING_DAYS
BSE_CSV_PATH = 'bse500_returns.csv'
MASTER_API = 'https://api.mfapi.in/mf'
NAV_API = 'https://api.mfapi.in/mf/{}'

# --- DATA SOURCES ---
@st.cache_data(show_spinner=False)
def fetch_equity_schemes():
    """Fetch only 'Equity' category schemes by inspecting metadata."""
    schemes = requests.get(MASTER_API).json()
    equity = []
    for s in schemes:
        code = s['schemeCode']
        js = requests.get(NAV_API.format(code)).json()
        cat = js.get('meta', {}).get('scheme_category', '')
        if 'Equity' in cat:
            equity.append((code, s['schemeName']))
    return equity

@st.cache_data(show_spinner=False)
def load_benchmark(path=BSE_CSV_PATH, start_date=None):
    """Load BSE-500 daily returns and filter by date."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%B-%Y')
    df = df.set_index('Date').sort_index()
    if start_date:
        df = df[df.index >= start_date]
    df['ret'] = df['Close'].pct_change().dropna()
    return df['ret']

@st.cache_data(show_spinner=False)
def fetch_nav_history(code, start_date=None):
    """Fetch NAV history for a given scheme code."""
    js = requests.get(NAV_API.format(code)).json()
    df = pd.DataFrame(js.get('data', []))
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df['nav'] = df['nav'].astype(float)
    df = df.set_index('date').sort_index()
    if start_date:
        df = df[df.index >= start_date]
    return df['nav']

# --- METRIC FUNCTIONS ---
def cagr(navs):
    if navs.empty:
        return np.nan
    span = (navs.index[-1] - navs.index[0]).days / 365.25
    return (navs.iloc[-1] / navs.iloc[0])**(1/span) - 1

def alpha_beta(fund_ret, bench_ret):
    df = pd.concat([fund_ret, bench_ret], axis=1, join='inner').dropna()
    fe = df.iloc[:, 0] - RF_DAILY
    be = df.iloc[:, 1] - RF_DAILY
    slope, intercept = np.polyfit(be, fe, 1)
    alpha_ann = intercept * TRADING_DAYS
    return alpha_ann, slope

def ann_std(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe(returns):
    excess = returns - RF_DAILY
    return excess.mean() / returns.std() * np.sqrt(TRADING_DAYS)

def sortino(returns):
    excess = returns - RF_DAILY
    neg = excess[excess < 0]
    down_dev = np.sqrt((neg**2).mean()) if not neg.empty else np.nan
    return excess.mean() / down_dev * np.sqrt(TRADING_DAYS) if down_dev else np.nan

def capture(returns, bench, upside=True):
    mask = bench > 0 if upside else bench < 0
    if mask.sum() == 0:
        return np.nan
    return returns[mask].mean() / bench[mask].mean() * 100

# --- STREAMLIT APP ---
def main():
    st.title("📈 Mutual Fund Ranking Tool")

    # Date-range picker (default last 5 years)
    end = datetime.today()
    start = st.sidebar.date_input("Start date", end.replace(year=end.year - 5))

    # Load data
    bse_ret = load_benchmark(start_date=start)
    schemes = fetch_equity_schemes()
    names = [name for _, name in schemes]

    # Sidebar: metric weights
    st.sidebar.header("Metric Weights")
    metrics = ['CAGR', 'Alpha', 'Beta', 'Std Dev', 'Sharpe', 'Sortino', 'Upside Cap', 'Downside Cap']
    weights = {m: st.sidebar.slider(m, 0.0, 2.0, 1.0, 0.1) for m in metrics}

    # Fund selector
    selected = st.multiselect("Select up to 5 Equity MFs", names, max_selections=5)
    if not selected:
        st.info("Select at least one fund to see metrics.")
        return

    records = []
    for name in selected:
        code = next(code for code, n in schemes if n == name)
        nav = fetch_nav_history(code, start_date=start)
        if nav.empty:
            continue
        ret = nav.pct_change().dropna()

        # NAV vs Benchmark chart
        idx_base = bse_ret.cumsum().add(nav.iloc[0])
        merged = pd.concat([nav, idx_base], axis=1, join='inner')
        merged.columns = ['NAV', 'BSE 500 (idx based)']
        st.subheader(name)
        st.line_chart(merged)

        # Compute metrics
        r5 = cagr(nav)
        alpha, beta_ = alpha_beta(ret, bse_ret)
        sd = ann_std(ret)
        sp = sharpe(ret)
        so = sortino(ret)
        up = capture(ret, bse_ret, upside=True)
        dn = capture(ret, bse_ret, upside=False)

        records.append({
            'Fund Name': name,
            'CAGR': r5 * 100,
            'Alpha': alpha * 100,
            'Beta': beta_,
            'Std Dev': sd * 100,
            'Sharpe': sp,
            'Sortino': so,
            'Upside Cap': up,
            'Downside Cap': dn
        })

    df = pd.DataFrame(records)
    if df.empty:
        st.warning("No valid NAV history for selected funds in this date range.")
        return

    # Ranking & scoring
    df_rank = df.copy()
    score_cols = []
    for m in metrics:
        asc = m in ['Beta', 'Std Dev', 'Downside Cap']
        df_rank[f'{m}_rank'] = df_rank[m].rank(ascending=asc, method='min')
        max_r = df_rank[f'{m}_rank'].max()
        df_rank[f'{m}_score'] = ((max_r - df_rank[f'{m}_rank']) + 1) * weights[m]
        score_cols.append(f'{m}_score')

    # Normalize to a 0–10 scale
    max_possible = sum(weights.values())
    df_rank['Score (Out of 10)'] = df_rank[score_cols].sum(axis=1) / max_possible * 10
    df_rank = df_rank.sort_values('Score (Out of 10)', ascending=False).reset_index(drop=True)

    # Styling
    def style_rows(x):
        n = len(x)
        if x.name < n * 0.25:
            return ['background-color:#d4f8d4'] * len(x)
        if x.name >= n * 0.75:
            return ['background-color:#f8d4d4'] * len(x)
        return [''] * len(x)

    display_cols = ['Fund Name'] + metrics + ['Score (Out of 10)']
    st.subheader("Ranking")
    st.dataframe(df_rank[display_cols].style.apply(style_rows, axis=1), use_container_width=True)

if __name__ == '__main__':
    main()

