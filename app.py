import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---
RF_ANNUAL    = 0.06297        # 6.297% annual risk-free rate
TRADING_DAYS = 252
RF_DAILY     = RF_ANNUAL / TRADING_DAYS
BSE_CSV_PATH = 'bse500_returns.csv'
MASTER_API   = 'https://api.mfapi.in/mf'
NAV_API      = 'https://api.mfapi.in/mf/{}'

# --- DATA FETCHING ---
@st.cache_data(show_spinner=False)
def fetch_equity_schemes():
    """Fetch all schemes and filter only equity ones."""
    schemes = requests.get(MASTER_API).json()
    return [
        (s['schemeCode'], s['schemeName'])
        for s in schemes
        if 'Equity' in s['schemeName']
    ]

@st.cache_data(show_spinner=False)
def load_benchmark(path=BSE_CSV_PATH):
    """Load BSE-500 daily returns."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%B-%Y')
    df = df.set_index('Date').sort_index()
    df['ret'] = df['Close'].pct_change().dropna()
    return df['ret']

@st.cache_data(show_spinner=False)
def fetch_nav_history(code):
    """Fetch NAV history for a given scheme."""
    js = requests.get(NAV_API.format(code)).json()
    df = pd.DataFrame(js.get('data', []))
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df['nav']  = df['nav'].astype(float)
    df = df.set_index('date').sort_index()
    return df['nav']

# --- METRICS ---
def cagr(navs):
    if navs.empty:
        return np.nan
    span = (navs.index[-1] - navs.index[0]).days / 365.25
    return (navs.iloc[-1] / navs.iloc[0]) ** (1/span) - 1

def alpha_beta(fund_ret, bench_ret):
    df = pd.concat([fund_ret, bench_ret], axis=1, join='inner').dropna()
    fe = df.iloc[:,0] - RF_DAILY
    be = df.iloc[:,1] - RF_DAILY
    slope, intercept = np.polyfit(be, fe, 1)
    return intercept * TRADING_DAYS, slope

def ann_std(r):    return r.std() * np.sqrt(TRADING_DAYS)
def sharpe(r):     return ((r - RF_DAILY).mean() / r.std()) * np.sqrt(TRADING_DAYS)

def sortino(r):
    ex = r - RF_DAILY
    neg = ex[ex < 0]
    dd  = np.sqrt((neg**2).mean()) if len(neg)>0 else np.nan
    return (ex.mean()/dd)*np.sqrt(TRADING_DAYS) if dd else np.nan

def capture(r, bench, upside=True):
    mask = bench > 0 if upside else bench < 0
    if mask.sum() == 0:
        return np.nan
    return r[mask].mean() / bench[mask].mean() * 100

# --- STREAMLIT APP ---
def main():
    st.title("📈 Mutual Fund Ranking Tool")

    # Default to the last 5 years
    today      = datetime.today()
    start_date = today.replace(year=today.year - 5)

    # Load data sources
    bse_ret = load_benchmark()
    schemes = fetch_equity_schemes()
    names   = [n for _, n in schemes]

    # Sidebar: metric weights
    st.sidebar.header("Metric Weights")
    metrics = ['CAGR','Alpha','Beta','Std Dev','Sharpe','Sortino','Upside Cap','Downside Cap']
    weights = {m: st.sidebar.slider(m, 0.0, 2.0, 1.0, 0.1) for m in metrics}

    # Fund selector
    selected = st.multiselect("Select up to 5 Equity MFs", names, max_selections=5)
    if not selected:
        st.info("Pick at least one fund to compare.")
        return

    # Fetch, chart, compute
    records = []
    for name in selected:
        code = next(c for c,n in schemes if n==name)
        nav  = fetch_nav_history(code)
        # filter to last 5 years
        nav  = nav[nav.index >= pd.to_datetime(start_date)]
        if nav.empty:
            continue
        ret  = nav.pct_change().dropna()

        # NAV vs. BSE chart
        idx_base = bse_ret.cumsum().add(nav.iloc[0])
        merged   = pd.concat([nav, idx_base], axis=1, join='inner')
        merged.columns = ['NAV','BSE 500 (idx based)']
        st.subheader(name)
        st.line_chart(merged)

        # Collect metrics
        a, b = alpha_beta(ret, bse_ret)
        records.append({
            'Fund Name': name,
            'CAGR'     : cagr(nav)*100,
            'Alpha'    : a*100,
            'Beta'     : b,
            'Std Dev'  : ann_std(ret)*100,
            'Sharpe'   : sharpe(ret),
            'Sortino'  : sortino(ret),
            'Upside Cap'   : capture(ret, bse_ret, True),
            'Downside Cap' : capture(ret, bse_ret, False)
        })

    df = pd.DataFrame(records)
    if df.empty:
        st.warning("No NAV data for selected funds in the last 5 years.")
        return

    # Ranking & scoring
    df_rank, score_cols = df.copy(), []
    for m in metrics:
        asc = m in ['Beta','Std Dev','Downside Cap']
        df_rank[f'{m}_rank']  = df_rank[m].rank(ascending=asc, method='min')
        max_r                  = df_rank[f'{m}_rank'].max()
        df_rank[f'{m}_score'] = ((max_r - df_rank[f'{m}_rank']) + 1) * weights[m]
        score_cols.append(f'{m}_score')

    total = sum(weights.values())
    df_rank['Score (10)'] = df_rank[score_cols].sum(axis=1) / total * 10
    df_rank = df_rank.sort_values('Score (10)', ascending=False).reset_index(drop=True)

    # Style & display
    def style_rows(x):
        n = x.name; L = len(x)
        if n < L*0.25:      return ['background-color:#d4f8d4']*L
        if n >= L*0.75:     return ['background-color:#f8d4d4']*L
        return ['']*L

    display_cols = ['Fund Name'] + metrics + ['Score (10)']
    st.subheader("Ranking")
    st.dataframe(df_rank[display_cols].style.apply(style_rows, axis=1),
                 use_container_width=True)

if __name__ == '__main__':
    main()
