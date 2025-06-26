import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Ranker", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# --- Fetch and cache full mutual fund list ---
@st.cache_data
def fetch_fund_list():
    url = "https://api.mfapi.in/mf"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else []

fund_list = fetch_fund_list()

# --- Build Autocomplete Fund Picker Safely (using camelCase keys) ---
fund_mapping = {
    fund['schemeName']: fund['schemeCode']
    for fund in fund_list
    if fund.get('schemeName') and fund.get('schemeCode')
}

# --- Fund selection ---
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=list(fund_mapping.keys()),
    help="Start typing to filter by fund name",
    max_selections=10
)
scheme_codes = [fund_mapping[name] for name in selected_funds]

# --- Scoring weights ---
st.sidebar.header("🎯 Metric Weights (must sum to 1)")
w_roll = st.sidebar.slider("Rolling Return (5Y)", 0.0, 1.0, 0.15)
w_sd   = st.sidebar.slider("Standard Deviation", 0.0, 1.0, 0.20)
w_shp  = st.sidebar.slider("Sharpe Ratio", 0.0, 1.0, 0.05)
w_srt  = st.sidebar.slider("Sortino Ratio", 0.0, 1.0, 0.20)
w_dnc  = st.sidebar.slider("Downside Capture %", 0.0, 1.0, 0.20)
w_ups  = st.sidebar.slider("Upside Capture %", 0.0, 1.0, 0.05)
w_alpha= st.sidebar.slider("Alpha %", 0.0, 1.0, 0.10)
w_beta = st.sidebar.slider("Beta", 0.0, 1.0, 0.05)

total_w = w_roll + w_sd + w_shp + w_srt + w_dnc + w_ups + w_alpha + w_beta
if not np.isclose(total_w, 1.0):
    st.sidebar.error(f"⚠️ Weights sum to {total_w:.2f} (must be 1.0)")
    st.stop()

weights = {
    "Rolling Return": w_roll,
    "SD": w_sd,
    "Sharpe": w_shp,
    "Sortino": w_srt,
    "Downside Capture": w_dnc,
    "Upside Capture": w_ups,
    "Alpha": w_alpha,
    "Beta": w_beta
}

# --- Hardcoded benchmark return (5Y) ---
bench_ret_5y = 0.2359

# --- Fetch NAV history ---
@st.cache_data
def fetch_nav(code):
    r = requests.get(f"https://api.mfapi.in/mf/{code}")
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['nav']  = pd.to_numeric(df['nav'], errors='coerce')
    return df.dropna(subset=['nav']).set_index('date').sort_index()

# --- Compute metrics ---
def compute_metrics(df):
    if df is None or len(df) < 5*252:
        return None
    # Last 5 years
    df5 = df.iloc[-5*252:]
    rets = df5['nav'].pct_change().dropna()
    rr = (df5['nav'].iloc[-1]/df5['nav'].iloc[0])**(1/5) - 1
    sd = rets.std()*np.sqrt(252)
    sharpe = (rets.mean()*252 - 0.05)/sd if sd>0 else np.nan
    dn_rets = rets[rets<0]
    sortino = (rets.mean()*252 - 0.05)/(dn_rets.std()*np.sqrt(252)) if not dn_rets.empty else np.nan
    # Capture
    down_cap = (dn_rets.mean() / (-bench_ret_5y/252)) * 100 if not dn_rets.empty else np.nan
    up_rets = rets[rets>0]
    up_cap = (up_rets.mean()  / ( bench_ret_5y/252)) * 100 if not up_rets.empty else np.nan
    # Alpha/Beta via regression
    bench_series = np.full(len(rets), bench_ret_5y/252)
    b, a, *_ = linregress(bench_series, rets)
    alpha = a*252
    beta  = b
    return {
        "Rolling Return": rr*100,
        "SD": sd*100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Downside Capture": down_cap,
        "Upside Capture": up_cap,
        "Alpha": alpha*100,
        "Beta": beta
    }

# --- Process selections ---
results = []
for name, code in fund_mapping.items():
    if code in scheme_codes:
        nav = fetch_nav(code)
        m = compute_metrics(nav)
        if m:
            m["Fund Name"] = name
            results.append(m)

# --- Display table ---
if results:
    df = pd.DataFrame(results)
    # normalize & score out of 10
    for metric, w in weights.items():
        col = metric
        vals = df[col]
        if metric in ("SD","Beta","Downside Capture"):
            # lower is better
            norm = (vals.max()-vals)/(vals.max()-vals.min())
        else:
            norm = (vals-vals.min())/(vals.max()-vals.min())
        df[col+"_norm"] = norm*w
    df["Score"] = (df[[c for c in df if c.endswith("_norm")]].sum(axis=1)*10).round(2)
    # cleanup
    keep = ["Fund Name"] + list(weights.keys()) + ["Score"]
    df = df[keep].round(2).sort_values("Score", ascending=False).reset_index(drop=True)
    df.index = df.index+1
    df.insert(0,"SL No",df.index)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Select at least one fund to view metrics.")
