import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Ranker", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# --- Load static benchmark returns from local CSV ---
# Place 'bse500_returns.csv' in the project folder alongside this app
# CSV must have columns: date (YYYY-MM-DD) and bench_ret (daily return decimal)
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv('bse500_returns.csv', parse_dates=['date'])
        df.set_index('date', inplace=True)
        return df['bench_ret']
    except Exception as e:
        st.error(f"⚠️ Error loading benchmark CSV: {e}")
        st.stop()

benchmark_returns = load_benchmark()

# --- Fetch and cache full mutual fund list ---
@st.cache_data
def fetch_fund_list():
    url = "https://api.mfapi.in/mf"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else []

fund_list = fetch_fund_list()
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
w_roll  = st.sidebar.slider("Rolling Return (5Y)", 0.0, 1.0, 0.15)
w_sd    = st.sidebar.slider("Standard Deviation", 0.0, 1.0, 0.20)
w_shp   = st.sidebar.slider("Sharpe Ratio", 0.0, 1.0, 0.05)
w_srt   = st.sidebar.slider("Sortino Ratio", 0.0, 1.0, 0.20)
w_dnc   = st.sidebar.slider("Downside Capture", 0.0, 1.0, 0.20)
w_ups   = st.sidebar.slider("Upside Capture", 0.0, 1.0, 0.05)
w_alpha = st.sidebar.slider("Alpha %", 0.0, 1.0, 0.10)
w_beta  = st.sidebar.slider("Beta", 0.0, 1.0, 0.05)

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

# --- Fetch NAV history ---
@st.cache_data
def fetch_nav(code):
    r = requests.get(f"https://api.mfapi.in/mf/{code}")
    if r.status_code != 200:
        return None
    df = pd.DataFrame(r.json().get("data", []))
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['nav']  = pd.to_numeric(df['nav'], errors='coerce')
    return df.dropna(subset=['nav']).set_index('date').sort_index()

# --- Compute metrics using uploaded benchmark returns ---
def compute_metrics(df):
    if df is None or len(df) < 5*252:
        return None
    df5 = df.iloc[-5*252:]
    rets = df5['nav'].pct_change().dropna()
    rr    = (df5['nav'].iloc[-1]/df5['nav'].iloc[0])**(1/5) - 1
    sd    = rets.std()*np.sqrt(252)
    sharpe = (rets.mean()*252 - 0.05)/sd if sd>0 else np.nan
    dn = rets[rets<0]
    sortino = (rets.mean()*252 - 0.05)/(dn.std()*np.sqrt(252)) if not dn.empty else np.nan
    # Align with benchmark returns
    bench = benchmark_returns.reindex(rets.index).dropna()
    fund = rets.reindex(bench.index)
    up   = fund[bench>0]
    down = fund[bench<0]
    up_cap   = (up.mean()/bench[bench>0].mean())*100 if not up.empty else np.nan
    down_cap = (down.mean()/bench[bench<0].mean())*100 if not down.empty else np.nan
    alpha = (rr - (bench.sum()))*100  # rough 5y sum comparison
    beta  = np.nan
    return {
        "Rolling Return": rr*100,
        "SD": sd*100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Downside Capture": down_cap,
        "Upside Capture": up_cap,
        "Alpha": alpha,
        "Beta": beta
    }

# --- Process and display ---
results = []
for name, code in fund_mapping.items():
    if code in scheme_codes:
        nav = fetch_nav(code)
        m = compute_metrics(nav)
        if m:
            m["Fund Name"] = name
            results.append(m)
if results:
    df = pd.DataFrame(results)
    for col, w in weights.items():
        vals = df[col]
        if col in ("SD","Beta","Downside Capture"):
            norm = (vals.max()-vals)/(vals.max()-vals.min())
        else:
            norm = (vals-vals.min())/(vals.max()-vals.min())
        df[col+"_norm"] = norm*w
    df["Score"] = (df.filter(like="_norm").sum(axis=1)*10).round(2)
    display = df[["Fund Name"]+list(weights.keys())+["Score"]].round(2)
    display = display.sort_values("Score", ascending=False).reset_index(drop=True)
    display.index += 1
    display.insert(0, "SL No", display.index)
    st.dataframe(display, use_container_width=True)
else:
    st.info("Select at least one fund to view metrics.")
