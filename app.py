import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Ranker", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# --- Load benchmark returns from local CSV ---
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv("bse500_returns.csv")
    except FileNotFoundError:
        st.error("⚠️ Could not find bse500_returns.csv in the app folder.")
        st.stop()
    if not any(c.lower() in ("date", "close", "bench_ret") for c in df.columns):
        df = pd.read_csv("bse500_returns.csv", skiprows=1)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.set_index("date", inplace=True)
    else:
        st.error("⚠️ CSV must have a Date or date column.")
        st.stop()
    if "bench_ret" in df.columns:
        bench = df["bench_ret"]
    else:
        if "Close" not in df.columns:
            st.error("⚠️ CSV must have a Close column to compute returns.")
            st.stop()
        bench = df["Close"].pct_change()
    return bench.dropna()

benchmark = load_benchmark()

# --- Fetch and map all schemes ---
@st.cache_data
def fetch_scheme_list():
    r = requests.get("https://api.mfapi.in/mf")
    return r.json() if r.status_code == 200 else []

schemes = fetch_scheme_list()
fund_mapping = {
    s["schemeName"]: s["schemeCode"]
    for s in schemes
    if s.get("schemeName") and s.get("schemeCode")
}

# --- UI: Fund selector + weight sliders ---
st.markdown("### 🔍 Search and select mutual funds")
selected = st.multiselect(
    "Start typing fund name…",
    options=list(fund_mapping.keys()),
    max_selections=10,
)
codes = [fund_mapping[name] for name in selected]

st.sidebar.header("🎯 Metric Weights (must sum to 1)")
w_roll  = st.sidebar.slider("Rolling Return (5Y)",     0.0, 1.0, 0.15)
w_sd    = st.sidebar.slider("Standard Deviation",      0.0, 1.0, 0.20)
w_shp   = st.sidebar.slider("Sharpe Ratio",            0.0, 1.0, 0.05)
w_srt   = st.sidebar.slider("Sortino Ratio",           0.0, 1.0, 0.20)
w_dnc   = st.sidebar.slider("Downside Capture (%)",    0.0, 1.0, 0.20)
w_ups   = st.sidebar.slider("Upside Capture (%)",      0.0, 1.0, 0.05)
w_alpha = st.sidebar.slider("Alpha (annual %)",        0.0, 1.0, 0.10)
w_beta  = st.sidebar.slider("Beta (sensitivity)",      0.0, 1.0, 0.05)

total = sum([w_roll, w_sd, w_shp, w_srt, w_dnc, w_ups, w_alpha, w_beta])
if not np.isclose(total, 1.0):
    st.sidebar.error(f"⚠️ Weights sum to {total:.2f}, must be 1.0")
    st.stop()

weights = {
    "Rolling Return":    w_roll,
    "SD":                w_sd,
    "Sharpe":            w_shp,
    "Sortino":           w_srt,
    "Downside Capture":  w_dnc,
    "Upside Capture":    w_ups,
    "Alpha":             w_alpha,
    "Beta":              w_beta,
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
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    return df.dropna(subset=["nav"]).set_index("date").sort_index()

# --- Compute metrics with proper alpha & beta ---
def compute_metrics(df_nav):
    if df_nav is None or len(df_nav) < 5*252:
        return None

    df5  = df_nav.iloc[-5*252:]
    rets = df5["nav"].pct_change().dropna()

    rr     = (df5["nav"].iloc[-1]/df5["nav"].iloc[0])**(1/5) - 1
    sd_ann = rets.std()*np.sqrt(252)
    sharpe = (rets.mean()*252 - 0.05)/sd_ann if sd_ann>0 else np.nan
    dn     = rets[rets<0]
    sortino= (rets.mean()*252 - 0.05)/(dn.std()*np.sqrt(252)) if not dn.empty else np.nan

    bench = benchmark.reindex(rets.index).dropna()
    fund  = rets.reindex(bench.index)

    up    = fund[bench>0]
    dnpt  = fund[bench<0]
    up_cap   = (up.mean()/bench[bench>0].mean())*100 if not up.empty else np.nan
    down_cap = (dnpt.mean()/bench[bench<0].mean())*100 if not dnpt.empty else np.nan

    # regression for beta & alpha
    slope, intercept, *_ = linregress(bench.values, fund.values)
    beta  = slope
    alpha = intercept * 252 * 100

    return {
        "Rolling Return":   rr*100,
        "SD":               sd_ann*100,
        "Sharpe":           sharpe,
        "Sortino":          sortino,
        "Downside Capture": down_cap,
        "Upside Capture":   up_cap,
        "Alpha":            alpha,
        "Beta":             beta,
    }

# --- Gather metrics ---
rows = []
for name, code in fund_mapping.items():
    if code in codes:
        nav = fetch_nav(code)
        m   = compute_metrics(nav)
        if m:
            m["Fund Name"] = name
            rows.append(m)

# --- Display ranked table ---
if rows:
    df = pd.DataFrame(rows)

    # normalize & weighted scoring (0–10)
    for metric, w in weights.items():
        vals = df[metric]
        if metric in ("SD", "Beta", "Downside Capture"):
            norm = (vals.max()-vals)/(vals.max()-vals.min())
        else:
            norm = (vals-vals.min())/(vals.max()-vals.min())
        df[f"{metric}_score"] = norm * w

    df["Score"] = (df.filter(like="_score").sum(axis=1)*10).round(2)

    # select and rename columns
    cols = ["Fund Name"] + list(weights.keys()) + ["Score"]
    display = df[cols].round(2).sort_values("Score", ascending=False).reset_index(drop=True)
    display.insert(0, "SL No", range(1, len(display)+1))
    display = display.rename(columns={"Score": "Score (Out of 10)"})

    st.dataframe(display.style.hide_index(), use_container_width=True)
else:
    st.info("🔔 Select at least one fund above to see metrics.")
