import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

# Load benchmark data
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv("bse500_returns.csv", parse_dates=["Date"])
        df = df.sort_values("Date")
        df["Returns"] = df["Close"].pct_change()
        return df.dropna()
    except Exception as e:
        st.error(f"⚠️ Error loading benchmark CSV: {e}")
        return pd.DataFrame()

benchmark_df = load_benchmark()

# Fetch fund list
@st.cache_data
def fetch_funds():
    try:
        response = requests.get("https://api.mfapi.in/mf")
        return response.json()
    except Exception as e:
        st.error(f"⚠️ Failed to fetch fund list: {e}")
        return []

fund_list = fetch_funds()
fund_mapping = {fund["scheme_name"]: fund["scheme_code"] for fund in fund_list if "scheme_name" in fund}

# Fund selection – nothing pre-selected
valid_fund_names = sorted(fund_mapping.keys())
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=valid_fund_names,
    default=[],
    help="Type to search and select one or more mutual funds"
)

# Fetch NAVs
@st.cache_data
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        response = requests.get(url)
        data = response.json()
        nav_df = pd.DataFrame(data.get("data", []))
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df["nav"] = pd.to_numeric(nav_df["nav"], errors="coerce")
        nav_df = nav_df.dropna()
        nav_df = nav_df.sort_values("date")
        return nav_df
    except Exception:
        return pd.DataFrame()

# Compute metrics
def compute_metrics(nav_df):
    merged = pd.merge(nav_df, benchmark_df, left_on="date", right_on="Date")
    merged["fund_return"] = merged["nav"].pct_change()
    merged["benchmark_return"] = merged["Returns"]

    merged = merged.dropna()
    rets = merged["fund_return"]
    bench = merged["benchmark_return"]

    avg_return = rets.mean() * 252
    volatility = rets.std() * np.sqrt(252)
    downside = rets[rets < 0].std() * np.sqrt(252)
    sharpe = avg_return / volatility if volatility != 0 else 0
    sortino = avg_return / downside if downside != 0 else 0

    up = bench > 0
    down = bench < 0

    upside = rets[up].mean() / bench[up].mean() if up.any() else np.nan
    downside = rets[down].mean() / bench[down].mean() if down.any() else np.nan

    if len(bench.unique()) == 1:
        alpha, beta = np.nan, np.nan
    else:
        slope, intercept, *_ = linregress(bench, rets)
        beta = slope
        alpha = (rets.mean() - slope * bench.mean()) * 252

    return {
        "Return": avg_return * 100,
        "Volatility": volatility * 100,
        "Downside Risk": downside * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Upside Capture": upside,
        "Downside Capture": downside,
        "Alpha": alpha,
        "Beta": beta
    }

# Main Table
def build_table():
    rows = []
    for name in selected_funds:
        code = fund_mapping.get(name)
        nav = fetch_nav(code)
        if nav.empty:
            continue
        metrics = compute_metrics(nav)
        metrics["Fund"] = name
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Score (Out of 10)"] = (
        df[["Sharpe", "Sortino"]].mean(axis=1)
        + (df["Upside Capture"] - df["Downside Capture"]).fillna(0) / 2
    ).clip(upper=10).round(2)

    df.insert(0, "SL No", range(1, len(df) + 1))

    df.sort_values("Score (Out of 10)", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Rank"] = df.index + 1

    return df

# Benchmark Metrics
def benchmark_metrics():
    bench_rets = benchmark_df["Returns"].dropna()
    return {
        "Fund": "BSE 500 (Benchmark)",
        "Return": bench_rets.mean() * 252 * 100,
        "Volatility": bench_rets.std() * np.sqrt(252) * 100,
        "Downside Risk": bench_rets[bench_rets < 0].std() * np.sqrt(252) * 100,
        "Sharpe": (bench_rets.mean() / bench_rets.std()) if bench_rets.std() != 0 else 0,
        "Sortino": (bench_rets.mean() / bench_rets[bench_rets < 0].std()) if bench_rets[bench_rets < 0].std() != 0 else 0,
        "Upside Capture": np.nan,
        "Downside Capture": np.nan,
        "Alpha": np.nan,
        "Beta": np.nan,
        "Score (Out of 10)": np.nan,
        "SL No": "",
        "Rank": ""
    }

# Show Table
st.title("📊 Mutual Fund Ranker")
if selected_funds:
    final = build_table()
    bench_row = pd.DataFrame([benchmark_metrics()])
    final = pd.concat([final, bench_row], ignore_index=True)

    metrics_to_display = [
        "SL No", "Fund", "Return", "Volatility", "Downside Risk", "Sharpe", "Sortino",
        "Upside Capture", "Downside Capture", "Alpha", "Beta", "Score (Out of 10)", "Rank"
    ]

    display = final[metrics_to_display]
    display = display.round(2)
    display.loc[display["Fund"] == "BSE 500 (Benchmark)", "Fund"] = "**BSE 500 (Benchmark)**"
    st.dataframe(display.style.hide_index(), use_container_width=True)
else:
    st.info("Please search and select mutual funds to begin.")
