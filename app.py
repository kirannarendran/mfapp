import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import linregress

# Streamlit page config
st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# Load benchmark data from local CSV
benchmark_df = pd.read_csv("bse_500_benchmark.csv", parse_dates=["date"])
benchmark_df.set_index("date", inplace=True)
benchmark_df = benchmark_df.sort_index()
benchmark_df["returns"] = benchmark_df["nav"].pct_change()
benchmark_cagr = (
    (benchmark_df["nav"].iloc[-1] / benchmark_df["nav"].iloc[0]) ** (1 / ((benchmark_df.index[-1] - benchmark_df.index[0]).days / 365.25)) - 1
) * 100

def compute_metrics(nav_df, benchmark_returns):
    nav_df = nav_df.copy()
    nav_df["returns"] = nav_df["nav"].pct_change()
    nav_df = nav_df.dropna()

    daily_returns = nav_df["returns"]
    excess_returns = daily_returns - benchmark_returns.loc[daily_returns.index].fillna(0)

    cagr = (
        (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0]) ** (1 / ((nav_df.index[-1] - nav_df.index[0]).days / 365.25)) - 1
    ) * 100
    std_dev = daily_returns.std() * np.sqrt(252) * 100
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    sortino = (daily_returns.mean() / daily_returns[daily_returns < 0].std()) * np.sqrt(252)
    downside_std = daily_returns[daily_returns < 0].std() * np.sqrt(252) * 100
    upside_capture = (daily_returns[benchmark_returns > 0].mean() / benchmark_returns[benchmark_returns > 0].mean()) * 100
    downside_capture = (daily_returns[benchmark_returns < 0].mean() / benchmark_returns[benchmark_returns < 0].mean()) * 100
    beta, alpha, _, _, _ = linregress(benchmark_returns.loc[daily_returns.index], daily_returns)

    return {
        "Rolling Return (CAGR)": round(cagr, 2),
        "Standard Deviation": round(std_dev, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Alpha": round(alpha * 100, 2),
        "Beta": round(beta, 2),
        "Upside Capture": round(upside_capture, 2),
        "Downside Capture": round(downside_capture, 2)
    }

# Metric Weights
with st.sidebar:
    st.subheader("Adjust Metric Weights (%)")
    weight_std = st.slider("Standard Deviation Weight (%)", 0, 100, 20)
    weight_down = st.slider("Downside Capture Weight (%)", 0, 100, 20)
    weight_cagr = st.slider("CAGR Weight (%)", 0, 100, 15)
    weight_sharpe = st.slider("Sharpe Ratio Weight (%)", 0, 100, 10)
    weight_up = st.slider("Upside Capture Weight (%)", 0, 100, 10)
    weight_alpha = st.slider("Alpha Weight (%)", 0, 100, 5)

# Fetch list of funds from API
@st.cache_data
def fetch_fund_list():
    url = "https://api.mfapi.in/mf"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        st.error("Failed to fetch fund list")
        return []

fund_list = fetch_fund_list()
fund_mapping = {fund['scheme_name']: fund['scheme_code'] for fund in fund_list}
selected_funds = st.multiselect("🔍 Search and select mutual funds", options=list(fund_mapping.keys()), default=[])

# Score calculation
def compute_score(metrics):
    score = 0
    score += (metrics["CAGR"] / 20) * weight_cagr
    score += (metrics["Sharpe"] / 1.5) * weight_sharpe
    score += (metrics["Sortino"] / 1.5) * 5
    score += (metrics["Upside"] / 100) * weight_up
    score += ((100 - metrics["Downside"]) / 100) * weight_down
    score += (metrics["Alpha"] / 10) * weight_alpha
    score += ((100 - metrics["Standard Deviation"]) / 30) * weight_std
    return round(min(score / 10, 10), 2)  # Cap at 10

# Fetch NAV data
@st.cache_data
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.set_index("date").sort_index()
        return df[["nav"]].dropna()
    else:
        return pd.DataFrame()

# Compute metrics for selected funds
if selected_funds:
    rows = []
    benchmark_returns = benchmark_df["returns"]

    for fund in selected_funds:
        nav_df = fetch_nav(fund_mapping[fund])
        if not nav_df.empty:
            metrics = compute_metrics(nav_df, benchmark_returns)
            rows.append({
                "Fund": fund,
                "CAGR": metrics["Rolling Return (CAGR)"],
                "Standard Deviation": metrics["Standard Deviation"],
                "Sharpe": metrics["Sharpe Ratio"],
                "Sortino": metrics["Sortino Ratio"],
                "Alpha": metrics["Alpha"],
                "Beta": metrics["Beta"],
                "Upside": metrics["Upside Capture"],
                "Downside": metrics["Downside Capture"],
                "Score (Out of 10)": compute_score(metrics)
            })

    # Add Benchmark Row
    bench_metrics = compute_metrics(benchmark_df[["nav"]], benchmark_returns)
    rows.append({
        "Fund": "**BSE 500 Index (Benchmark)**",
        "CAGR": round(benchmark_cagr, 2),
        "Standard Deviation": bench_metrics["Standard Deviation"],
        "Sharpe": bench_metrics["Sharpe Ratio"],
        "Sortino": bench_metrics["Sortino Ratio"],
        "Alpha": "",
        "Beta": "",
        "Upside": 100.0,
        "Downside": 100.0,
        "Score (Out of 10)": ""
    })

    df = pd.DataFrame(rows)
    df.index = np.arange(1, len(df) + 1)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "SL No"}, inplace=True)

    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Please search and select one or more mutual funds to begin.")
