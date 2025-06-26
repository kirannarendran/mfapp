import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import linregress

# Set Streamlit page config
st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")

# Load benchmark data (CSV file with 'Date' and 'Close')
benchmark_df = pd.read_csv("bse500_returns.csv", parse_dates=["Date"])
benchmark_df = benchmark_df.sort_values("Date")
benchmark_df["Return"] = benchmark_df["Close"].pct_change()
benchmark_cagr = (
    (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[0]) ** (1 / (len(benchmark_df) / 252)) - 1
) * 100

# Helper: compute all metrics
def compute_metrics(df, benchmark_returns):
    returns = df["Nav"].pct_change().dropna()
    benchmark_returns = benchmark_returns.loc[returns.index].dropna()

    excess_returns = returns - benchmark_returns

    std_dev = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    sortino = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252)

    downside_capture = (
        returns[benchmark_returns < 0].mean() / benchmark_returns[benchmark_returns < 0].mean()
    ) * 100
    upside_capture = (
        returns[benchmark_returns > 0].mean() / benchmark_returns[benchmark_returns > 0].mean()
    ) * 100

    beta, alpha, *_ = linregress(benchmark_returns.values, returns.values)
    alpha = (alpha * 252) * 100
    beta = round(beta, 2)

    cagr = (
        (df["Nav"].iloc[-1] / df["Nav"].iloc[0]) ** (1 / (len(df) / 252)) - 1
    ) * 100

    return {
        "Rolling Return (CAGR)": round(cagr, 2),
        "Standard Deviation": round(std_dev, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Alpha": round(alpha, 2),
        "Beta": round(beta, 2),
        "Upside Capture": round(upside_capture, 2),
        "Downside Capture": round(downside_capture, 2),
    }

# Load mutual fund list
@st.cache_data
def get_funds():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    return response.json()

fund_list = get_funds()
fund_mapping = {fund["scheme_name"]: fund["scheme_code"] for fund in fund_list}

# UI: Fund selection
st.title("📊 Mutual Fund Ranking Tool")
selected_funds = st.multiselect("🔍 Search and select mutual funds", options=list(fund_mapping.keys()))

# Weight sliders
st.sidebar.markdown("### Adjust Metric Weights (%)")
weights = {
    "Standard Deviation": st.sidebar.slider("Standard Deviation Weight (%)", 0, 100, 20),
    "Downside Capture": st.sidebar.slider("Downside Capture Weight (%)", 0, 100, 20),
    "Rolling Return (CAGR)": st.sidebar.slider("CAGR Weight (%)", 0, 100, 15),
    "Sharpe Ratio": st.sidebar.slider("Sharpe Ratio Weight (%)", 0, 100, 10),
    "Upside Capture": st.sidebar.slider("Upside Capture Weight (%)", 0, 100, 10),
    "Alpha": st.sidebar.slider("Alpha Weight (%)", 0, 100, 5),
}
total_weight = sum(weights.values())

# Prepare benchmark returns aligned to trading days
benchmark_df = benchmark_df.set_index("Date")
benchmark_returns = benchmark_df["Return"]

# Compute metrics
rows = []
for fund_name in selected_funds:
    try:
        code = fund_mapping[fund_name]
        url = f"https://api.mfapi.in/mf/{code}"
        r = requests.get(url).json()
        data = r.get("data", [])

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["Nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna().set_index("date").sort_index()

        aligned_benchmark = benchmark_returns[df.index.min():df.index.max()]
        metrics = compute_metrics(df, aligned_benchmark)

        score = sum((metrics[k] / 100) * w for k, w in weights.items())
        score = round(min(score, 10), 2)

        rows.append({
            "Fund": fund_name,
            **metrics,
            "Score (Out of 10)": score
        })
    except Exception as e:
        st.warning(f"⚠️ Error loading data for {fund_name}: {e}")

# Append benchmark row
benchmark_metrics = {
    "Fund": "**BSE 500 Index (Benchmark)**",
    "Rolling Return (CAGR)": round(benchmark_cagr, 2),
    "Standard Deviation": round(benchmark_df["Return"].std() * np.sqrt(252) * 100, 2),
    "Sharpe Ratio": round((benchmark_df["Return"].mean() / benchmark_df["Return"].std()) * np.sqrt(252), 2),
    "Sortino Ratio": round((benchmark_df["Return"].mean() / benchmark_df["Return"][benchmark_df["Return"] < 0].std()) * np.sqrt(252), 2),
    "Alpha": "",
    "Beta": "",
    "Upside Capture": 100.00,
    "Downside Capture": 100.00,
    "Score (Out of 10)": ""
}
rows.append(benchmark_metrics)

# Show table
if rows:
    df = pd.DataFrame(rows)
    df.insert(0, "SL No", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True)
