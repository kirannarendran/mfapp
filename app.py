import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import linregress

# ---- CONFIG ----
st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# ---- SIDEBAR WEIGHTS ----
st.sidebar.header("Adjust Metric Weights (%)")

weights = {
    "std": st.sidebar.slider("Standard Deviation Weight (%)", 0, 100, 20),
    "downside": st.sidebar.slider("Downside Capture Weight (%)", 0, 100, 20),
    "cagr": st.sidebar.slider("CAGR Weight (%)", 0, 100, 15),
    "sharpe": st.sidebar.slider("Sharpe Ratio Weight (%)", 0, 100, 10),
    "upside": st.sidebar.slider("Upside Capture Weight (%)", 0, 100, 10),
    "alpha": st.sidebar.slider("Alpha Weight (%)", 0, 100, 5),
}

# ---- BENCHMARK DATA ----
@st.cache_data
def load_benchmark():
    df = pd.read_csv("bse500_returns.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

benchmark_df = load_benchmark()
benchmark_cagr = ((benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[0]) ** 
                  (1 / (len(benchmark_df) / 252))) - 1
benchmark_std = benchmark_df["Return"].std() * np.sqrt(252)
benchmark_downside = benchmark_df[benchmark_df["Return"] < 0]["Return"].std() * np.sqrt(252)
benchmark_sharpe = benchmark_cagr / benchmark_std
benchmark_sortino = benchmark_cagr / benchmark_downside

benchmark_metrics = {
    "Fund": "**BSE 500 Index (Benchmark)**",
    "Rolling Return (CAGR)": round(benchmark_cagr * 100, 2),
    "Standard Deviation": round(benchmark_std * 100, 2),
    "Sharpe Ratio": round(benchmark_sharpe, 2),
    "Sortino Ratio": round(benchmark_sortino, 2),
    "Alpha": "",
    "Beta": "",
    "Upside Capture": 100,
    "Downside Capture": 100,
    "Score (Out of 10)": ""
}

# ---- MUTUAL FUND LIST ----
@st.cache_data
def get_fund_mapping():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    data = response.json()
    mapping = {
        fund["schemeName"]: fund["schemeCode"]
        for fund in data
        if any(keyword in fund["schemeName"].lower() for keyword in ["flexi", "parag", "quant", "hdfc"])
    }
    return mapping

fund_mapping = get_fund_mapping()

# ---- USER FUND SELECTION ----
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=list(fund_mapping.keys()),
    default=[
        name for name in fund_mapping.keys()
        if any(k in name.lower() for k in ["parag", "quant", "hdfc"])
    ]
)

# ---- FETCH NAV & METRICS ----
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()
    nav_data = data.get("data", [])
    df = pd.DataFrame(nav_data)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("date", inplace=True)
    df["return"] = df["nav"].pct_change()
    df.dropna(inplace=True)
    return df

def compute_metrics(df, benchmark_df):
    merged = pd.merge(df, benchmark_df[["Date", "Return"]], left_on="date", right_on="Date", how="inner")
    fund_returns = merged["return"]
    benchmark_returns = merged["Return"]

    cagr = ((df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1 / (len(df) / 252))) - 1
    std = fund_returns.std() * np.sqrt(252)
    downside = fund_returns[fund_returns < 0].std() * np.sqrt(252)
    sharpe = cagr / std if std else 0
    sortino = cagr / downside if downside else 0

    slope, _, r_value, _, _ = linregress(benchmark_returns, fund_returns)
    beta = r_value
    alpha = (cagr - (benchmark_cagr * beta)) * 100

    upside = (fund_returns[benchmark_returns > 0].mean() /
              benchmark_returns[benchmark_returns > 0].mean()) * 100
    downside_capture = (fund_returns[benchmark_returns < 0].mean() /
                        benchmark_returns[benchmark_returns < 0].mean()) * 100

    return {
        "Rolling Return (CAGR)": round(cagr * 100, 2),
        "Standard Deviation": round(std * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Alpha": round(alpha, 2),
        "Beta": round(beta, 2),
        "Upside Capture": round(upside, 2),
        "Downside Capture": round(downside_capture, 2)
    }

# ---- PROCESS ALL FUNDS ----
rows = []
for name in selected_funds:
    code = fund_mapping[name]
    try:
        nav = fetch_nav(code)
        metrics = compute_metrics(nav, benchmark_df)
        score = (
            metrics["Standard Deviation"] * weights["std"] +
            metrics["Downside Capture"] * weights["downside"] +
            metrics["Rolling Return (CAGR)"] * weights["cagr"] +
            metrics["Sharpe Ratio"] * weights["sharpe"] +
            metrics["Upside Capture"] * weights["upside"] +
            metrics["Alpha"] * weights["alpha"]
        ) / 100
        row = {
            "Fund": name,
            **metrics,
            "Score (Out of 10)": round(score, 2)
        }
        rows.append(row)
    except Exception as e:
        st.error(f"Error loading {name}: {e}")

# ---- DISPLAY TABLE ----
df = pd.DataFrame(rows)
df = df.sort_values("Score (Out of 10)", ascending=False).reset_index(drop=True)
df.index += 1
df.insert(0, "SL No", df.index)

# Add benchmark row
benchmark_row = pd.DataFrame([benchmark_metrics])
benchmark_row["SL No"] = len(df) + 1
df = pd.concat([df, benchmark_row], ignore_index=True)

# Drop index before SL No
df = df.loc[:, ~df.columns.duplicated()]

# Show final table
st.dataframe(df, use_container_width=True)
