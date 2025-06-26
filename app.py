import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from datetime import datetime

st.set_page_config(layout="wide")

st.title("📈 Mutual Fund Ranker - Flexi Cap Category")

# Fetch and cache full mutual fund list
@st.cache_data
def fetch_fund_list():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else []

fund_list = fetch_fund_list()

# Create mapping of scheme name -> code
fund_mapping = {fund['scheme_name']: fund['scheme_code'] for fund in fund_list}
fund_names = list(fund_mapping.keys())

# Fund selection UI
selected_funds = st.multiselect("Enter mutual fund scheme codes from MFAPI.in, one per line:", fund_names)
scheme_codes = [fund_mapping[name] for name in selected_funds if name in fund_mapping]

with st.expander("🎯 Adjust Metric Weights"):
    sortino_weight = st.slider("Sortino Ratio Weight", 0.0, 1.0, 0.20)
    downside_weight = st.slider("Downside Capture Weight", 0.0, 1.0, 0.20)
    sd_weight = st.slider("Standard Deviation (SD) Weight", 0.0, 1.0, 0.20)
    rolling_weight = st.slider("Rolling Return (5Y) Weight", 0.0, 1.0, 0.15)
    alpha_weight = st.slider("Alpha Weight", 0.0, 1.0, 0.10)
    sharpe_weight = st.slider("Sharpe Ratio Weight", 0.0, 1.0, 0.05)
    upside_weight = st.slider("Upside Capture Weight", 0.0, 1.0, 0.05)
    beta_weight = st.slider("Beta Weight", 0.0, 1.0, 0.05)

    total = sum([
        sortino_weight, downside_weight, sd_weight,
        rolling_weight, alpha_weight, sharpe_weight,
        upside_weight, beta_weight
    ])

    if total != 1.0:
        st.error("⚠️ Total weight must be 1.0. Please adjust sliders.")
        st.stop()

@st.cache_data

def fetch_benchmark():
    url = "https://query1.finance.yahoo.com/v7/finance/download/^BSE500?period1=1514764800&period2=1706659200&interval=1d&events=history&includeAdjustedClose=true"
    df = pd.read_csv(url, parse_dates=['Date'])
    df = df[['Date', 'Adj Close']].rename(columns={'Adj Close': 'Benchmark'})
    df['Benchmark'] = df['Benchmark'].pct_change()
    df = df.dropna()
    return df

benchmark = fetch_benchmark()

@st.cache_data

def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json().get("data", [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df = df.dropna()
    df = df.sort_values('date')
    df['return'] = df['nav'].pct_change()
    return df[['date', 'return']].dropna()


def calculate_metrics(scheme_code):
    nav_df = fetch_nav(scheme_code)
    if nav_df is None:
        return None

    merged = pd.merge(nav_df, benchmark, left_on='date', right_on='Date')
    merged = merged.dropna()

    rolling_return = ((1 + merged['return']).prod() ** (252 / len(merged))) - 1
    sd = np.std(merged['return']) * np.sqrt(252)
    sharpe = (np.mean(merged['return']) * 252) / (np.std(merged['return']) * np.sqrt(252))
    downside = merged[merged['Benchmark'] < 0]
    downside_capture = (downside['return'].mean() / downside['Benchmark'].mean()) * 100 if not downside.empty else np.nan
    upside = merged[merged['Benchmark'] > 0]
    upside_capture = (upside['return'].mean() / upside['Benchmark'].mean()) * 100 if not upside.empty else np.nan
    sortino = (merged['return'].mean() * 252) / (np.std(downside['return']) * np.sqrt(252)) if not downside.empty else np.nan
    slope, intercept, r_value, p_value, std_err = linregress(merged['Benchmark'], merged['return'])
    beta = slope
    alpha = (intercept * 252) * 100

    return {
        "Scheme Code": scheme_code,
        "Fund Name": [k for k, v in fund_mapping.items() if v == scheme_code][0],
        "Rolling Return (5Y) %": rolling_return * 100,
        "SD %": sd * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Alpha %": alpha,
        "Beta": beta,
        "Upside Capture %": upside_capture,
        "Downside Capture %": downside_capture
    }

if scheme_codes:
    st.markdown("### 🏆 Ranked Mutual Funds")
    results = []
    for code in scheme_codes:
        metrics = calculate_metrics(code)
        if metrics:
            results.append(metrics)

    if results:
        df = pd.DataFrame(results)

        # Normalize scores and calculate total score
        metric_columns = [
            "Sortino", "Downside Capture %", "SD %", "Rolling Return (5Y) %",
            "Alpha %", "Sharpe", "Upside Capture %", "Beta"
        ]

        weights = [
            sortino_weight, downside_weight, sd_weight, rolling_weight,
            alpha_weight, sharpe_weight, upside_weight, beta_weight
        ]

        df_scores = df[metric_columns].copy()

        for i, col in enumerate(metric_columns):
            if col == "SD %" or col == "Beta":
                df_scores[col] = 1 - (df_scores[col] - df_scores[col].min()) / (df_scores[col].max() - df_scores[col].min())
            else:
                df_scores[col] = (df_scores[col] - df_scores[col].min()) / (df_scores[col].max() - df_scores[col].min())

        df["Score"] = df_scores.mul(weights).sum(axis=1) * 10
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1

        display_cols = [
            "Fund Name", "Scheme Code", "Rolling Return (5Y) %", "SD %", "Sharpe",
            "Sortino", "Alpha %", "Beta", "Upside Capture %", "Downside Capture %", "Score"
        ]

        df = df[display_cols].round(2)
        st.dataframe(df, use_container_width=True)
