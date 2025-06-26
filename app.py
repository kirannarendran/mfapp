import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Flexi Cap Metrics", layout="wide")
st.title("📊 Mutual Fund Flexi Cap Metrics")

@st.cache_data(show_spinner=False)
def fetch_scheme_list():
    url = "https://api.mfapi.in/mf"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

@st.cache_data(show_spinner=False)
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        response = requests.get(url)
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df.dropna(inplace=True)
        df.sort_values('date', inplace=True)
        return df[['date', 'nav']].copy()
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_benchmark_returns():
    try:
        df = pd.read_csv("bse500_returns.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", inplace=True)
        df['Return'] = df['Close'].pct_change()
        return df[['Date', 'Close', 'Return']].dropna()
    except Exception as e:
        st.error(f"⚠️ Error loading benchmark CSV: {e}")
        return pd.DataFrame()

def compute_metrics(nav_df, benchmark_df=None):
    nav_df = nav_df.copy()
    nav_df['Return'] = nav_df['nav'].pct_change()
    nav_df.dropna(inplace=True)

    merged = pd.merge(nav_df, benchmark_df, left_on='date', right_on='Date', suffixes=('', '_bmk'))
    merged.dropna(inplace=True)

    rets = merged['Return']
    bench_rets = merged['Return_bmk']

    CAGR = (nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0]) ** (1 / ((nav_df['date'].iloc[-1] - nav_df['date'].iloc[0]).days / 365)) - 1
    std = rets.std() * np.sqrt(252)
    downside_std = rets[rets < 0].std() * np.sqrt(252)
    sharpe = CAGR / std if std else 0
    sortino = CAGR / downside_std if downside_std else 0

    # Regression for alpha and beta
    if (bench_rets == bench_rets.iloc[0]).all():
        alpha, beta = 0.0, 1.0
    else:
        slope, intercept, *_ = linregress(bench_rets, rets)
        beta = slope
        alpha = CAGR - beta * ((benchmark_df['Close'].iloc[-1] / benchmark_df['Close'].iloc[0]) ** (1 / ((benchmark_df['Date'].iloc[-1] - benchmark_df['Date'].iloc[0]).days / 365)) - 1)

    # Upside/downside capture
    up_capture = rets[bench_rets > 0].mean() / bench_rets[bench_rets > 0].mean() * 100 if not bench_rets[bench_rets > 0].empty else 0
    down_capture = rets[bench_rets < 0].mean() / bench_rets[bench_rets < 0].mean() * 100 if not bench_rets[bench_rets < 0].empty else 0

    return {
        'Rolling Return (CAGR)': CAGR * 100,
        'Standard Deviation': std * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Alpha': alpha * 100,
        'Beta': beta,
        'Upside Capture': up_capture,
        'Downside Capture': down_capture
    }

fund_list = fetch_scheme_list()
fund_mapping = {fund['scheme_name']: fund['scheme_code'] for fund in fund_list}

selected_funds = st.multiselect("Select Funds", options=list(fund_mapping.keys()))
benchmark_df = load_benchmark_returns()

weightings = {
    'Sortino Ratio': st.slider("Sortino Ratio Weight (%)", 0, 100, 20),
    'Standard Deviation': st.slider("Standard Deviation Weight (%)", 0, 100, 20),
    'Downside Capture': st.slider("Downside Capture Weight (%)", 0, 100, 20),
    'Rolling Return (CAGR)': st.slider("CAGR Weight (%)", 0, 100, 15),
    'Sharpe Ratio': st.slider("Sharpe Ratio Weight (%)", 0, 100, 10),
    'Upside Capture': st.slider("Upside Capture Weight (%)", 0, 100, 10),
    'Alpha': st.slider("Alpha Weight (%)", 0, 100, 5)
}

if sum(weightings.values()) != 100:
    st.warning("Weights must sum to 100%")

results = []
for fund in selected_funds:
    nav_df = fetch_nav(fund_mapping[fund])
    if nav_df.empty or benchmark_df.empty:
        continue
    metrics = compute_metrics(nav_df, benchmark_df)
    score = 0
    for k, w in weightings.items():
        norm = abs(metrics[k])
        score += (norm * w) / 100
    results.append({"Fund": fund, **metrics, "Score (Out of 10)": score})

if results:
    df = pd.DataFrame(results)
    df = df.round(2)
    df.sort_values("Score (Out of 10)", ascending=False, inplace=True)
    df.insert(0, "SL No", range(1, len(df) + 1))

    # Append benchmark row
    bench_nav = benchmark_df[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'nav'})
    bench_metrics = compute_metrics(bench_nav, benchmark_df)
    benchmark_row = {"Fund": "BSE 500 Index (Benchmark)", **bench_metrics, "Score (Out of 10)": 10.00}
    benchmark_row = pd.DataFrame([benchmark_row])
    benchmark_row.insert(0, "SL No", len(df) + 1)
    df = pd.concat([df, benchmark_row], ignore_index=True)

    st.dataframe(df.style.format(precision=2).highlight_max(axis=0), use_container_width=True)
else:
    st.info("Please select valid funds to compute metrics.")
