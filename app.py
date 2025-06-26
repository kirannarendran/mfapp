import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Ranker", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# --- Fetch Fund List ---
st.info("Fetching list of mutual fund schemes...")
fund_list_url = "https://api.mfapi.in/mf"
response = requests.get(fund_list_url)
fund_list = response.json() if response.status_code == 200 else []

# --- Build Autocomplete Fund Picker Safely ---
fund_mapping = {
    fund.get("scheme_name"): fund.get("scheme_code")
    for fund in fund_list
    if fund.get("scheme_name") and fund.get("scheme_code")
}

# --- Fund selection ---
selected_funds = st.multiselect(
    "Select Mutual Funds to Analyze",
    options=list(fund_mapping.keys()),
    max_selections=10,
)

# --- Scoring weights ---
st.sidebar.header("Metric Weights")
weights = {
    "Rolling Return %": st.sidebar.slider("Rolling Return %", 0, 100, 20),
    "Standard Deviation %": st.sidebar.slider("Standard Deviation %", 0, 100, 20),
    "Sharpe Ratio": st.sidebar.slider("Sharpe Ratio", 0, 100, 15),
    "Sortino Ratio": st.sidebar.slider("Sortino Ratio", 0, 100, 20),
    "Downside Capture %": st.sidebar.slider("Downside Capture %", 0, 100, 20),
    "Alpha": st.sidebar.slider("Alpha", 0, 100, 5),
}
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}  # Normalize

# --- Benchmark Return (Hardcoded) ---
benchmark_return_5y = 0.2359  # 23.59% over 5 years

# --- Utility functions ---
def compute_metrics(nav_df):
    nav_df = nav_df.copy()
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    nav_df.sort_index(inplace=True)
    nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
    nav_df.dropna(inplace=True)

    if len(nav_df) < 5 * 252:
        return None

    nav_df = nav_df[-5 * 252:]  # Use last 5 years
    returns = nav_df['nav'].pct_change().dropna()

    rolling_return = (nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0])**(1/5) - 1
    std_dev = returns.std() * np.sqrt(252)
    sharpe = (rolling_return - 0.05) / std_dev if std_dev != 0 else 0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
    sortino = (rolling_return - 0.05) / downside_std if downside_std != 0 else 0

    downside_benchmark = benchmark_return_5y if benchmark_return_5y < rolling_return else rolling_return
    downside_capture = (rolling_return / downside_benchmark) * 100 if downside_benchmark != 0 else 0

    # Alpha using linear regression
    x = np.arange(len(nav_df))
    y = nav_df['nav'].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    alpha = slope

    return {
        "Rolling Return %": rolling_return * 100,
        "Standard Deviation %": std_dev * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Downside Capture %": downside_capture,
        "Alpha": alpha,
    }

# --- Process Selected Funds ---
all_metrics = []
fund_names = []

for fund_name in selected_funds:
    code = fund_mapping[fund_name]
    url = f"https://api.mfapi.in/mf/{code}/latest"
    res = requests.get(url)
    if res.status_code != 200:
        continue
    data = res.json()
    if not data.get("data"):
        continue
    nav_df = pd.DataFrame(data['data'])
    metrics = compute_metrics(nav_df)
    if metrics:
        metrics["Fund Name"] = fund_name
        all_metrics.append(metrics)

# --- Display Table ---
if all_metrics:
    df = pd.DataFrame(all_metrics)

    # Compute weighted score (0–10 scale)
    for col in weights:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max - col_min != 0:
            df[col + '_norm'] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col + '_norm'] = 0.5  # fallback mid value

    df['Score'] = sum(df[col + '_norm'] * weight for col, weight in weights.items())
    df['Score'] = (df['Score'] * 10).round(2)

    df.drop(columns=[col + '_norm' for col in weights], inplace=True)

    df = df.round(2)
    df.sort_values("Score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.insert(0, "SL No", df.index)

    st.markdown("### 🏆 Fund Rankings")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Select at least one valid fund to view metrics.")
