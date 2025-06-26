import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Ranker", layout="wide")
st.title("📈 Mutual Fund Ranker - Flexi Cap Category")

# --------------------------
# Fetch Fund List
# --------------------------
@st.cache_data

def get_fund_list():
    response = requests.get("https://api.mfapi.in/mf")
    if response.status_code == 200:
        return response.json()
    else:
        return []

fund_list = get_fund_list()
fund_name_to_code = {f['schemeName']: f['schemeCode'] for f in fund_list}

selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=list(fund_name_to_code.keys()),
    help="Start typing the fund name and select from the suggestions"
)

selected_scheme_codes = [fund_name_to_code[name] for name in selected_funds]

# --------------------------
# Metric Weight Sliders
# --------------------------
with st.expander("🎯 Adjust Metric Weights"):
    sortino_weight = st.slider("Sortino Ratio Weight", 0.0, 1.0, 0.2)
    downside_weight = st.slider("Downside Capture Weight", 0.0, 1.0, 0.2)
    sd_weight = st.slider("Standard Deviation (SD) Weight", 0.0, 1.0, 0.2)
    return_weight = st.slider("Rolling Return (5Y) Weight", 0.0, 1.0, 0.15)
    alpha_weight = st.slider("Alpha Weight", 0.0, 1.0, 0.1)
    sharpe_weight = st.slider("Sharpe Ratio Weight", 0.0, 1.0, 0.05)
    upside_weight = st.slider("Upside Capture Weight", 0.0, 1.0, 0.05)
    beta_weight = st.slider("Beta Weight", 0.0, 1.0, 0.05)

weights = {
    'Sortino': sortino_weight,
    'Downside Capture %': downside_weight,
    'SD %': sd_weight,
    'Rolling Return (5Y) %': return_weight,
    'Alpha %': alpha_weight,
    'Sharpe': sharpe_weight,
    'Upside Capture %': upside_weight,
    'Beta': beta_weight
}

# --------------------------
# Benchmark Return (hardcoded)
# --------------------------
bm_return = 0.2359  # 23.59% for BSE 500

# --------------------------
# Utility Functions
# --------------------------
def get_nav_history(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    if 'data' not in data:
        return None
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df.dropna(inplace=True)
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df

def calculate_metrics(df):
    if df is None or len(df) < 252 * 5:
        return None

    returns = df['nav'].pct_change().dropna()
    annualized_sd = returns.std() * np.sqrt(252)

    downside_returns = returns[returns < 0]
    downside_sd = downside_returns.std() * np.sqrt(252)

    rolling_return = (df['nav'].iloc[-1] / df['nav'].iloc[-252*5])**(1/5) - 1

    excess_returns = returns - bm_return / 252
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    # Benchmark index dummy (constant growth)
    bm_df = df.copy()
    bm_df['bm_nav'] = 100 * (1 + bm_return)**(np.arange(len(bm_df)) / 252)
    bm_returns = bm_df['bm_nav'].pct_change().dropna()

    # Alpha/Beta from regression
    aligned_returns = pd.concat([returns, bm_returns], axis=1).dropna()
    if aligned_returns.empty:
        return None

    slope, intercept, r_value, p_value, std_err = linregress(
        aligned_returns.iloc[:, 1], aligned_returns.iloc[:, 0]
    )

    beta = slope
    alpha = (intercept) * 252

    # Capture Ratios
    upside = aligned_returns[aligned_returns.iloc[:, 1] > 0]
    downside = aligned_returns[aligned_returns.iloc[:, 1] < 0]

    upside_capture = (upside.iloc[:, 0].mean() / upside.iloc[:, 1].mean()) if not upside.empty else np.nan
    downside_capture = (downside.iloc[:, 0].mean() / downside.iloc[:, 1].mean()) if not downside.empty else np.nan

    return {
        'Rolling Return (5Y) %': round(rolling_return * 100, 4),
        'SD %': round(annualized_sd * 100, 4),
        'Sharpe': round(sharpe, 4),
        'Sortino': round(sortino, 4),
        'Alpha %': round(alpha * 100, 4),
        'Beta': round(beta, 4),
        'Upside Capture %': round(upside_capture * 100, 4),
        'Downside Capture %': round(downside_capture * 100, 4)
    }

# --------------------------
# Main Processing
# --------------------------
results = []

for scheme_code in selected_scheme_codes:
    df = get_nav_history(scheme_code)
    metrics = calculate_metrics(df)
    if metrics:
        metrics['Scheme Code'] = scheme_code
        results.append(metrics)

if results:
    df = pd.DataFrame(results)

    # Normalize each metric and calculate score
    df_score = df.copy()
    for metric, weight in weights.items():
        if metric in df_score.columns:
            col = df_score[metric]
            if metric in ['SD %', 'Downside Capture %', 'Beta']:  # Lower is better
                score = (col.max() - col) / (col.max() - col.min())
            else:
                score = (col - col.min()) / (col.max() - col.min())
            df_score[metric + "_score"] = score * weight

    df_score['Total Score'] = df_score[[c for c in df_score.columns if '_score' in c]].sum(axis=1)
    df_score = df_score.sort_values("Total Score", ascending=False)

    st.subheader("🏆 Ranked Mutual Funds")
    st.dataframe(df_score[[
        'Scheme Code', 'Rolling Return (5Y) %', 'SD %', 'Sharpe', 'Sortino',
        'Alpha %', 'Beta', 'Upside Capture %', 'Downside Capture %', 'Total Score'
    ]].reset_index(drop=True), use_container_width=True)
else:
    if selected_scheme_codes:
        st.warning("No valid data returned for the selected funds.")
