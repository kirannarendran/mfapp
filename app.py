import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

# --- Constants --- #
RISK_FREE_RATE = 0.05
BENCHMARK_RETURN_5Y = 0.2359  # 5-year return as decimal
REQUIRED_DAYS = 5 * 252

# --- UI Setup --- #
st.title("📈 Mutual Fund Ranker - Flexi Cap Category")
st.markdown("Enter mutual fund scheme codes (from MFAPI.in), one per line:")

fund_input = st.text_area("Scheme Codes", "120503\n119834\n118834")
fund_codes = [x.strip() for x in fund_input.splitlines() if x.strip()]

st.markdown("### 🎯 Adjust Metric Weights")
weight_sortino = st.slider("Sortino Ratio Weight", 0.0, 1.0, 0.20)
weight_downside = st.slider("Downside Capture Weight", 0.0, 1.0, 0.20)
weight_sd = st.slider("Standard Deviation (SD) Weight", 0.0, 1.0, 0.20)
weight_rolling = st.slider("Rolling Return (5Y) Weight", 0.0, 1.0, 0.15)
weight_alpha = st.slider("Alpha Weight", 0.0, 1.0, 0.10)
weight_sharpe = st.slider("Sharpe Ratio Weight", 0.0, 1.0, 0.05)
weight_upside = st.slider("Upside Capture Weight", 0.0, 1.0, 0.05)
weight_beta = st.slider("Beta Weight", 0.0, 1.0, 0.05)

weight_sum = sum([weight_sortino, weight_downside, weight_sd, weight_rolling,
                  weight_alpha, weight_sharpe, weight_upside, weight_beta])

if weight_sum != 1.0:
    st.warning(f"Total weight must equal 1.0. Current total: {weight_sum:.2f}")
    st.stop()

# --- Helper Functions --- #
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url)
    data = r.json().get('data', [])
    if not data:
        raise ValueError("No data returned for scheme")
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df.dropna(subset=['nav'], inplace=True)
    return df.set_index('date').sort_index()

def compute_metrics(df):
    df['ret'] = np.log(df['nav'] / df['nav'].shift(1))
    df.dropna(inplace=True)

    if len(df) < REQUIRED_DAYS:
        raise ValueError("Not enough data (need ~5 years)")

    rolling_return = (df['nav'].iloc[-1] / df['nav'].iloc[-REQUIRED_DAYS])**(1/5) - 1
    sd = df['ret'].std() * np.sqrt(252)
    sharpe = (df['ret'].mean() * 252 - RISK_FREE_RATE) / sd if sd > 0 else np.nan

    downside_std = df['ret'][df['ret'] < 0].std() * np.sqrt(252)
    sortino = (df['ret'].mean() * 252 - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan

    # Simulated benchmark return for Alpha/Beta
    fund_return = df['ret'].mean() * 252
    alpha = fund_return - BENCHMARK_RETURN_5Y
    beta = 1.0  # Placeholder if no benchmark series

    upside = df[df['ret'] > 0]['ret'].mean() * 252
    downside = df[df['ret'] < 0]['ret'].mean() * 252
    upside_capture = upside / BENCHMARK_RETURN_5Y
    downside_capture = downside / BENCHMARK_RETURN_5Y

    return {
        'Rolling Return (5Y) %': rolling_return * 100,
        'SD %': sd * 100,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Alpha %': alpha * 100,
        'Beta': beta,
        'Upside Capture %': upside_capture * 100,
        'Downside Capture %': downside_capture * 100
    }

# --- Processing --- #
results = []
for code in fund_codes:
    try:
        nav_df = fetch_nav(code)
        metrics = compute_metrics(nav_df)
        metrics['Scheme Code'] = code
        results.append(metrics)
    except Exception as e:
        st.error(f"❌ Error for code {code}: {e}")

if results:
    df = pd.DataFrame(results)
    df = df[['Scheme Code', 'Rolling Return (5Y) %', 'SD %', 'Sharpe', 'Sortino',
             'Alpha %', 'Beta', 'Upside Capture %', 'Downside Capture %']]

    # Scoring
    df_valid = df.dropna()
    df_norm = df_valid.copy()
    for col in ['Rolling Return (5Y) %', 'SD %', 'Sharpe', 'Sortino', 'Alpha %', 'Beta', 'Upside Capture %', 'Downside Capture %']:
        if col in ['SD %', 'Beta', 'Downside Capture %']:
            df_norm[col] = df_norm[col].max() - df_norm[col]
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min) if col_max != col_min else 1

    df_norm['Score'] = (
        df_norm['Sortino'] * weight_sortino +
        df_norm['Downside Capture %'] * weight_downside +
        df_norm['SD %'] * weight_sd +
        df_norm['Rolling Return (5Y) %'] * weight_rolling +
        df_norm['Alpha %'] * weight_alpha +
        df_norm['Sharpe'] * weight_sharpe +
        df_norm['Upside Capture %'] * weight_upside +
        df_norm['Beta'] * weight_beta
    )

    df_norm['Score'] = (df_norm['Score'] * 10).round(2)
    final_df = df.merge(df_norm[['Scheme Code', 'Score']], on='Scheme Code', how='left')
    final_df = final_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    st.markdown("### 🏆 Ranked Mutual Funds")
    st.dataframe(final_df, use_container_width=True)
else:
    st.info("Enter at least one valid mutual fund scheme code to begin.")
