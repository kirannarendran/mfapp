import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress

# ---------------------- Config ----------------------
CATEGORY_RETURN_5Y = 23.59  # BSE500 TRI 5Y return
RISK_FREE_RATE = 0.06  # Assume 6% annual risk-free return

# ---------------------- Helper Functions ----------------------
def fetch_nav_history(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    if "data" not in data:
        return None
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
    df = df.dropna()
    df = df.sort_values("date")
    df = df.set_index("date")
    return df

def compute_metrics(nav_df):
    try:
        if len(nav_df) < 5 * 252:
            return None

        nav_df = nav_df.resample("D").ffill()
        nav_df = nav_df.dropna()
        nav_df["returns"] = nav_df["nav"].pct_change()
        nav_df = nav_df.dropna()

        rolling_return = (nav_df["nav"][-1] / nav_df["nav"].iloc[-5*252])**(1/5) - 1
        std_dev = nav_df["returns"].std() * np.sqrt(252)
        downside_returns = nav_df["returns"][nav_df["returns"] < RISK_FREE_RATE / 252]
        downside_dev = downside_returns.std() * np.sqrt(252)

        sharpe = (rolling_return - RISK_FREE_RATE) / std_dev if std_dev else None
        sortino = (rolling_return - RISK_FREE_RATE) / downside_dev if downside_dev else None

        market_returns = np.full(len(nav_df), CATEGORY_RETURN_5Y / 252)
        beta, alpha, _, _, _ = linregress(market_returns, nav_df["returns"])
        alpha = alpha * 252

        upside = nav_df["returns"][market_returns > 0].mean()
        downside = nav_df["returns"][market_returns < 0].mean()
        market_upside = market_returns[market_returns > 0].mean()
        market_downside = market_returns[market_returns < 0].mean()

        upside_capture = upside / market_upside if market_upside else None
        downside_capture = downside / market_downside if market_downside else None

        return {
            "Rolling Return (5Y)": rolling_return * 100,
            "SD %": std_dev * 100,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Alpha %": alpha * 100,
            "Beta": beta,
            "Upside Capture %": upside_capture * 100,
            "Downside Capture %": downside_capture * 100
        }
    except:
        return None

def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) * 10

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Mutual Fund Ranker", layout="centered")
st.title("\U0001F4C8 Mutual Fund Ranker - Flexi Cap Category")

scheme_codes_input = st.text_area("Enter mutual fund scheme codes (from MFAPI.in), one per line:")
scheme_codes = [code.strip() for code in scheme_codes_input.strip().split("\n") if code.strip()]

with st.expander("\U0001F3AF Adjust Metric Weights"):
    sortino_wt = st.slider("Sortino Ratio Weight", 0.0, 1.0, 0.20)
    downside_wt = st.slider("Downside Capture Weight", 0.0, 1.0, 0.20)
    sd_wt = st.slider("Standard Deviation (5Y) Weight", 0.0, 1.0, 0.20)
    return_wt = st.slider("Rolling Return (5Y) Weight", 0.0, 1.0, 0.15)
    alpha_wt = st.slider("Alpha Weight", 0.0, 1.0, 0.10)
    sharpe_wt = st.slider("Sharpe Ratio Weight", 0.0, 1.0, 0.05)
    upside_wt = st.slider("Upside Capture Weight", 0.0, 1.0, 0.05)
    beta_wt = st.slider("Beta Weight", 0.0, 1.0, 0.05)

total_weight = sum([sortino_wt, downside_wt, sd_wt, return_wt, alpha_wt, sharpe_wt, upside_wt, beta_wt])

if scheme_codes:
    rows = []
    for code in scheme_codes:
        nav_df = fetch_nav_history(code)
        if nav_df is not None:
            metrics = compute_metrics(nav_df)
            if metrics:
                metrics["Scheme Code"] = code
                rows.append(metrics)

    if rows:
        df = pd.DataFrame(rows)
        df = df.dropna()

        if df.empty:
            st.warning("All funds had missing or insufficient data.")
        else:
            df["Total Score"] = (
                normalize(df["Sortino"]) * sortino_wt +
                normalize(df["Downside Capture %"]) * downside_wt +
                normalize(df["SD %"]) * sd_wt +
                normalize(df["Rolling Return (5Y)"]) * return_wt +
                normalize(df["Alpha %"]) * alpha_wt +
                normalize(df["Sharpe"]) * sharpe_wt +
                normalize(df["Upside Capture %"]) * upside_wt +
                normalize(df["Beta"]) * beta_wt
            ) / total_weight

            df = df.sort_values("Total Score", ascending=False)
            st.subheader("\U0001F3C6 Ranked Mutual Funds")
            st.dataframe(df.set_index("Scheme Code").round(3))
    else:
        st.warning("Could not fetch data for the entered scheme codes.")
