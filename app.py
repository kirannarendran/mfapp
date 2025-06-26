import streamlit as st
import pandas as pd
import numpy as np
import os

# Page config
st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")

# App title
st.markdown("## 📊 Mutual Fund Ranking Tool")

# Load mutual fund data
@st.cache_data
def load_fund_data():
    df = pd.read_csv("flexicap_metrics_to_excel.csv")
    df.dropna(subset=["scheme_name"], inplace=True)
    return df

# Load benchmark data
@st.cache_data
def load_benchmark_data():
    file_path = os.path.join(os.path.dirname(__file__), "bse500_returns.csv")
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
        df.sort_values("Date", inplace=True)
        df["Returns"] = df["Close"].pct_change()
        return df
    except FileNotFoundError:
        st.error("❌ Benchmark CSV not found. Please ensure 'bse500_returns.csv' is in the same folder as app.py.")
        st.stop()

fund_data = load_fund_data()
benchmark_df = load_benchmark_data()

# Create mapping for scheme names to codes
fund_mapping = {row["scheme_name"]: row["scheme_code"] for _, row in fund_data.iterrows()}

# Sidebar - Metric Weights
st.sidebar.markdown("### Adjust Metric Weights (%)")
std_weight = st.sidebar.slider("Standard Deviation Weight (%)", 0, 100, 20)
downside_weight = st.sidebar.slider("Downside Capture Weight (%)", 0, 100, 20)
cagr_weight = st.sidebar.slider("CAGR Weight (%)", 0, 100, 15)
sharpe_weight = st.sidebar.slider("Sharpe Ratio Weight (%)", 0, 100, 10)
upside_weight = st.sidebar.slider("Upside Capture Weight (%)", 0, 100, 10)
alpha_weight = st.sidebar.slider("Alpha Weight (%)", 0, 100, 5)

# Main panel - Fund selection
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=list(fund_mapping.keys()),
    default=[],
    placeholder="Type to search..."
)

if selected_funds:
    selected_df = fund_data[fund_data["scheme_name"].isin(selected_funds)].copy()

    # Merge benchmark return into calculations (optional)
    benchmark_std = benchmark_df["Returns"].std()
    benchmark_downside = benchmark_df["Returns"][benchmark_df["Returns"] < 0].std()
    benchmark_upside = benchmark_df["Returns"][benchmark_df["Returns"] > 0].std()

    # Calculate score
    def calculate_score(row):
        score = 0
        if row["standard_deviation"] and benchmark_std:
            score += (1 - row["standard_deviation"] / benchmark_std) * std_weight
        if row["downside_capture"] and benchmark_downside:
            score += (1 - row["downside_capture"] / benchmark_downside) * downside_weight
        if row["CAGR"]:
            score += row["CAGR"] * cagr_weight
        if row["sharpe_ratio"]:
            score += row["sharpe_ratio"] * sharpe_weight
        if row["upside_capture"] and benchmark_upside:
            score += (row["upside_capture"] / benchmark_upside) * upside_weight
        if row["alpha"]:
            score += row["alpha"] * alpha_weight
        return score

    selected_df["score"] = selected_df.apply(calculate_score, axis=1)
    selected_df.sort_values(by="score", ascending=False, inplace=True)
    st.dataframe(selected_df[["scheme_name", "score"] + [col for col in selected_df.columns if col not in ["scheme_name", "score", "scheme_code"]]], use_container_width=True)

else:
    st.warning("Please select at least one mutual fund from the dropdown above to see rankings.")
