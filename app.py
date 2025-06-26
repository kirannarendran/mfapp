# app.py
import streamlit as st
from utils import fetch_fund_metadata, calculate_metrics, benchmark_metrics

# Set page title
st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# Sample fund scheme codes (replace or extend as needed)
fund_codes = [100171, 100122, 118550, 119551, 120321]
fund_list = fetch_fund_metadata(fund_codes)

# Multiselect for funds
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=fund_list,
    format_func=lambda x: x["label"] if isinstance(x, dict) else x
)

selected_codes = [fund["value"] for fund in selected_funds if isinstance(fund, dict)]

# Sliders for metric weights
col1, col2, col3, col4 = st.columns(4)
with col1:
    cagr_weight = st.slider("CAGR Weight (%)", 0, 100, 30)
with col2:
    std_weight = st.slider("Standard Deviation Weight (%)", 0, 100, 20)
with col3:
    sharpe_weight = st.slider("Sharpe Ratio Weight (%)", 0, 100, 20)
with col4:
    sortino_weight = st.slider("Sortino Ratio Weight (%)", 0, 100, 30)

# Validate input
if not selected_codes:
    st.info("👈 Please select at least one mutual fund to begin.")
    st.stop()

# Calculate scores
weights = {
    "cagr": cagr_weight,
    "std": std_weight,
    "sharpe": sharpe_weight,
    "sortino": sortino_weight
}

ranking_df = calculate_metrics(selected_codes, weights)
benchmark_df = benchmark_metrics()

# Show results
st.subheader("📈 Ranked Mutual Funds")
st.dataframe(ranking_df, use_container_width=True)

st.subheader("📉 Benchmark Performance (BSE 500)")
st.line_chart(benchmark_df.set_index("date")[["Close"]])
