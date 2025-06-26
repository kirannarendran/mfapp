import streamlit as st
import pandas as pd
import numpy as np
from utils import fetch_nav, compute_metrics, compute_score
from datetime import datetime

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")

# Title and intro
st.markdown("## 📊 Mutual Fund Ranking Tool")
st.markdown("🔍 **Search and select mutual funds**")

# Load scheme mapping
@st.cache_data
def load_scheme_mapping():
    df = pd.read_csv("mapping.csv")
    return {row["scheme_name"]: row["scheme_code"] for _, row in df.iterrows()}

fund_mapping = load_scheme_mapping()

# Fund selection
selected_funds = st.multiselect("Start typing fund name…", options=list(fund_mapping.keys()))

# Weight sliders
st.markdown("### 🎚️ Adjust Metric Weights (Total should sum to 100)")
weights = {
    "Standard Deviation": st.slider("Standard Deviation Weight (%)", 0, 100, 20),
    "Downside Capture": st.slider("Downside Capture Weight (%)", 0, 100, 20),
    "Rolling Return (CAGR)": st.slider("CAGR Weight (%)", 0, 100, 15),
    "Sharpe Ratio": st.slider("Sharpe Ratio Weight (%)", 0, 100, 10),
    "Upside Capture": st.slider("Upside Capture Weight (%)", 0, 100, 10),
    "Alpha": st.slider("Alpha Weight (%)", 0, 100, 5),
}
total_weight = sum(weights.values())

if total_weight != 100:
    st.error(f"❌ Weights must sum to 100. Current total: {total_weight}")
    st.stop()

# Collect and compute metrics
rows = []
for fund_name in selected_funds:
    code = fund_mapping[fund_name]
    nav = fetch_nav(code)
    if nav is not None:
        metrics = compute_metrics(nav)
        if metrics:
            row = {"Fund": fund_name, **metrics}
            row["Score (Out of 10)"] = compute_score(metrics, weights, total_weight)
            rows.append(row)

# Load and compute benchmark
def load_benchmark():
    df = pd.read_csv("bse500_returns.csv", parse_dates=["Date"])
    df = df.dropna()
    df = df.rename(columns=lambda x: x.strip().lower())
    df = df.rename(columns={"close": "nav"})
    df = df[["date", "nav"]]
    df = df.sort_values("date").reset_index(drop=True)
    return df

benchmark = load_benchmark()
benchmark_metrics = compute_metrics(benchmark)
if benchmark_metrics:
    benchmark_row = {
        "Fund": "**BSE 500 Index (Benchmark)**",
        **benchmark_metrics,
        "Alpha": "",
        "Beta": "",
        "Score (Out of 10)": ""
    }
    rows.append(benchmark_row)

# Create and display DataFrame
if rows:
    df = pd.DataFrame(rows)
    df.insert(0, "SL No", range(1, len(df)+1))

    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Sort by score if available
    if "Score (Out of 10)" in df.columns:
        df["Score (Out of 10)"] = pd.to_numeric(df["Score (Out of 10)"], errors='coerce')
        df = df.sort_values("Score (Out of 10)", ascending=False).reset_index(drop=True)
        df["SL No"] = range(1, len(df)+1)

    # Highlight benchmark row
    def highlight_benchmark(s):
        return ['font-weight: bold' if '**BSE 500' in str(val) else '' for val in s]

    st.dataframe(df.style.apply(highlight_benchmark, subset=["Fund"]), use_container_width=True)
else:
    st.info("👈 Please select at least one fund to see results.")
