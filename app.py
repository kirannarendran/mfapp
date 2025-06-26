import streamlit as st
import pandas as pd
import requests
from utils import calculate_metrics, benchmark_metrics

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

@st.cache_data
def load_fund_data(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    return response.json()

@st.cache_data
def get_all_funds():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    return response.json()

# Load fund list and build mapping
fund_list = get_all_funds()
fund_mapping = {fund["scheme_name"]: fund["scheme_code"] for fund in fund_list if fund.get("scheme_name") and fund.get("scheme_code")}

# Let user search and select
selected_funds = st.multiselect("🔍 Search and select mutual funds", options=list(fund_mapping.keys()), default=[])

# Sliders for weights
c1, c2, c3, c4 = st.columns(4)
with c1:
    weight_cagr = st.slider("CAGR Weight (%)", 0, 100, 30)
with c2:
    weight_std = st.slider("Standard Deviation Weight (%)", 0, 100, 20)
with c3:
    weight_sharpe = st.slider("Sharpe Ratio Weight (%)", 0, 100, 20)
with c4:
    weight_sortino = st.slider("Sortino Ratio Weight (%)", 0, 100, 30)

weight_total = weight_cagr + weight_std + weight_sharpe + weight_sortino
if weight_total != 100:
    st.warning("⚠️ Total weight must equal 100%. Current total: {}%".format(weight_total))
    st.stop()

# Process metrics
all_metrics = []
for fund_name in selected_funds:
    scheme_code = fund_mapping[fund_name]
    fund_data = load_fund_data(scheme_code)
    metrics = calculate_metrics(fund_name, fund_data)
    all_metrics.append(metrics)

# Create DataFrame and rank
df = pd.DataFrame(all_metrics)
if not df.empty:
    df["Score"] = (
        df["CAGR"] * (weight_cagr / 100)
        - df["Standard Deviation"] * (weight_std / 100)
        + df["Sharpe Ratio"] * (weight_sharpe / 100)
        + df["Sortino Ratio"] * (weight_sortino / 100)
    )
    df = df.sort_values("Score", ascending=False)
    df.reset_index(drop=True, inplace=True)

    # Format numbers
    df = df.round(2)

    # Benchmark row
    bench_row = pd.DataFrame([benchmark_metrics()])
    bench_row["Fund"] = "**BSE 500 Benchmark**"
    bench_row["Score"] = ""  # no score

    # Append benchmark
    display_df = pd.concat([df, bench_row], ignore_index=True)

    # Remove unwanted columns if any (like NAV)
    unwanted_cols = [col for col in display_df.columns if "NAV" in col]
    display_df.drop(columns=unwanted_cols, inplace=True, errors="ignore")

    # Show
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("👆 Please select at least one mutual fund to begin.")
