import streamlit as st
import pandas as pd
import requests
from utils import calculate_metrics, benchmark_metrics

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# --- Load Fund Data from MFAPI ---
@st.cache_data(show_spinner=False)
def load_fund_data():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

fund_list = [
    {"scheme_name": "Axis Bluechip Fund", "scheme_code": "123456"},
    {"scheme_name": "HDFC Flexi Cap Fund", "scheme_code": "234567"},
    {"scheme_name": "Mirae Asset Large Cap Fund", "scheme_code": "345678"},
]

fund_mapping = {fund["scheme_name"]: fund["scheme_code"] for fund in fund_list}
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds", list(fund_mapping.keys())
)

fund_mapping = {fund["scheme_name"]: fund["scheme_code"] for fund in fund_list}

# --- Fund Selection (searchable, nothing pre-selected) ---
selected_names = st.multiselect("🔍 Search and select mutual funds", options=list(fund_mapping.keys()))

# --- Weight Sliders ---
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
cagr_weight = c1.slider("CAGR Weight (%)", 0, 100, 30)
sd_weight = c2.slider("Standard Deviation Weight (%)", 0, 100, 20)
sharpe_weight = c3.slider("Sharpe Ratio Weight (%)", 0, 100, 20)
sortino_weight = c4.slider("Sortino Ratio Weight (%)", 0, 100, 30)

# --- Validate Weights ---
if cagr_weight + sd_weight + sharpe_weight + sortino_weight != 100:
    st.warning("⚠️ Total weight must be 100%.")
    st.stop()

# --- Show Prompt if No Selection ---
if not selected_names:
    st.info("👉 Please select at least one mutual fund to begin.")
    st.stop()

# --- Load and Process Metrics ---
st.markdown("---")
st.subheader("📈 Ranking Results")

selected_codes = [fund_mapping[name] for name in selected_names]

with st.spinner("Fetching fund data and calculating scores..."):
    result_df = calculate_metrics(
        selected_codes,
        cagr_weight,
        sd_weight,
        sharpe_weight,
        sortino_weight
    )
    bench_row = pd.DataFrame([benchmark_metrics()])
    result_df = pd.concat([result_df, bench_row], ignore_index=True)
    result_df = result_df.sort_values("Score", ascending=False)

st.dataframe(result_df, use_container_width=True)
