import streamlit as st
import pandas as pd
from utils import fetch_fund_metadata, calculate_metrics, benchmark_metrics

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# Load fund metadata once
@st.cache_data
def load_metadata():
    return fetch_fund_metadata()

fund_metadata = load_metadata()
fund_options = {f"{v['scheme_name']} ({v['scheme_code']})": v['scheme_code'] for k, v in fund_metadata.items()}

# Dropdown for selecting funds
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=list(fund_options.keys()),
)

# Weight sliders
cagr_weight = st.slider("CAGR Weight (%)", 0, 100, 30)
stddev_weight = st.slider("Standard Deviation Weight (%)", 0, 100, 20)
sharpe_weight = st.slider("Sharpe Ratio Weight (%)", 0, 100, 20)
sortino_weight = st.slider("Sortino Ratio Weight (%)", 0, 100, 30)

weights = {
    "CAGR": cagr_weight,
    "Standard Deviation": stddev_weight,
    "Sharpe Ratio": sharpe_weight,
    "Sortino Ratio": sortino_weight,
}

if not selected_funds:
    st.info("☝️ Please select at least one mutual fund to begin.")
else:
    with st.spinner("📈 Calculating scores..."):
        fund_scores = []
        for label in selected_funds:
            code = fund_options[label]
            metrics = calculate_metrics(code)
            if metrics:
                weighted_score = sum([
                    metrics.get("CAGR", 0) * cagr_weight,
                    -metrics.get("Standard Deviation", 0) * stddev_weight,
                    metrics.get("Sharpe Ratio", 0) * sharpe_weight,
                    metrics.get("Sortino Ratio", 0) * sortino_weight
                ])
                fund_scores.append({
                    "Fund Name": label,
                    **metrics,
                    "Score": weighted_score
                })

        df = pd.DataFrame(fund_scores)
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)

        st.success("✅ Ranking complete!")
        st.dataframe(df)

        # Optionally compare with benchmark
        st.subheader("📉 Benchmark Metrics (BSE 500)")
        st.write(pd.DataFrame([benchmark_metrics()]))
