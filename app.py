import streamlit as st
import pandas as pd
from utils import fetch_all_funds, fetch_fund_data, calculate_metrics, benchmark_metrics

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")
st.title("📊 Mutual Fund Ranking Tool")

# Load all available funds once
@st.cache_data
def load_all_funds():
    return fetch_all_funds()

try:
    all_funds = load_all_funds()
    if not all_funds:
        st.error("❌ Unable to fetch fund list. Please check your internet connection.")
        st.stop()
    
    fund_options = {f"{fund['schemeName']} ({fund['schemeCode']})": fund['schemeCode'] 
                   for fund in all_funds}
except Exception as e:
    st.error(f"❌ Error loading funds: {str(e)}")
    st.stop()

# Dropdown for selecting funds
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=list(fund_options.keys()),
    help="Start typing to search for funds"
)

# Weight sliders
st.subheader("⚖️ Metric Weights")
col1, col2 = st.columns(2)

with col1:
    cagr_weight = st.slider("CAGR Weight (%)", 0, 100, 30)
    stddev_weight = st.slider("Standard Deviation Weight (%)", 0, 100, 20)

with col2:
    sharpe_weight = st.slider("Sharpe Ratio Weight (%)", 0, 100, 20)
    sortino_weight = st.slider("Sortino Ratio Weight (%)", 0, 100, 30)

# Show total weight
total_weight = cagr_weight + stddev_weight + sharpe_weight + sortino_weight
if total_weight == 0:
    st.warning("⚠️ Please assign at least some weight to the metrics.")

if not selected_funds:
    st.info("☝️ Please select at least one mutual fund to begin.")
else:
    with st.spinner("📈 Calculating metrics and scores..."):
        fund_scores = []
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for i, fund_label in enumerate(selected_funds):
            scheme_code = fund_options[fund_label]
            
            # Fetch fund data
            fund_data = fetch_fund_data(scheme_code)
            
            if fund_data and len(fund_data) > 30:  # Need sufficient data
                metrics = calculate_metrics(fund_data)
                
                if metrics:
                    # Calculate weighted score (normalize by total weight)
                    weighted_score = 0
                    if total_weight > 0:
                        weighted_score = (
                            metrics.get("CAGR", 0) * cagr_weight +
                            -metrics.get("Standard Deviation", 0) * stddev_weight +  # Negative for volatility
                            metrics.get("Sharpe Ratio", 0) * sharpe_weight +
                            metrics.get("Sortino Ratio", 0) * sortino_weight
                        ) / total_weight
                    
                    fund_scores.append({
                        "Fund Name": fund_label.split(" (")[0],  # Remove scheme code from display
                        "Scheme Code": scheme_code,
                        "CAGR (%)": metrics.get("CAGR", 0),
                        "Std Dev (%)": metrics.get("Standard Deviation", 0),
                        "Sharpe Ratio": metrics.get("Sharpe Ratio", 0),
                        "Sortino Ratio": metrics.get("Sortino Ratio", 0),
                        "Weighted Score": round(weighted_score, 2)
                    })
            else:
                st.warning(f"⚠️ Insufficient data for {fund_label}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(selected_funds))
        
        progress_bar.empty()
        
        if fund_scores:
            # Create DataFrame and sort by score
            df = pd.DataFrame(fund_scores)
            df = df.sort_values("Weighted Score", ascending=False).reset_index(drop=True)
            df.index += 1  # Start ranking from 1
            
            st.success(f"✅ Ranking complete! Analyzed {len(fund_scores)} funds.")
            
            # Display results
            st.subheader("🏆 Fund Rankings")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=False
            )
            
            # Show top performer
            if len(df) > 0:
                top_fund = df.iloc[0]
                st.success(f"🥇 **Top Performer:** {top_fund['Fund Name']} (Score: {top_fund['Weighted Score']})")
        else:
            st.error("❌ No valid data found for the selected funds.")

# Benchmark comparison
st.subheader("📉 Benchmark Metrics (BSE 500)")
try:
    benchmark_data = benchmark_metrics()
    if benchmark_data:
        bench_df = pd.DataFrame([benchmark_data])
        st.dataframe(bench_df, use_container_width=True)
    else:
        st.warning("⚠️ Benchmark data not available. Please ensure 'bse500_returns.csv' exists.")
except Exception as e:
    st.warning(f"⚠️ Could not load benchmark data: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("*Data provided by [MFAPI](https://www.mfapi.in/) • Built with Streamlit*")