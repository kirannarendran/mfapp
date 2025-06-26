import streamlit as st
import pandas as pd
import requests
from utils import compute_metrics, compute_score

# Load benchmark data from local CSV
benchmark_df = pd.read_csv("bse500_returns.csv", parse_dates=["Date"])
benchmark_df["Returns"] = benchmark_df["Close"].pct_change()
benchmark_df.dropna(inplace=True)

# Fetch list of mutual funds (all funds)
@st.cache_data
def fetch_mutual_funds():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    return response.json()

fund_list = fetch_mutual_funds()
fund_mapping = {fund["scheme_name"]: fund["scheme_code"] for fund in fund_list if "scheme_name" in fund and "scheme_code" in fund}

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 Mutual Fund Ranking Tool")
st.write("### Search and select mutual funds")

# Multiselect without pre-selection
selected_funds = st.multiselect("Search and select mutual funds", options=list(fund_mapping.keys()))

# Metric weights
st.sidebar.title("Adjust Metric Weights (%)")
weights = {
    "Standard Deviation": st.sidebar.slider("Standard Deviation Weight (%)", 0, 100, 20),
    "Downside Capture": st.sidebar.slider("Downside Capture Weight (%)", 0, 100, 20),
    "CAGR": st.sidebar.slider("CAGR Weight (%)", 0, 100, 15),
    "Sharpe Ratio": st.sidebar.slider("Sharpe Ratio Weight (%)", 0, 100, 10),
    "Upside Capture": st.sidebar.slider("Upside Capture Weight (%)", 0, 100, 10),
    "Alpha": st.sidebar.slider("Alpha Weight (%)", 0, 100, 5),
}

if selected_funds:
    all_data = []
    for fund_name in selected_funds:
        scheme_code = fund_mapping[fund_name]
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        response = requests.get(url)
        data = response.json()

        if "data" in data:
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            df = df.sort_values("date").dropna()
            df["returns"] = df["nav"].pct_change()
            df.dropna(inplace=True)

            metrics = compute_metrics(df["returns"], benchmark_df["Returns"])
            metrics["Fund"] = fund_name
            all_data.append(metrics)

    if all_data:
        results = pd.DataFrame(all_data)
        results["Score"] = results.apply(lambda row: compute_score(row, weights), axis=1)
        results = results.sort_values("Score", ascending=False).reset_index(drop=True)
        st.write("### 📈 Ranked Results")
        st.dataframe(results, use_container_width=True)
    else:
        st.warning("No data returned for the selected funds.")
else:
    st.info("Please select at least one mutual fund to begin.")
