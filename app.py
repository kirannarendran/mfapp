import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from scipy.stats import linregress

st.set_page_config(page_title="Mutual Fund Ranking Tool", layout="wide")

# ----------------------------------------------
# Utility Functions
# ----------------------------------------------

def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()
    nav_data = data.get("data", [])
    df = pd.DataFrame(nav_data)
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    return df.dropna()

def compute_metrics(df, benchmark_df=None):
    df = df.sort_values("date").dropna()
    df["return"] = df["nav"].pct_change()
    cagr = (df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1 / ((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365)) - 1
    std_dev = df["return"].std() * np.sqrt(252)
    sharpe_ratio = df["return"].mean() / df["return"].std() * np.sqrt(252)
    sortino_ratio = df["return"].mean() / df[df["return"] < 0]["return"].std() * np.sqrt(252)
    alpha = np.nan
    beta = np.nan
    upside = np.nan
    downside = np.nan

    if benchmark_df is not None:
        benchmark_df = benchmark_df.sort_values("date")
        combined = pd.merge(df, benchmark_df, on="date", suffixes=("", "_bench"))
        combined["excess_return"] = combined["return"] - combined["return_bench"]
        slope, intercept, r_value, p_value, std_err = linregress(combined["return_bench"].dropna(), combined["return"].dropna())
        beta = slope
        alpha = (combined["return"].mean() - beta * combined["return_bench"].mean()) * 252

        upside = 100 * combined[combined["return_bench"] > 0]["return"].mean() / combined[combined["return_bench"] > 0]["return_bench"].mean()
        downside = 100 * combined[combined["return_bench"] < 0]["return"].mean() / combined[combined["return_bench"] < 0]["return_bench"].mean()

    return {
        "CAGR": round(cagr * 100, 2),
        "Standard Deviation": round(std_dev * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Sortino Ratio": round(sortino_ratio, 2),
        "Alpha": round(alpha, 2) if not np.isnan(alpha) else "",
        "Beta": round(beta, 2) if not np.isnan(beta) else "",
        "Upside Capture": round(upside, 2) if not np.isnan(upside) else "",
        "Downside Capture": round(downside, 2) if not np.isnan(downside) else "",
    }

def compute_score(row, weights, benchmark=None):
    if row["Fund"] == "**BSE 500 Index (Benchmark)**":
        return ""
    score = 0
    for metric, weight in weights.items():
        benchmark_val = benchmark[metric] if benchmark else 1
        if benchmark_val == 0:
            continue
        score += (row[metric] / benchmark_val) * weight
    return round(score, 2)

# ----------------------------------------------
# Load Data
# ----------------------------------------------

@st.cache_data
def load_fund_list():
    url = "https://api.mfapi.in/mf"
    response = requests.get(url)
    return response.json()

@st.cache_data
def load_benchmark():
    df = pd.read_csv("bse500_returns.csv")
    df["date"] = pd.to_datetime(df["Date"])
    df["nav"] = df["Close"]
    df = df[["date", "nav"]]
    df = df.sort_values("date")
    df["return"] = df["nav"].pct_change()
    return df.dropna()

fund_list = load_fund_list()
fund_mapping = {f["scheme_name"]: f["scheme_code"] for f in fund_list}
valid_funds = [name for name in fund_mapping if not any(x in name.lower() for x in ["fmp", "gold", "etf", "liquid", "overnight"])]
default_funds = [
    "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
    "Kotak Flexi Cap Fund - Direct Plan - Growth",
    "Quant Flexi Cap Fund - Direct Plan - Growth"
]

# ----------------------------------------------
# Streamlit UI
# ----------------------------------------------

st.title("📊 Mutual Fund Ranking Tool")

# Weight sliders
st.sidebar.subheader("Adjust Metric Weights (%)")
weights = {
    "Standard Deviation": st.sidebar.slider("Standard Deviation Weight (%)", 0, 100, 20),
    "Downside Capture": st.sidebar.slider("Downside Capture Weight (%)", 0, 100, 20),
    "CAGR": st.sidebar.slider("CAGR Weight (%)", 0, 100, 15),
    "Sharpe Ratio": st.sidebar.slider("Sharpe Ratio Weight (%)", 0, 100, 10),
    "Upside Capture": st.sidebar.slider("Upside Capture Weight (%)", 0, 100, 10),
    "Alpha": st.sidebar.slider("Alpha Weight (%)", 0, 100, 5),
}

# Fund selector
selected_funds = st.multiselect(
    "🔍 Search and select mutual funds",
    options=valid_funds,
    default=[f for f in default_funds if f in valid_funds]
)

benchmark_df = load_benchmark()
benchmark_metrics = compute_metrics(benchmark_df)

# ----------------------------------------------
# Compute Metrics
# ----------------------------------------------

rows = []
for i, fund_name in enumerate(selected_funds, start=1):
    scheme_code = fund_mapping[fund_name]
    nav_df = fetch_nav(scheme_code)
    metrics = compute_metrics(nav_df, benchmark_df)
    score = compute_score({**metrics, "Fund": fund_name}, weights, benchmark_metrics)
    rows.append({
        "SL No": i,
        "Fund": fund_name,
        **metrics,
        "Score (Out of 10)": score
    })

# Append Benchmark row
rows.append({
    "SL No": len(rows) + 1,
    "Fund": "**BSE 500 Index (Benchmark)**",
    **benchmark_metrics,
    "Score (Out of 10)": ""
})

df = pd.DataFrame(rows)
score_col = "Score (Out of 10)"
if score_col in df.columns:
    df = df.sort_values(by=score_col, ascending=False, na_position='last').reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df)+1))
    df.loc[df["Fund"] == "**BSE 500 Index (Benchmark)**", "Rank"] = ""

# ----------------------------------------------
# Display
# ----------------------------------------------
st.dataframe(
    df.style.applymap(lambda v: 'font-weight: bold' if isinstance(v, str) and "Benchmark" in v else ''),
    use_container_width=True
)
