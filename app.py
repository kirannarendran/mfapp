import streamlit as st
import requests
import pandas as pd
import numpy as np

@st.cache_data
def load_benchmark():
    return pd.read_csv("bse500_returns.csv")

@st.cache_data
def fetch_all_funds():
    try:
        response = requests.get("https://api.mfapi.in/mf")
        if response.status_code == 200:
            funds = response.json()
            return {f['schemeCode']: f['schemeName'] for f in funds if 'schemeCode' in f and 'schemeName' in f}
        else:
            return {}
    except:
        return {}

def fetch_fund_data(scheme_code):
    try:
        response = requests.get(f"https://api.mfapi.in/mf/{scheme_code}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def calculate_metrics(nav_data):
    df = pd.DataFrame(nav_data)
    df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").set_index("date")

    df['returns'] = df['nav'].pct_change()
    cagr = (df['nav'].iloc[-1] / df['nav'].iloc[0])**(1/((df.index[-1] - df.index[0]).days/365)) - 1
    std = df['returns'].std() * np.sqrt(252)
    sharpe = (df['returns'].mean() * 252) / std if std != 0 else 0
    sortino = (df['returns'].mean() * 252) / (df['returns'][df['returns'] < 0].std() * np.sqrt(252)) if df['returns'][df['returns'] < 0].std() != 0 else 0

    return round(cagr * 100, 2), round(std * 100, 2), round(sharpe, 2), round(sortino, 2)

def calculate_score(cagr, std, sharpe, sortino, weights):
    return (weights["CAGR"] * cagr - weights["Standard Deviation"] * std + weights["Sharpe Ratio"] * sharpe + weights["Sortino Ratio"] * sortino) / 100

st.title("📈 Mutual Fund Ranking Tool")

all_funds = fetch_all_funds()
fund_options = list(all_funds.items())

selected_funds = st.multiselect("Select Mutual Funds", options=fund_options, format_func=lambda x: x[1])

if not selected_funds:
    st.warning("⚠️ Please select at least one mutual fund to see rankings.")
    st.stop()

st.sidebar.header("Adjust Metric Weights (0–100)")
cagr_w = st.sidebar.slider("CAGR Weight", 0, 100, 25)
std_w = st.sidebar.slider("Standard Deviation Weight (negative impact)", 0, 100, 25)
sharpe_w = st.sidebar.slider("Sharpe Ratio Weight", 0, 100, 25)
sortino_w = st.sidebar.slider("Sortino Ratio Weight", 0, 100, 25)

weights = {
    "CAGR": cagr_w,
    "Standard Deviation": std_w,
    "Sharpe Ratio": sharpe_w,
    "Sortino Ratio": sortino_w,
}

results = []

for code, name in selected_funds:
    data = fetch_fund_data(code)
    if data and "data" in data:
        try:
            cagr, std, sharpe, sortino = calculate_metrics(data['data'])
            score = calculate_score(cagr, std, sharpe, sortino, weights)
            results.append({
                "Fund Name": name,
                "CAGR (%)": cagr,
                "Std Dev (%)": std,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Score": round(score, 2)
            })
        except:
            continue

if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)
    st.subheader("📊 Fund Rankings")
    st.dataframe(df, use_container_width=True)
else:
    st.error("⚠️ Failed to fetch metrics for selected funds.")

benchmark = load_benchmark()
st.subheader("📎 Benchmark: BSE 500")
st.dataframe(benchmark, use_container_width=True)

st.sidebar.header("🧪 API Debug")
debug_code = st.sidebar.text_input("Enter Fund Code to Inspect")
if debug_code:
    debug_data = fetch_fund_data(debug_code)
    st.sidebar.write(debug_data if debug_data else "No data found or invalid code.")
