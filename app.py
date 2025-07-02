import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression

@st.cache_data(ttl=24*3600)
def load_benchmark():
    df = pd.read_csv("bse500_returns.csv", parse_dates=["Date"])
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df = df.set_index("date").sort_index()
    df["returns"] = df["close"].pct_change()
    return df["returns"]

@st.cache_data(ttl=24*3600)
def load_fund_list():
    df = pd.read_csv("fund_list.csv", dtype={"schemeCode": str})
    return df

def fetch_fund_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("data", [])

def calculate_metrics(nav_data, benchmark_returns):
    df = pd.DataFrame(nav_data)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna()
    if df.empty:
        st.write("DEBUG: NAV DataFrame empty after cleaning")
        return (None,) * 8

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").set_index("date")
    df["returns"] = df["nav"].pct_change()

    st.write(f"DEBUG: Fund date range: {df.index.min()} to {df.index.max()}")
    st.write(f"DEBUG: Benchmark date range: {benchmark_returns.index.min()} to {benchmark_returns.index.max()}")

    # Slice benchmark returns to fund date range for overlap
    benchmark_returns_slice = benchmark_returns[(benchmark_returns.index >= df.index.min()) & (benchmark_returns.index <= df.index.max())]

    combined = pd.merge(df[["returns"]], benchmark_returns_slice.to_frame(), left_index=True, right_index=True, how="inner")
    combined.columns = ["fund_returns", "bench_returns"]

    st.write(f"DEBUG: Combined data length after merging fund & benchmark returns: {len(combined)}")

    if len(combined) < 30:
        st.write("DEBUG: Not enough overlapping data points for metrics calculation")
        return (None,) * 8

    fund_ret = combined["fund_returns"].values.reshape(-1, 1)
    bench_ret = combined["bench_returns"].values.reshape(-1, 1)

    days = (df.index[-1] - df.index[0]).days
    cagr = (df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (365.25 / days) - 1
    std = df["returns"].std() * np.sqrt(252)
    sharpe = (df["returns"].mean() * 252) / std if std != 0 else None
    neg_std = df["returns"][df["returns"] < 0].std() * np.sqrt(252)
    sortino = (df["returns"].mean() * 252) / neg_std if neg_std != 0 else None

    model = LinearRegression().fit(bench_ret, fund_ret)
    beta = model.coef_[0][0]
    alpha = model.intercept_[0]
    alpha = (1 + alpha) ** 252 - 1

    positive_mask = combined["bench_returns"] > 0
    negative_mask = combined["bench_returns"] < 0

    upside_capture = None
    downside_capture = None

    if positive_mask.sum() > 0:
        upside_capture = combined.loc[positive_mask, "fund_returns"].mean() / combined.loc[positive_mask, "bench_returns"].mean()
    if negative_mask.sum() > 0:
        downside_capture = combined.loc[negative_mask, "fund_returns"].mean() / combined.loc[negative_mask, "bench_returns"].mean()

    def to_pct(val):
        return round(val * 100, 2) if val is not None else None

    return (
        to_pct(cagr),
        to_pct(std),
        round(sharpe, 2) if sharpe is not None else None,
        round(sortino, 2) if sortino is not None else None,
        to_pct(alpha),
        round(beta, 3) if beta is not None else None,
        to_pct(upside_capture),
        to_pct(downside_capture),
    )

def calculate_score(cagr, std, sharpe, sortino, weights):
    cagr = cagr or 0
    std = std or 0
    sharpe = sharpe or 0
    sortino = sortino or 0
    return (weights["CAGR"] * cagr - weights["Standard Deviation"] * std + weights["Sharpe Ratio"] * sharpe + weights["Sortino Ratio"] * sortino) / 100

def safe_format(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, str):
        return x
    return f"{x:.2f}"

def main():
    st.title("📈 Mutual Fund Ranking Tool")

    benchmark_returns = load_benchmark()
    funds_df = load_fund_list()
    fund_options = list(zip(funds_df["schemeCode"], funds_df["schemeName"]))

    selected_funds = st.multiselect(
        "Select Mutual Funds",
        options=fund_options,
        format_func=lambda x: f"{x[1]} ({x[0]})",
    )

    if not selected_funds:
        st.warning("⚠️ Please select at least one mutual fund.")
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
        nav_data = fetch_fund_nav(code)
        if nav_data:
            cagr, std, sharpe, sortino, alpha, beta, upside, downside = calculate_metrics(nav_data, benchmark_returns)
            score = calculate_score(cagr, std, sharpe, sortino, weights)
            results.append({
                "Fund Name": name,
                "CAGR (%)": cagr,
                "Std Dev (%)": std,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Alpha (%)": alpha,
                "Beta": beta,
                "Upside Capture (%)": upside,
                "Downside Capture (%)": downside,
                "Score (Out of 10)": round(score, 2)
            })

    # Benchmark metrics
    benchmark_cagr = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
    benchmark_std = benchmark_returns.std() * np.sqrt(252)
    benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252) if benchmark_returns.std() != 0 else None
    neg_std = benchmark_returns[benchmark_returns < 0].std() * np.sqrt(252)
    benchmark_sortino = benchmark_returns.mean() / neg_std * np.sqrt(252) if neg_std != 0 else None

    benchmark_row = {
        "Fund Name": "📌 BSE 500 Benchmark",
        "CAGR (%)": round(benchmark_cagr * 100, 2),
        "Std Dev (%)": round(benchmark_std * 100, 2),
        "Sharpe": round(benchmark_sharpe, 2) if benchmark_sharpe else "",
        "Sortino": round(benchmark_sortino, 2) if benchmark_sortino else "",
        "Alpha (%)": "",
        "Beta": "",
        "Upside Capture (%)": "",
        "Downside Capture (%)": "",
        "Score (Out of 10)": ""
    }

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Score (Out of 10)", ascending=False)
        df = pd.concat([pd.DataFrame([benchmark_row]), df], ignore_index=True)
    else:
        df = pd.DataFrame([benchmark_row])

    df.index += 1
    df.index.name = "SL"

    st.subheader("📊 Fund Rankings")
    st.dataframe(
        df.style.hide(axis="index").format({
            "CAGR (%)": safe_format,
            "Std Dev (%)": safe_format,
            "Sharpe": safe_format,
            "Sortino": safe_format,
            "Alpha (%)": safe_format,
            "Beta": safe_format,
            "Upside Capture (%)": safe_format,
            "Downside Capture (%)": safe_format,
            "Score (Out of 10)": safe_format,
        }),
        use_container_width=True,
    )

    st.sidebar.header("API Debug")
    debug_code = st.sidebar.text_input("Enter Fund Code to Inspect")
    if debug_code:
        debug_data = fetch_fund_nav(debug_code)
        st.sidebar.write(debug_data if debug_data else "No data found or invalid code.")

if __name__ == "__main__":
    main()
