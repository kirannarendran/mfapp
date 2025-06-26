import pandas as pd
import numpy as np
from scipy.stats import linregress

def fetch_nav(scheme_code):
    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        data = pd.read_json(url)
        navs = pd.DataFrame(data["data"].tolist())
        navs["date"] = pd.to_datetime(navs["date"], format="%d-%m-%Y")
        navs["nav"] = pd.to_numeric(navs["nav"], errors="coerce")
        return navs.dropna().sort_values("date")
    except Exception:
        return None

def compute_metrics(df):
    df = df.copy()
    df = df.sort_values("date").dropna()

    df["return"] = df["nav"].pct_change()
    df.dropna(inplace=True)

    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.0
    cagr = ((df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    std_dev = df["return"].std() * np.sqrt(252) * 100

    downside = df[df["return"] < 0]["return"]
    sortino = (df["return"].mean() / downside.std()) * np.sqrt(252) if not downside.empty else np.nan

    sharpe = (df["return"].mean() / df["return"].std()) * np.sqrt(252) if df["return"].std() > 0 else np.nan

    if "benchmark" not in df.columns:
        return {
            "Rolling Return (CAGR)": round(cagr, 2),
            "Standard Deviation": round(std_dev, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Alpha": None,
            "Beta": None,
            "Upside Capture": None,
            "Downside Capture": None
        }

    benchmark_returns = df["benchmark"].pct_change()
    merged = pd.DataFrame({
        "fund": df["return"],
        "benchmark": benchmark_returns
    }).dropna()

    slope, intercept, r_value, p_value, std_err = linregress(merged["benchmark"], merged["fund"])
    alpha = (intercept * 252) * 100
    beta = slope

    up = merged[merged["benchmark"] > 0]
    down = merged[merged["benchmark"] < 0]

    up_capture = (up["fund"].mean() / up["benchmark"].mean()) * 100 if not up.empty else None
    down_capture = (down["fund"].mean() / down["benchmark"].mean()) * 100 if not down.empty else None

    return {
        "Rolling Return (CAGR)": round(cagr, 2),
        "Standard Deviation": round(std_dev, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Alpha": round(alpha, 2),
        "Beta": round(beta, 2),
        "Upside Capture": round(up_capture, 2),
        "Downside Capture": round(down_capture, 2)
    }

def compute_score(metrics, weights, total_weight):
    score = 0
    for metric, weight in weights.items():
        val = metrics.get(metric)
        if val is None or val == "":
            continue
        # Normalize each score relative to benchmark = 100
        if metric in ["Standard Deviation", "Downside Capture"]:
            val = max(0, 200 - val)
        score += val * (weight / total_weight)
    return round(score / 10, 2)  # Score is out of 10
