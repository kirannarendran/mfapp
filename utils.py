# utils.py
import pandas as pd
import requests
import numpy as np
from datetime import datetime

def calculate_cagr(df):
    df = df.sort_values("date")
    start_value = df["nav"].iloc[0]
    end_value = df["nav"].iloc[-1]
    num_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365
    return ((end_value / start_value) ** (1 / num_years)) - 1 if start_value > 0 and num_years > 0 else np.nan

def calculate_std_dev(df):
    return df["nav"].pct_change().std() * np.sqrt(252)

def calculate_sharpe_ratio(df, risk_free_rate=0.05):
    returns = df["nav"].pct_change().dropna()
    excess_returns = returns - (risk_free_rate / 252)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def calculate_sortino_ratio(df, risk_free_rate=0.05):
    returns = df["nav"].pct_change().dropna()
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    excess_returns = returns.mean() - (risk_free_rate / 252)
    return excess_returns / downside_std if downside_std > 0 else np.nan

def fetch_fund_data(fund_code):
    url = f"https://api.mfapi.in/mf/{fund_code}"
    try:
        response = requests.get(url)
        data = response.json()
        if "data" in data:
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            df = df.dropna(subset=["date", "nav"])
            df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
            df = df.dropna()
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data for {fund_code}: {e}")
        return pd.DataFrame()

def calculate_metrics(fund_code):
    df = fetch_fund_data(fund_code)
    if df.empty:
        return None
    metrics = {
        "CAGR": calculate_cagr(df) * 100,
        "Standard Deviation": calculate_std_dev(df) * 100,
        "Sharpe Ratio": calculate_sharpe_ratio(df),
        "Sortino Ratio": calculate_sortino_ratio(df),
    }
    return metrics

def benchmark_metrics():
    # Replace this with fixed benchmark values if local CSV fails or is not available
    return {
        "CAGR": 10,
        "Standard Deviation": 12,
        "Sharpe Ratio": 0.8,
        "Sortino Ratio": 1.0
    }
