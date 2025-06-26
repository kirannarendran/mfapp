import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress
import yfinance as yf
import os

# ----------- Constants ----------- #
risk_free_rate = 0.05
required_days = 5 * 252  # ~5 years
output_path = "/Users/kirannarendran/Desktop/flexi_cap_fund_metrics_5y_final.xlsx"
benchmark_ticker = "^BSE500"
benchmark_cache = "/Users/kirannarendran/Desktop/bse500_cache.csv"

# ----------- Benchmark Automation ----------- #
def get_benchmark_data(ticker="^BSE500", start="2020-01-01"):
    try:
        print("⬇️ Downloading benchmark data from Yahoo Finance...")
        df = yf.download(ticker, start=start, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        df = df[['Close']].rename(columns={'Close': 'Close'})
        df.index.name = 'Date'
        df.to_csv(benchmark_cache)
    except Exception as e:
        print(f"⚠️ Benchmark download failed. Using cached data. Error: {e}")
        if not os.path.exists(benchmark_cache):
            raise RuntimeError("❌ No benchmark data available.")
        df = pd.read_csv(benchmark_cache, parse_dates=['Date'], index_col='Date')

    df = df.sort_index()
    df['benchmark'] = np.log(df['Close'] / df['Close'].shift(1))
    return df[['benchmark']].dropna()

benchmark_returns = get_benchmark_data()

# ----------- Fund List ----------- #
funds = {
    "Parag Parikh Flexi Cap Fund": "120503",
    "UTI Flexi Cap Fund": "119834",
    "HDFC Flexi Cap Fund": "102638",
    "Kotak Flexi Cap Fund": "118834",
    "Canara Robeco Flexi Cap Fund": "120832"
}

# ----------- Fetch NAV Data ----------- #
def fetch_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url)
    data = r.json().get('data', [])
    if not data:
        raise ValueError("No data returned for scheme")
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df.dropna(subset=['nav'], inplace=True)
    return df.set_index('date').sort_index()

# ----------- Compute Metrics ----------- #
def compute_metrics(nav_df, benchmark_df):
    df = nav_df.copy()
    df['ret'] = np.log(df['nav'] / df['nav'].shift(1))
    df.dropna(inplace=True)
    aligned = df.join(benchmark_df, how='inner').dropna()

    if len(aligned) < required_days:
        raise ValueError("Not enough data for 5-year calculation")

    rolling_return = (aligned['nav'].iloc[-1] / aligned['nav'].iloc[-required_days])**(1/5) - 1
    sd = aligned['ret'].std() * np.sqrt(252)
    sharpe = (aligned['ret'].mean() * 252 - risk_free_rate) / sd if sd > 0 else np.nan
    downside = aligned['ret'][aligned['ret'] < 0].std() * np.sqrt(252)
    sortino = (aligned['ret'].mean() * 252 - risk_free_rate) / downside if downside > 0 else np.nan

    slope, intercept, _, _, _ = linregress(aligned['benchmark'], aligned['ret'])
    beta = slope
    alpha = (aligned['ret'].mean() - beta * aligned['benchmark'].mean()) * 252

    upside = aligned[aligned['benchmark'] > 0]
    downside_df = aligned[aligned['benchmark'] < 0]
    upside_capture = (upside['ret'].mean() / upside['benchmark'].mean()) if not upside.empty else np.nan
    downside_capture = (downside_df['ret'].mean() / downside_df['benchmark'].mean()) if not downside_df.empty else np.nan

    return {
        'Rolling Return (5Y)': rolling_return,
        'SD': sd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Alpha': alpha,
        'Beta': beta,
        'Upside Capture': upside_capture,
        'Downside Capture': downside_capture
    }

# ----------- Process Funds ----------- #
results = []
for fund, code in funds.items():
    print(f"➡️ Processing: {fund}")
    try:
        nav_df = fetch_nav(code)
        metrics = compute_metrics(nav_df, benchmark_returns)
        metrics['Fund'] = fund
        results.append(metrics)
        time.sleep(1)
    except Exception as e:
        print(f"❌ Skipped {fund}: {e}")

# ----------- Format & Score ----------- #
if results:
    df = pd.DataFrame(results)

    percent_cols = ['Rolling Return (5Y)', 'SD', 'Alpha', 'Upside Capture', 'Downside Capture']
    for col in percent_cols:
        df[col] = df[col] * 100

    df = df.round(2)

    df = df[['Fund', 'Rolling Return (5Y)', 'SD', 'Sharpe', 'Sortino',
             'Alpha', 'Beta', 'Upside Capture', 'Downside Capture']]

    df.rename(columns={
        'Rolling Return (5Y)': 'Rolling Return (5Y) %',
        'SD': 'SD %',
        'Alpha': 'Alpha %',
        'Upside Capture': 'Upside Capture %',
        'Downside Capture': 'Downside Capture %'
    }, inplace=True)

    df_valid = df.dropna()
    df_norm = df_valid.copy()

    for col in ['Rolling Return (5Y) %', 'SD %', 'Sharpe', 'Sortino',
                'Alpha %', 'Beta', 'Upside Capture %', 'Downside Capture %']:
        if col in ['SD %', 'Beta', 'Downside Capture %']:
            df_norm[col] = df_norm[col].max() - df_norm[col]
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min) if col_max != col_min else 1

    df_norm['Score'] = (
        df_norm['Sortino'] * 0.20 +
        df_norm['Downside Capture %'] * 0.20 +
        df_norm['SD %'] * 0.20 +
        df_norm['Rolling Return (5Y) %'] * 0.15 +
        df_norm['Alpha %'] * 0.10 +
        df_norm['Sharpe'] * 0.05 +
        df_norm['Upside Capture %'] * 0.05 +
        df_norm['Beta'] * 0.05
    )

    df_norm['Score'] = (df_norm['Score'] * 10).round(2)
    df = df.merge(df_norm[['Fund', 'Score']], on='Fund', how='left')
    df = df.sort_values(by='Score', ascending=False)

    # Add category average
    df = pd.concat([df, pd.DataFrame([{
        'Fund': 'Category Average',
        'Rolling Return (5Y) %': round(df['Rolling Return (5Y) %'].mean(), 2),
        'SD %': '', 'Sharpe': '', 'Sortino': '', 'Alpha %': '',
        'Beta': '', 'Upside Capture %': '', 'Downside Capture %': '', 'Score': ''
    }])], ignore_index=True)

    df.to_excel(output_path, index=False)
    print(f"✅ Excel saved to {output_path}")
else:
    print("⚠️ No valid data to process.")
