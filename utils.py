# utils.py
import requests
import pandas as pd

def fetch_fund_metadata(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'meta' in data:
            return {
                'scheme_code': data['meta'].get('scheme_code'),
                'scheme_name': data['meta'].get('scheme_name')
            }
    except Exception as e:
        print(f"Error fetching metadata: {e}")
    return None

def calculate_metrics(nav_df):
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df = nav_df.sort_values('date')
    nav_df['nav'] = pd.to_numeric(nav_df['nav'], errors='coerce')
    nav_df = nav_df.dropna(subset=['nav'])

    start_nav = nav_df.iloc[0]['nav']
    end_nav = nav_df.iloc[-1]['nav']
    days = (nav_df.iloc[-1]['date'] - nav_df.iloc[0]['date']).days
    cagr = ((end_nav / start_nav) ** (365 / days) - 1) * 100 if days > 0 else 0

    nav_returns = nav_df['nav'].pct_change().dropna()
    std_dev = nav_returns.std() * (252 ** 0.5) * 100  # Annualized
    sharpe = (nav_returns.mean() / nav_returns.std()) * (252 ** 0.5)
    downside_returns = nav_returns[nav_returns < 0]
    sortino = (nav_returns.mean() / downside_returns.std()) * (252 ** 0.5) if not downside_returns.empty else 0

    return {
        'CAGR': round(cagr, 2),
        'Standard Deviation': round(std_dev, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Sortino Ratio': round(sortino, 2)
    }

def benchmark_metrics():
    df = pd.read_csv("bse500_returns.csv")
    df['date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("date")
    df['close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['close'])
    df['returns'] = df['close'].pct_change().dropna()

    downside = df['returns'][df['returns'] < 0]
    return {
        'mean': df['returns'].mean(),
        'std': df['returns'].std(),
        'downside_std': downside.std()
    }
