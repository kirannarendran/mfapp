import pandas as pd

def calculate_metrics(fund_name, fund_data):
    nav_data = fund_data.get("data", [])
    df = pd.DataFrame(nav_data)
    
    if df.empty or "nav" not in df.columns:
        return {
            "Fund": fund_name,
            "CAGR": 0,
            "Standard Deviation": 0,
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0
        }

    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(inplace=True)
    df = df.sort_values("date")
    
    df["returns"] = df["nav"].pct_change()
    
    cagr = (df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1 / ((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25)) - 1
    std_dev = df["returns"].std() * (252 ** 0.5)
    sharpe = df["returns"].mean() / df["returns"].std() * (252 ** 0.5)
    downside_returns = df["returns"][df["returns"] < 0]
    sortino = df["returns"].mean() / downside_returns.std() * (252 ** 0.5) if not downside_returns.empty else 0

    return {
        "Fund": fund_name,
        "CAGR": round(cagr * 100, 2),
        "Standard Deviation": round(std_dev * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2)
    }

def benchmark_metrics():
    try:
        df = pd.read_csv("bse500_returns.csv", parse_dates=["Date"])
        df = df.sort_values("Date")
        df["returns"] = df["Close"].pct_change()
        
        cagr = (df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / ((df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25)) - 1
        std_dev = df["returns"].std() * (252 ** 0.5)
        sharpe = df["returns"].mean() / df["returns"].std() * (252 ** 0.5)
        downside_returns = df["returns"][df["returns"] < 0]
        sortino = df["returns"].mean() / downside_returns.std() * (252 ** 0.5) if not downside_returns.empty else 0

        return {
            "Fund": "BSE 500 Benchmark",
            "CAGR": round(cagr * 100, 2),
            "Standard Deviation": round(std_dev * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2)
        }

    except Exception as e:
        return {
            "Fund": "BSE 500 Benchmark",
            "CAGR": 0,
            "Standard Deviation": 0,
            "Sharpe Ratio": 0,
            "Sortino Ratio": 0
        }
