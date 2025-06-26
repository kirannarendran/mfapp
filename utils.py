import numpy as np

def compute_metrics(returns, benchmark_returns):
    excess_returns = returns - benchmark_returns[-len(returns):]
    std_dev = returns.std()
    downside = returns[returns < 0].std()
    cagr = (1 + returns.mean()) ** 252 - 1
    sharpe = returns.mean() / std_dev * np.sqrt(252)
    upside = returns[returns > 0].mean()
    downside_capture = returns.mean() / benchmark_returns.mean()
    alpha = excess_returns.mean()

    return {
        "Standard Deviation": std_dev,
        "Downside Capture": downside_capture,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Upside Capture": upside,
        "Alpha": alpha,
    }

def compute_score(row, weights):
    return sum(row[k] * weights[k] / 100 for k in weights)
