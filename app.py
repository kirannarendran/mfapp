import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Mapping of fund names to codes
mapping = {
    "Parag Parikh Flexi Cap Fund - Direct Plan - Growth": "120503",
    "HDFC Flexi Cap Fund - Growth Plan": "102638",
    "quant Flexi Cap Fund - Growth Option-Direct Plan": "120732"
}

# Load benchmark returns
@st.cache_data
def load_benchmark():
    try:
        df = pd.read_csv('bse500_returns.csv')
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date', 'Close'])
        df = df.sort_values('Date')
        df['bench_ret'] = df['Close'].pct_change()
        return df[['Date', 'bench_ret']].dropna()
    except Exception as e:
        st.error(f"⚠️ Error loading benchmark CSV: {e}")
        return pd.DataFrame(columns=['Date', 'bench_ret'])

benchmark = load_benchmark()

# Dummy NAV loader for example (replace with API call or local NAV)
@st.cache_data
def fetch_nav(code):
    df = pd.read_csv(f"nav_data/nav_{code}.csv")
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'Nav'])
    return df[['Date', 'Nav']].sort_values('Date')

# Calculate metrics
def compute_metrics(nav_df):
    merged = pd.merge(nav_df, benchmark, on='Date', how='inner')
    merged['ret'] = merged['Nav'].pct_change()
    merged = merged.dropna()

    if len(merged) < 2:
        return {k: np.nan for k in ['Rolling Return', 'SD', 'Sharpe', 'Sortino',
                                    'Downside Capture', 'Upside Capture', 'Alpha', 'Beta']}

    rets = merged['ret']
    bench = merged['bench_ret']

    mean_ret = rets.mean() * 252
    sd = rets.std() * np.sqrt(252)
    sharpe = mean_ret / sd if sd != 0 else np.nan

    downside = rets[rets < 0]
    sortino = mean_ret / (downside.std() * np.sqrt(252)) if not downside.empty else np.nan

    downside_bench = bench[bench < 0]
    downside_capture = (rets[bench < 0].mean() / downside_bench.mean()) * 100 if not downside_bench.empty else np.nan
    upside_bench = bench[bench > 0]
    upside_capture = (rets[bench > 0].mean() / upside_bench.mean()) * 100 if not upside_bench.empty else np.nan

    slope, intercept, r_value, p_value, std_err = linregress(bench, rets)
    beta = slope
    alpha = ((mean_ret - beta * (bench.mean() * 252)) * 100)

    return {
        'Rolling Return': round(mean_ret * 100, 2),
        'SD': round(sd * 100, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Downside Capture': round(downside_capture, 2),
        'Upside Capture': round(upside_capture, 2),
        'Alpha': round(alpha, 2),
        'Beta': round(beta, 2)
    }

# UI
st.markdown("## 📊 Mutual Fund Ranking Tool")
st.markdown("#### 🔍 Search and select mutual funds")
selected = st.multiselect("Start typing fund name…", list(mapping.keys()))

# Set weights
weights = {
    "Rolling Return": 1.5,
    "SD": -1.0,
    "Sharpe": 1.0,
    "Sortino": 1.0,
    "Downside Capture": 1.0,
    "Upside Capture": 1.0,
    "Alpha": 1.5,
    "Beta": -1.0
}

# Collect data
rows = []
for name in selected:
    code = mapping[name]
    nav = fetch_nav(code)
    metrics = compute_metrics(nav)
    metrics['Fund Name'] = name
    rows.append(metrics)

# Display
if rows:
    df = pd.DataFrame(rows)
    df.insert(0, "SL No", range(1, len(df) + 1))

    # Calculate Score
    score_cols = [k for k in weights if k in df.columns]
    scores = np.zeros(len(df))
    for col in score_cols:
        col_data = df[col].astype(float)
        norm = (col_data - col_data.min()) / (col_data.max() - col_data.min()) if col_data.max() != col_data.min() else 1
        scores += weights[col] * norm

    df["Score (Out of 10)"] = (10 * (scores - scores.min()) / (scores.max() - scores.min())).round(2)

    df = df.sort_values("Score (Out of 10)", ascending=False)
    display = df[["SL No", "Fund Name"] + list(weights.keys()) + ["Score (Out of 10)"]]
    st.dataframe(display, use_container_width=True)
else:
    st.info("Please select at least one fund.")
