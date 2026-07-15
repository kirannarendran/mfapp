import pandas as pd
import numpy as np
import yaml
from pandas.tseries.offsets import MonthEnd

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_target():
    config = load_config()
    horizon = config['target']['horizon_months']
    date_col = config['data']['date_column']
    
    print("Loading features...")
    df = pd.read_parquet("data/processed/features.parquet")
    
    df = df.sort_values(['scheme_code', date_col])
    
    print(f"Calculating {horizon}-month forward target...")
    # Calculate exact calendar target end date
    df['target_end_date'] = df[date_col] + pd.DateOffset(months=horizon)
    df['target_end_date'] = df['target_end_date'] + MonthEnd(0)
    
    # We want the return from 'date_col' to 'target_end_date'.
    # This is exactly the 'ret_6m' value recorded at 'target_end_date'.
    future_returns = df[['scheme_code', date_col, 'ret_6m', 'excess_ret_6m']].rename(
        columns={date_col: 'target_end_date', 'ret_6m': 'forward_ret_6m', 'excess_ret_6m': 'forward_excess_ret_6m'}
    )
    
    df = df.merge(future_returns, on=['scheme_code', 'target_end_date'], how='left')
    
    # Calculate category median forward return for each date
    cat_median = df.groupby([date_col, 'category'])['forward_ret_6m'].transform('median')
    
    # Fix: leave missing targets as NaN, do NOT treat them as zero
    df['target'] = np.where(df['forward_ret_6m'].isna(), np.nan, (df['forward_ret_6m'] > cat_median).astype(float))
    
    # Mark evaluation status
    df['evaluation_status'] = np.where(df['forward_ret_6m'].notna(), 'evaluated', 'pending')
    
    # We just know they don't have a valid target yet.
    df['has_target'] = df['forward_ret_6m'].notna()
    
    print("Saving modeling dataset...")
    df.to_parquet("data/processed/model_dataset.parquet", index=False)
    print(f"Model dataset saved: {len(df)} rows.")

if __name__ == "__main__":
    build_target()
