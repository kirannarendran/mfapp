import pandas as pd
import numpy as np
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_features():
    config = load_config()
    date_col = config['data']['date_column']
    nav_col = config['data']['nav_column']
    
    print("Loading cleaned data...")
    df = pd.read_parquet("data/interim/cleaned_data.parquet")
    
    # Calculate daily returns
    df = df.sort_values(['scheme_code', date_col])
    df['log_nav'] = np.log(df[nav_col])
    df['daily_ret'] = df.groupby('scheme_code')['log_nav'].diff()
    
    print("Calculating daily volatilities...")
    # Set index for time-based rolling
    df = df.set_index(date_col)
    
    # Calculate rolling std. min_periods=60 (about 3 months) for 6m, 120 for 12m
    vol_6m = df.groupby('scheme_code')['daily_ret'].rolling('182D', min_periods=60).std() * np.sqrt(252)
    vol_12m = df.groupby('scheme_code')['daily_ret'].rolling('365D', min_periods=120).std() * np.sqrt(252)
    
    # Reset index and merge back
    df = df.reset_index()
    vol_6m = vol_6m.reset_index(name='vol_6m')
    vol_12m = vol_12m.reset_index(name='vol_12m')
    
    df = df.merge(vol_6m, on=['scheme_code', date_col])
    df = df.merge(vol_12m, on=['scheme_code', date_col])
    
    # Save the actual nav date before resampling
    df['nav_date'] = df[date_col]
    
    print("Resampling to month-end...")
    df = df.set_index(date_col)
    
    # We want the last valid NAV of each month per fund
    df_monthly = df.groupby('scheme_code').resample('ME').last()
    df_monthly = df_monthly.reset_index()
    
    # Exclude stale NAVs: if the actual NAV date is more than 5 days before the month end date, 
    # it means the fund didn't report NAV at month end (e.g. fund closed or stopped reporting)
    df_monthly['days_stale'] = (df_monthly[date_col] - df_monthly['nav_date']).dt.days
    df_monthly = df_monthly[df_monthly['days_stale'] <= 7].copy()
    
    features = []
    
    print("Building rolling returns...")
    for name, group in df_monthly.groupby('scheme_code'):
        group = group.sort_values(date_col).copy()
        
        # Trailing returns
        group['ret_1m'] = group['log_nav'] - group['log_nav'].shift(1)
        group['ret_3m'] = group['log_nav'] - group['log_nav'].shift(3)
        group['ret_6m'] = group['log_nav'] - group['log_nav'].shift(6)
        group['ret_12m'] = group['log_nav'] - group['log_nav'].shift(12)
        
        # Momentum excluding recent month
        group['mom_12_1'] = group['log_nav'].shift(1) - group['log_nav'].shift(12)
        group['mom_6_1'] = group['log_nav'].shift(1) - group['log_nav'].shift(6)
        
        # Distance from Moving Averages
        ma_6m = group[nav_col].rolling(6).mean()
        ma_12m = group[nav_col].rolling(12).mean()
        group['dist_ma_6m'] = (group[nav_col] / ma_6m) - 1
        group['dist_ma_12m'] = (group[nav_col] / ma_12m) - 1
        
        features.append(group)
        
    df_features = pd.concat(features)
    
    print("Calculating relative features...")
    # Count eligible funds per category and date
    cat_counts = df_features.groupby([date_col, 'category']).size().reset_index(name='fund_count')
    df_features = df_features.merge(cat_counts, on=[date_col, 'category'])
    
    # Require a minimum of 10 eligible funds
    df_features = df_features[df_features['fund_count'] >= 10].copy()
    
    df_features['rank_ret_6m'] = df_features.groupby([date_col, 'category'])['ret_6m'].rank(pct=True)
    df_features['rank_ret_12m'] = df_features.groupby([date_col, 'category'])['ret_12m'].rank(pct=True)
    df_features['rank_vol_12m'] = df_features.groupby([date_col, 'category'])['vol_12m'].rank(pct=True)
    
    cat_median = df_features.groupby([date_col, 'category'])['ret_6m'].transform('median')
    df_features['excess_ret_6m'] = df_features['ret_6m'] - cat_median
    
    cat_median_12 = df_features.groupby([date_col, 'category'])['ret_12m'].transform('median')
    df_features['excess_ret_12m'] = df_features['ret_12m'] - cat_median_12
    
    print("Saving features...")
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_parquet("data/processed/features.parquet", index=False)
    print(f"Features saved: {len(df_features)} rows.")

if __name__ == "__main__":
    build_features()
