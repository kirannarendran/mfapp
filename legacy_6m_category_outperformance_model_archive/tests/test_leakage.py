import pandas as pd
import numpy as np

def test_leakage():
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    
    # 1. Target End Date must be after Date + Horizon
    # Since horizon is 6 months, it should be at least 180 days.
    valid_targets = df[df['forward_ret_6m'].notna()]
    days_diff = (pd.to_datetime(valid_targets['target_end_date']) - pd.to_datetime(valid_targets['date'])).dt.days
    
    assert (days_diff >= 180).all(), "Leakage detected: Target end date is less than 6 months ahead."
    
    # 2. Min 10 funds per category per date
    cat_counts = df.groupby(['date', 'category']).size()
    assert (cat_counts >= 10).all(), "Leakage/Rules violated: Found categories with less than 10 funds."
    
    print("All leakage and integrity tests passed!")

if __name__ == "__main__":
    test_leakage()
