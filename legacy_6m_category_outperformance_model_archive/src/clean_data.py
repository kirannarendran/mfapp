import pandas as pd
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def clean_data():
    config = load_config()
    date_col = config['data']['date_column']
    nav_col = config['data']['nav_column']
    
    print("Loading raw data...")
    funds = pd.read_parquet("data/raw/funds.parquet")
    navs = pd.read_parquet("data/raw/nav_history.parquet")
    
    print("Cleaning navs...")
    # The DB date might be YYYY-MM-DD or DD-MM-YYYY depending on how it was synced.
    # We will try both using pd.to_datetime with format='mixed' or infer_datetime_format
    navs[date_col] = pd.to_datetime(navs[date_col], format='mixed', errors='coerce')
    
    navs = navs.dropna(subset=[date_col, nav_col])
    navs[nav_col] = pd.to_numeric(navs[nav_col], errors='coerce')
    navs = navs.dropna(subset=[nav_col])
    navs = navs.drop_duplicates(subset=['scheme_code', date_col])
    navs = navs.sort_values(['scheme_code', date_col])
    
    print("Cleaning funds and mapping plan/option types...")
    funds['scheme_name'] = funds['scheme_name'].fillna('')
    # Extract plan type (Direct vs Regular)
    funds['plan_type'] = 'Regular' # default
    funds.loc[funds['scheme_name'].str.contains('Direct', case=False), 'plan_type'] = 'Direct'
    
    # Extract option type (Growth vs Dividend/IDCW)
    funds['option_type'] = 'Dividend' # default
    funds.loc[funds['scheme_name'].str.contains('Growth', case=False), 'option_type'] = 'Growth'
    
    if config['filters']['keep_growth_only']:
        print("Filtering for Growth options only...")
        funds = funds[funds['option_type'] == 'Growth']
        
    print("Filtering for Direct plans only...")
    funds = funds[funds['plan_type'] == 'Direct']
        
    print(f"Remaining funds: {len(funds)}")
    
    # Merge navs with funds to get category and plan_type
    df = navs.merge(funds[['scheme_code', 'category', 'plan_type']], on='scheme_code', how='inner')
    
    # Drop where category is missing or empty
    df = df.dropna(subset=['category'])
    df = df[df['category'].str.strip() != '']
    df = df[df['category'].str.strip() != 'Others']
    
    print("Saving cleaned data...")
    os.makedirs("data/interim", exist_ok=True)
    df.to_parquet("data/interim/cleaned_data.parquet", index=False)
    print(f"Cleaned data saved: {len(df)} rows.")

if __name__ == "__main__":
    clean_data()
