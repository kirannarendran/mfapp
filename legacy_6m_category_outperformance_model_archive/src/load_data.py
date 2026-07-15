import sqlite3
import pandas as pd
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data():
    print("Loading configuration...")
    config = load_config()
    db_path = config['data']['database_path']
    subset_years = config['filters'].get('subset_years', None)
    
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    funds_query = "SELECT * FROM funds"
    print("Extracting funds table...")
    funds_df = pd.read_sql_query(funds_query, conn)
    
    nav_query = "SELECT * FROM nav_history"
    if subset_years:
        print(f"Extracting nav_history for the last {subset_years} years for testing...")
        nav_query += f" WHERE date >= date('now', '-{subset_years} years')"
    else:
        print("Extracting full nav_history...")
        
    nav_df = pd.read_sql_query(nav_query, conn)
    conn.close()
    
    print(f"Loaded {len(funds_df)} funds and {len(nav_df)} NAV records.")
    
    os.makedirs("data/raw", exist_ok=True)
    funds_df.to_parquet("data/raw/funds.parquet", index=False)
    nav_df.to_parquet("data/raw/nav_history.parquet", index=False)
    print("Raw data saved to data/raw/")

if __name__ == "__main__":
    load_data()
