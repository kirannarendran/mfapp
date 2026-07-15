import pandas as pd

def load_nav_history(path):
    return pd.read_parquet(path)

def load_scheme_metadata(path):
    return pd.read_parquet(path)
