import pandas as pd

def filter_direct_growth(df):
    """Filter for Direct Growth plans only."""
    mask = (
        df['scheme_name'].str.contains('direct', case=False, na=False) &
        df['scheme_name'].str.contains('growth', case=False, na=False) &
        ~df['scheme_name'].str.contains('idcw', case=False, na=False) &
        ~df['scheme_name'].str.contains('dividend', case=False, na=False)
    )
    return df[mask]
