import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def get_next_nav(df, scheme_code, target_date, max_tolerance_days=10):
    subset = df[(df['scheme_code'] == scheme_code) & (df['parsed_NAV_date'] >= target_date)]
    if subset.empty:
        return pd.NaT, np.nan
        
    first_row = subset.iloc[0]
    actual_date = first_row['parsed_NAV_date']
    
    if (actual_date - target_date).days <= max_tolerance_days:
        return actual_date, first_row['nav']
    return pd.NaT, np.nan

def build_5y_windows(nav_df, prediction_dates, tolerance_days=10, output_dir="reports/tables"):
    """
    Builds exact calendar-aligned 5y past and future windows.
    Returns the dataframe of completed windows.
    """
    nav_df = nav_df.sort_values('parsed_NAV_date')
    
    schemes = nav_df['scheme_code'].unique()
    windows = []
    
    logger.info(f"Building 5y windows for {len(schemes)} schemes across {len(prediction_dates)} prediction dates.")
    
    for p_date in prediction_dates:
        past_date = p_date - pd.DateOffset(years=5)
        future_date = p_date + pd.DateOffset(years=5)
        
        # We need a quick way to look up first nav >= date per scheme.
        # Since we're in pandas, groupby and searching might be slow, but it's a batch script.
        # Let's optimize slightly:
        
        p_slice = nav_df[(nav_df['parsed_NAV_date'] >= p_date) & (nav_df['parsed_NAV_date'] <= p_date + pd.Timedelta(days=tolerance_days))]
        past_slice = nav_df[(nav_df['parsed_NAV_date'] >= past_date) & (nav_df['parsed_NAV_date'] <= past_date + pd.Timedelta(days=tolerance_days))]
        future_slice = nav_df[(nav_df['parsed_NAV_date'] >= future_date) & (nav_df['parsed_NAV_date'] <= future_date + pd.Timedelta(days=tolerance_days))]
        
        p_first = p_slice.groupby('scheme_code').first().reset_index()
        past_first = past_slice.groupby('scheme_code').first().reset_index()
        future_first = future_slice.groupby('scheme_code').first().reset_index()
        
        # Join them
        m1 = pd.merge(p_first[['scheme_code', 'parsed_NAV_date', 'nav']], past_first[['scheme_code', 'parsed_NAV_date', 'nav']], on='scheme_code', suffixes=('_p', '_past'), how='left')
        m2 = pd.merge(m1, future_first[['scheme_code', 'parsed_NAV_date', 'nav']], on='scheme_code', how='left')
        m2.rename(columns={'parsed_NAV_date': 'parsed_NAV_date_future', 'nav': 'nav_future'}, inplace=True)
        
        for _, row in m2.iterrows():
            w = {
                'scheme_code': row['scheme_code'],
                'prediction_date': p_date,
                'past_start_intended': past_date,
                'past_start_actual': row['parsed_NAV_date_past'],
                'prediction_NAV_date': row['parsed_NAV_date_p'],
                'future_end_intended': future_date,
                'future_end_actual': row['parsed_NAV_date_future'],
                'past_5y_CAGR': np.nan,
                'future_5y_CAGR': np.nan,
                'window_status': 'incomplete'
            }
            
            if pd.notna(row['nav_past']) and pd.notna(row['nav_p']) and row['nav_past'] > 0:
                w['past_5y_CAGR'] = (row['nav_p'] / row['nav_past']) ** (1/5) - 1
                
            if pd.notna(row['nav_p']) and pd.notna(row['nav_future']) and row['nav_p'] > 0:
                w['future_5y_CAGR'] = (row['nav_future'] / row['nav_p']) ** (1/5) - 1
                
            if pd.notna(w['past_start_actual']) and pd.notna(w['prediction_NAV_date']) and pd.notna(w['future_end_actual']):
                w['window_status'] = 'complete'
                
            windows.append(w)
            
    win_df = pd.DataFrame(windows)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    win_df.to_csv(out_dir / "five_year_window_audit.csv", index=False)
    
    complete = win_df[win_df['window_status'] == 'complete']
    logger.info(f"Built {len(complete)} complete 5y past-to-future transitions.")
    
    return win_df
