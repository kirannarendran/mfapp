import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def wilson_interval(successes, n, confidence=0.95):
    """Calculates Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    
    return max(0.0, lower), min(1.0, upper)

def calc_base_rates(windows_df, funds_df=None, output_dir="reports/tables"):
    """
    Calculates empirical persistence probabilities for funds with past CAGR between bounds.
    Primary band: 11% - 13%, Target: >= 12%
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = windows_df[windows_df['window_status'] == 'complete'].copy()
    if df.empty:
        logger.warning("No complete windows available for base rate calculation.")
        # Output empty schemas
        return
        
    if funds_df is not None and not funds_df.empty:
        df = df.merge(funds_df[['scheme_code', 'category']], on='scheme_code', how='left')
    else:
        df['category'] = 'Unknown'
        
    df['cohort'] = df['prediction_date'].dt.year
    
    bands = [
        ("11_to_13", 0.11, 0.13),
        ("10_to_14", 0.10, 0.14),
        ("8_to_16", 0.08, 0.16)
    ]
    
    results = []
    
    for band_name, lower_b, upper_b in bands:
        # Overall
        band_df = df[(df['past_5y_CAGR'] >= lower_b) & (df['past_5y_CAGR'] <= upper_b)]
        
        n = len(band_df)
        successes = len(band_df[band_df['future_5y_CAGR'] >= 0.12])
        p = successes / n if n > 0 else 0
        w_low, w_high = wilson_interval(successes, n)
        
        row = {
            'band': band_name,
            'grouping': 'Overall',
            'group_name': 'All',
            'case_count': n,
            'success_count_future_CAGR_ge_12': successes,
            'estimated_probability': p,
            'Wilson_interval_lower': w_low,
            'Wilson_interval_upper': w_high,
            'median_future_CAGR': band_df['future_5y_CAGR'].median() if n > 0 else np.nan,
            'p10_future_CAGR': band_df['future_5y_CAGR'].quantile(0.10) if n > 0 else np.nan,
            'p25_future_CAGR': band_df['future_5y_CAGR'].quantile(0.25) if n > 0 else np.nan,
            'p75_future_CAGR': band_df['future_5y_CAGR'].quantile(0.75) if n > 0 else np.nan,
            'p90_future_CAGR': band_df['future_5y_CAGR'].quantile(0.90) if n > 0 else np.nan
        }
        results.append(row)
        
        # By Category
        for cat, grp in band_df.groupby('category'):
            nc = len(grp)
            if nc < 5:
                continue # Suppress unstable
            sc = len(grp[grp['future_5y_CAGR'] >= 0.12])
            pc = sc / nc
            wc_low, wc_high = wilson_interval(sc, nc)
            
            results.append({
                'band': band_name,
                'grouping': 'Category',
                'group_name': cat,
                'case_count': nc,
                'success_count_future_CAGR_ge_12': sc,
                'estimated_probability': pc,
                'Wilson_interval_lower': wc_low,
                'Wilson_interval_upper': wc_high,
                'median_future_CAGR': grp['future_5y_CAGR'].median(),
                'p10_future_CAGR': grp['future_5y_CAGR'].quantile(0.10),
                'p25_future_CAGR': grp['future_5y_CAGR'].quantile(0.25),
                'p75_future_CAGR': grp['future_5y_CAGR'].quantile(0.75),
                'p90_future_CAGR': grp['future_5y_CAGR'].quantile(0.90)
            })
            
        # By Cohort
        for co, grp in band_df.groupby('cohort'):
            nc = len(grp)
            sc = len(grp[grp['future_5y_CAGR'] >= 0.12])
            pc = sc / nc if nc > 0 else 0
            wc_low, wc_high = wilson_interval(sc, nc)
            
            results.append({
                'band': band_name,
                'grouping': 'Cohort',
                'group_name': str(co),
                'case_count': nc,
                'success_count_future_CAGR_ge_12': sc,
                'estimated_probability': pc,
                'Wilson_interval_lower': wc_low,
                'Wilson_interval_upper': wc_high,
                'median_future_CAGR': grp['future_5y_CAGR'].median() if nc > 0 else np.nan,
                'p10_future_CAGR': grp['future_5y_CAGR'].quantile(0.10) if nc > 0 else np.nan,
                'p25_future_CAGR': grp['future_5y_CAGR'].quantile(0.25) if nc > 0 else np.nan,
                'p75_future_CAGR': grp['future_5y_CAGR'].quantile(0.75) if nc > 0 else np.nan,
                'p90_future_CAGR': grp['future_5y_CAGR'].quantile(0.90) if nc > 0 else np.nan
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "base_rate_by_cagr_band.csv", index=False)
    logger.info("Base rates calculated.")
    return res_df
