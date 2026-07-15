import pandas as pd
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def run_max_date_audit(legacy_nav_path, legacy_funds_path, output_path):
    """
    Quarantines rows with the max date (likely the extraction timestamp)
    from the legacy parquet file.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(legacy_nav_path).exists():
        logger.error(f"Legacy NAV path {legacy_nav_path} not found.")
        return
        
    df = pd.read_parquet(legacy_nav_path)
    df['date'] = pd.to_datetime(df['date'])
    max_d = df['date'].max()
    logger.info(f"Auditing max date: {max_d}")
    
    # Quarantine the exact max date
    affected = df[df['date'] == max_d].copy()
    
    if Path(legacy_funds_path).exists():
        funds = pd.read_parquet(legacy_funds_path)
        # Assuming funds.parquet has scheme_code, scheme_name, fund_house as AMC, category, type, isin
        # Some fields like plan_type, option_type might be missing and need parsing
        funds = funds.rename(columns={'fund_house': 'AMC'})
        affected = affected.merge(funds, on='scheme_code', how='left')
    
    # Fill missing columns expected by audit
    expected_cols = [
        'scheme_code', 'scheme_name', 'AMC', 'category', 'plan_type', 'option_type', 
        'ISIN', 'NAV', 'raw_date', 'source_file', 'legacy_ingestion_timestamp'
    ]
    
    # Map current cols
    affected['NAV'] = affected['nav']
    affected['raw_date'] = affected['date']
    affected['source_file'] = 'legacy/nav_history.parquet'
    affected['legacy_ingestion_timestamp'] = max_d
    
    for c in expected_cols:
        if c not in affected.columns:
            if c == 'ISIN' and 'isin' in affected.columns:
                affected[c] = affected['isin']
            else:
                affected[c] = 'Unknown'
            
    affected = affected[expected_cols]
    
    # Note: verification_status will be assigned during AMFI download verification phase.
    affected['verification_status'] = 'unable_to_verify'
    
    affected.to_csv(output_path, index=False)
    logger.info(f"Exported {len(affected)} max-date rows to {output_path}")
    return affected

if __name__ == "__main__":
    run_max_date_audit(
        "../mutual_fund_ml/data/raw/nav_history.parquet",
        "../mutual_fund_ml/data/raw/funds.parquet",
        "reports/tables/max_date_audit.csv"
    )
