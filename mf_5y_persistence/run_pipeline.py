import logging
from src.audit_max_date import run_max_date_audit
from src.download_amfi_history import AMFIDownloader, run_smoke_tests, download_full_history
from src.scheme_identity import map_scheme_identities
from src.build_5y_windows import build_5y_windows
from src.data_audit import DataAuditor
from src.persistence_baselines import calc_base_rates
import yaml
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting MF 5Y Persistence Pipeline")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 1. Max Date Audit
    logger.info("1. Running Max Date Audit on Legacy Data")
    run_max_date_audit(
        legacy_nav_path="../mutual_fund_ml/data/raw/nav_history.parquet",
        legacy_funds_path="../mutual_fund_ml/data/raw/funds.parquet",
        output_path="reports/tables/max_date_audit.csv"
    )
    
    # 2. Downloader Smoke Tests
    logger.info("2. Running Downloader Smoke Tests")
    dl = AMFIDownloader(raw_dir="data/raw/amfi")
    run_smoke_tests(dl)
    
    # 3. Download Full History
    logger.info("3. Downloading Full History")
    nav_df = download_full_history(dl, start_year=2013)
    
    if nav_df.empty:
        logger.error("Failed to acquire NAV data.")
        return
        
    # 4. Scheme Identity Mapping
    logger.info("4. Mapping Scheme Identities")
    funds_df = pd.DataFrame()
    if Path("../mutual_fund_ml/data/raw/funds.parquet").exists():
        funds_df = pd.read_parquet("../mutual_fund_ml/data/raw/funds.parquet")
    
    mapped_funds = map_scheme_identities(funds_df, nav_df, output_dir="reports/tables")
    
    # 5. Build 5Y Windows
    logger.info("5. Building 5-Year Windows")
    prediction_dates = pd.date_range(start='2013-01-01', end='2030-01-01', freq='Y')
    windows_df = build_5y_windows(nav_df, prediction_dates, tolerance_days=10, output_dir="reports/tables")
    
    # 6. Run Coverage Data Audit
    logger.info("6. Running Final Coverage Data Audit & Gate Check")
    auditor = DataAuditor(config)
    manifest = auditor.run_audit()
    
    # 7. Persistence Base Rates (Calculated before ML, even if ML gate fails)
    logger.info("7. Calculating Empirical Persistence Base Rates")
    calc_base_rates(windows_df, mapped_funds, output_dir="reports/tables")
    
    if manifest['pipeline_status'] != 'ready':
        logger.error(f"PIPELINE STOPPED: {manifest['pipeline_status']}")
        return
        
    logger.info("Pipeline Ready for ML.")

if __name__ == "__main__":
    main()
