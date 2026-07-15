import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
import io
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AMFIDownloader:
    def __init__(self, raw_dir="data/raw/amfi"):
        self.raw_dir = Path(raw_dir)
        self.base_url = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"

    def fetch_chunk(self, start_date, end_date, allow_overlap=False):
        """
        Fetch a chunk of NAV data from AMFI natively.
        Validates the 90-day inclusive limit explicitly.
        """
        inclusive_days = (end_date - start_date).days + 1
        if inclusive_days > 90 and not allow_overlap: # Strict 90-day AMFI limit
            raise ValueError(f"Requested chunk ({inclusive_days} days) exceeds 90-day strict limit.")
            
        frmdt = start_date.strftime("%d-%b-%Y")
        todt = end_date.strftime("%d-%b-%Y")
        url = f"{self.base_url}?frmdt={frmdt}&todt={todt}"
        
        chunk_dir = self.raw_dir / str(start_date.year)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = chunk_dir / f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.txt"
        
        return self._execute_request(url, cache_file, frmdt, todt, start_date, end_date)

    def _execute_request(self, url, cache_file, frmdt, todt, start_date, end_date):
        meta = {
            'request_start_date': start_date,
            'request_end_date': end_date,
            'inclusive_day_count': (end_date - start_date).days + 1,
            'request_parameters': f"frmdt={frmdt}&todt={todt}",
            'download_timestamp': datetime.now(),
            'HTTP_status': None,
            'content_type': None,
            'checksum': None,
            'raw_byte_count': 0,
            'raw_row_count': 0,
            'parsed_row_count': 0,
            'parser_version': '1.0',
            'parse_status': 'pending',
            'minimum_parsed_date': pd.NaT,
            'maximum_parsed_date': pd.NaT,
            'failure_reason': None,
            'validation_status': 'failed'
        }
        
        try:
            if cache_file.exists():
                logger.info(f"Using cached file for {frmdt} to {todt}")
                with open(cache_file, "rb") as f:
                    content_bytes = f.read()
                meta['HTTP_status'] = 200 # Implicitly OK since it's cached
                content = content_bytes.decode('utf-8', errors='replace')
            else:
                logger.info(f"Downloading from {frmdt} to {todt}")
                for attempt in range(3):
                    try:
                        response = requests.get(url, timeout=120)
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt == 2:
                            raise e
                        logger.warning(f"Timeout/Error on {frmdt} to {todt}, retrying... ({attempt+1}/3)")
                        time.sleep(5.0)
                        
                meta['HTTP_status'] = response.status_code
                meta['content_type'] = response.headers.get('Content-Type', '')
                
                if response.status_code != 200:
                    meta['failure_reason'] = f"HTTP {response.status_code}"
                    return None, meta
                    
                content = response.text
                content_bytes = content.encode('utf-8')
                
                # Reject HTML or error pages
                if '<html>' in content.lower() or '<!doctype html>' in content.lower():
                    meta['failure_reason'] = "Response is HTML, likely an error page."
                    return None, meta
                    
                # Save Raw Response Immutably
                with open(cache_file, "wb") as f:
                    f.write(content_bytes)
                time.sleep(1.0) # Polite rate limiting
                
            meta['raw_byte_count'] = len(content_bytes)
            meta['checksum'] = hashlib.sha256(content_bytes).hexdigest()
            meta['raw_row_count'] = content.count('\n')
            
            # Require expected NAV header
            if "Net Asset Value" not in content and "Scheme Code" not in content:
                meta['failure_reason'] = "Expected NAV headers missing from response."
                return None, meta
                
            meta['response_header_sample'] = content[:200].replace('\n', ' ')
            
            # Parse logic
            df = self.parse_amfi_text(content)
            meta['parsed_row_count'] = len(df)
            
            if len(df) == 0:
                meta['failure_reason'] = "No valid NAV rows parsed."
                return df, meta
                
            meta['minimum_parsed_date'] = df['parsed_NAV_date'].min()
            meta['maximum_parsed_date'] = df['parsed_NAV_date'].max()
            
            # Validate dates (allow some boundary slippage due to weekends, but shouldn't be > 10 days out of bounds)
            if meta['minimum_parsed_date'] < (start_date - timedelta(days=10)) or meta['maximum_parsed_date'] > (end_date + timedelta(days=10)):
                meta['failure_reason'] = "Parsed dates fall materially outside requested range."
                return df, meta
                
            meta['parse_status'] = 'success'
            meta['validation_status'] = 'success'
            
            return df, meta
            
        except requests.exceptions.RequestException as e:
            meta['failure_reason'] = f"Network Exception: {str(e)}"
            return None, meta
        except Exception as e:
            meta['failure_reason'] = f"Exception: {str(e)}"
            return None, meta

    def parse_amfi_text(self, text):
        lines = [line for line in text.split('\n') if ';' in line and line.strip()]
        if not lines:
            return pd.DataFrame()
            
        df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=';', low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        
        rename_map = {
            'Scheme Code': 'scheme_code',
            'Scheme Name': 'scheme_name',
            'Net Asset Value': 'nav',
            'Date': 'parsed_NAV_date' # Strictly keeping extraction independent
        }
        
        # Retain ISIN columns if available for Scheme Identity
        for col in ['ISIN Div Payout/ISIN Growth', 'ISIN Div Payout/ ISIN Growth', 'ISIN Growth']:
            if col in df.columns:
                rename_map[col] = 'isin'
                break
                
        df = df.rename(columns=rename_map)
        
        # We need code, name, nav, parsed_NAV_date
        req_cols = ['scheme_code', 'scheme_name', 'nav', 'parsed_NAV_date']
        for c in req_cols:
            if c not in df.columns:
                return pd.DataFrame() # Fail gracefully if columns are messed up
                
        out_cols = req_cols + (['isin'] if 'isin' in df.columns else [])
        df = df[out_cols].dropna(subset=['scheme_code', 'nav', 'parsed_NAV_date']).copy()
        
        df['scheme_code'] = pd.to_numeric(df['scheme_code'], errors='coerce')
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df['parsed_NAV_date'] = pd.to_datetime(df['parsed_NAV_date'], format='%d-%b-%Y', errors='coerce')
        
        # Add extraction timestamps
        df['data_extraction_timestamp'] = pd.Timestamp.now()
        df['raw_report_date'] = df['parsed_NAV_date']
        
        return df.dropna(subset=['scheme_code', 'nav', 'parsed_NAV_date'])

def run_smoke_tests(downloader):
    """Executes the dynamic and auditable live smoke tests."""
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define exact requests
    today = pd.Timestamp.now().floor('D')
    recent_start = today - pd.DateOffset(days=35)
    recent_end = today - pd.DateOffset(days=5) # Completed 30 days
    
    hist_start = pd.Timestamp('2015-01-01')
    hist_end = pd.Timestamp('2015-03-31') # 90 days inclusive
    
    over_start = pd.Timestamp('2015-03-15')
    over_end = pd.Timestamp('2015-04-15')
    
    tests = [
        ('recent_30_day', recent_start, recent_end),
        ('historical_90_day', hist_start, hist_end),
        ('overlapping_historical', over_start, over_end)
    ]
    
    results = []
    failed = False
    
    for name, s, e in tests:
        logger.info(f"Running smoke test: {name} ({s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')})")
        df, meta = downloader.fetch_chunk(s, e, allow_overlap=True)
        meta['test_name'] = name
        results.append(meta)
        if meta['validation_status'] != 'success':
            failed = True
            logger.error(f"Smoke test {name} failed: {meta['failure_reason']}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "download_smoke_test.csv", index=False)
    
    if failed:
        res_df[res_df['validation_status'] != 'success'].to_csv(out_dir / "download_failures.csv", index=False)
        raise RuntimeError("Downloader smoke tests failed. Halting acquisition.")
        
    logger.info("Live downloader smoke tests passed.")
    return True

def download_full_history(downloader, start_year=2013):
    """Chunked downloader strictly respecting 90-day bounds."""
    start_date = pd.Timestamp(f"{start_year}-01-01")
    final_end_date = pd.Timestamp.now().floor('D') - pd.Timedelta(days=1)
    
    current = start_date
    manifests = []
    dfs = []
    
    while current <= final_end_date:
        # chunk_end = min(current + 89 days, final_end_date)
        chunk_end = min(current + timedelta(days=89), final_end_date)
        
        df, meta = downloader.fetch_chunk(current, chunk_end)
        manifests.append(meta)
        
        if df is not None and not df.empty:
            dfs.append(df)
        
        # next chunk starts exactly day after
        current = chunk_end + timedelta(days=1)
        
    mf_df = pd.DataFrame(manifests)
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    mf_df.to_csv(out_dir / "raw_file_manifest.csv", index=False)
    
    # Combine all
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        
        # Verify deduplication
        dups = combined[combined.duplicated(subset=['scheme_code', 'parsed_NAV_date'], keep=False)]
        if not dups.empty:
            conflict_check = dups.groupby(['scheme_code', 'parsed_NAV_date'])['nav'].nunique()
            conflicts = conflict_check[conflict_check > 1]
            if not conflicts.empty:
                logger.error(f"Found {len(conflicts)} conflicting NAVs.")
                raise ValueError("Conflicting NAVs for same scheme and date.")
            
            combined = combined.drop_duplicates(subset=['scheme_code', 'parsed_NAV_date'])
        
        # Final output
        final_dir = Path("data/processed")
        final_dir.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(final_dir / "full_nav_history.parquet", index=False)
        logger.info(f"Full history acquired: {len(combined)} rows saved.")
        return combined
    return pd.DataFrame()
    
if __name__ == "__main__":
    dl = AMFIDownloader()
    run_smoke_tests(dl)
    # The pipeline runner will execute download_full_history
