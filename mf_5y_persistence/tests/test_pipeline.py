import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.download_amfi_history import AMFIDownloader
from src.build_5y_windows import build_5y_windows
from src.persistence_baselines import wilson_interval
from src.scheme_identity import normalize_scheme_name, classify_plan_and_option

def test_downloader_chunk_boundaries():
    dl = AMFIDownloader(raw_dir="tests/tmp")
    start = pd.Timestamp("2024-01-01")
    # This should fail if we request > 90 days
    end = pd.Timestamp("2024-05-01") # 122 days
    with pytest.raises(ValueError, match="exceeds 90-day strict limit"):
        dl.fetch_chunk(start, end)
        
def test_wilson_interval():
    # 0 successes, 0 trials
    w_low, w_high = wilson_interval(0, 0)
    assert w_low == 0.0 and w_high == 0.0
    
    # 10 successes, 100 trials, 95%
    w_low, w_high = wilson_interval(10, 100)
    assert 0.04 < w_low < 0.10
    assert 0.10 < w_high < 0.18
    
def test_scheme_name_variants():
    variants = [
        ("Direct Plan - Growth", "Direct", "Growth"),
        ("Direct Growth", "Direct", "Growth"),
        ("Growth - Direct Plan", "Direct", "Growth"),
        ("Growth Option - Direct", "Direct", "Growth"),
        ("Regular Plan Growth", "Regular", "Growth"),
        ("Growth Option", "Unclassified", "Growth"),
        ("IDCW", "Unclassified", "IDCW"),
        ("Dividend Reinvestment", "Unclassified", "IDCW"),
        ("Regular Dividend", "Regular", "IDCW")
    ]
    
    for raw, exp_plan, exp_opt in variants:
        p, o = classify_plan_and_option(raw)
        assert p == exp_plan
        assert o == exp_opt

def test_build_5y_windows():
    # Synthetic df
    dates = pd.date_range("2015-01-01", "2026-01-01", freq='D')
    df = pd.DataFrame({
        'scheme_code': [1]*len(dates) + [2]*len(dates),
        'parsed_NAV_date': list(dates) + list(dates),
        'nav': list(range(1, len(dates)+1)) + list(range(1, len(dates)+1))
    })
    
    pred_dates = pd.to_datetime(['2020-03-31'])
    wins = build_5y_windows(df, pred_dates, tolerance_days=10, output_dir="tests/tmp")
    
    assert len(wins) == 2
    for _, w in wins.iterrows():
        assert w['window_status'] == 'complete'
        assert pd.notna(w['past_start_actual'])
        assert pd.notna(w['future_end_actual'])
