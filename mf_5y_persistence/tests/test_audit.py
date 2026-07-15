import pytest
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.identity_audit import normalize_scheme_name, classify_plan_strict, find_direct_regular_pairs

def test_normalization_preserves_meaning():
    res1 = normalize_scheme_name("HDFC Flexi Cap Fund - Direct Plan - Growth Option")
    assert res1["normalized_underlying_name"] == "flexi cap fund"
    assert res1["is_direct"] == True
    assert res1["is_growth"] == True
    
    res2 = normalize_scheme_name("SBI Small Cap Fund Regular Growth")
    assert res2["normalized_underlying_name"] == "small cap fund"
    assert res2["is_direct"] == False
    assert res2["is_regular"] == True
    assert res2["is_growth"] == True

def test_plan_classification():
    assert classify_plan_strict({"is_direct": True, "is_regular": False, "is_growth": True, "is_idcw": False}) == "Direct Growth"
    assert classify_plan_strict({"is_direct": False, "is_regular": True, "is_growth": True, "is_idcw": False}) == "Regular Growth"
    assert classify_plan_strict({"is_direct": True, "is_regular": False, "is_growth": False, "is_idcw": True}) == "Direct IDCW"
    assert classify_plan_strict({"is_direct": True, "is_regular": True, "is_growth": True, "is_idcw": False}) == "Ambiguous"
    assert classify_plan_strict({"is_direct": True, "is_regular": False, "is_growth": True, "is_idcw": True}) == "Ambiguous"

def test_direct_regular_mapping():
    norm_df = pd.DataFrame([
        {"scheme_code": 1, "plan_classification": "Direct Growth", "normalized_underlying_name": "flexi cap fund"},
        {"scheme_code": 2, "plan_classification": "Regular Growth", "normalized_underlying_name": "flexi cap fund"},
        {"scheme_code": 3, "plan_classification": "Direct Growth", "normalized_underlying_name": "bluechip fund"},
        {"scheme_code": 4, "plan_classification": "Direct Growth", "normalized_underlying_name": "bluechip fund"},
        {"scheme_code": 5, "plan_classification": "Regular Growth", "normalized_underlying_name": "bluechip fund"},
    ])
    pairs = find_direct_regular_pairs(norm_df)
    assert len(pairs) == 2
    fc = pairs[pairs["normalized_underlying_name"] == "flexi cap fund"].iloc[0]
    assert fc["mapping_status"] == "high_confidence_candidate"
    bc = pairs[pairs["normalized_underlying_name"] == "bluechip fund"].iloc[0]
    assert bc["mapping_status"] == "ambiguous"

def test_primary_metrics_and_unverified_records():
    rep_dir = Path(__file__).parent.parent / "reports" / "tables"
    if not rep_dir.exists():
        pytest.skip("Reports directory not found")
        
    surv = pd.read_csv(rep_dir / "survivorship_sensitivity.csv")
    cc = surv[surv["treatment"] == "complete_case"].iloc[0]
    assert cc["complete_cases"] == 142
    assert cc["successes"] == 84
    assert cc["success_rate"] == 0.5915
    
    adv = surv[surv["treatment"] == "adverse_case_bound"].iloc[0]
    assert adv["unresolved_cases"] == 27
    assert adv["success_rate"] == 0.4970
    
    unver = pd.read_csv(rep_dir / "unverified_records_audit.csv")
    assert all(unver["analytical_eligibility"] == False)
    assert all(unver["exclusion_reason"] == "unverified_NAV_date")
    assert all(unver["source_record_preserved"] == True)

def test_cohorts_and_loco():
    rep_dir = Path(__file__).parent.parent / "reports" / "tables"
    if not rep_dir.exists():
        pytest.skip("Reports directory not found")
        
    cohorts = pd.read_csv(rep_dir / "annual_cohort_results.csv")
    assert cohorts["success_rate"].min() == 0.2759
    assert cohorts["success_rate"].max() == 1.0
    
    loco = pd.read_csv(rep_dir / "leave_one_cohort_out.csv")
    assert round(loco["remaining_success_rate"].min(), 4) == 0.4747
    assert round(loco["remaining_success_rate"].max(), 4) == 0.6726

def test_manifest_and_gate():
    rep_dir = Path(__file__).parent.parent / "reports" / "tables"
    if not (rep_dir / "audit_run_manifest.json").exists():
        pytest.skip("Manifest not found")
        
    with open(rep_dir / "audit_run_manifest.json") as f:
        manifest = json.load(f)
        
    assert manifest["pipeline_status"] == "empirical_analysis_only"
    assert manifest["training_executed"] == False
    assert manifest["independent_market_blocks"] == 1
    assert "combined_dataset_SHA256" in manifest
    assert len(manifest["combined_dataset_SHA256"]) == 64
