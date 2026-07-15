import re
import pandas as pd
import numpy as np

def normalize_scheme_name(name: str) -> dict:
    """
    Normalizes a scheme name, separating the AMC/structural prefixes and plan/option suffixes
    from the core underlying scheme name to enable robust cross-matching.
    """
    if not isinstance(name, str):
        return {"original": str(name), "normalized": "unknown", "version": "1.0"}
    
    orig_name = name.strip()
    n = orig_name.lower()

    # 1. Strip punctuation and multiple spaces
    n = re.sub(r'[\-\(\)\.,]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip()

    # 2. Extract structural tokens
    is_direct = "direct" in n
    is_regular = "regular" in n or "retail" in n
    is_growth = "growth" in n and not any(t in n for t in ("idcw", "dividend", "div ", "reinvestment", "payout"))
    is_idcw = any(t in n for t in ("idcw", "dividend", "div ", "reinvestment", "payout"))

    # 3. Strip structural/class suffixes to get the core name
    remove_tokens = [
        "direct", "regular", "retail", "institutional", "inst", "plan", "option", 
        "growth", "idcw", "dividend", "div", "reinvestment", "payout", "bonus"
    ]
    
    words = n.split()
    core_words = [w for w in words if w not in remove_tokens]
    
    # 4. Remove common AMC prefixes
    amc_prefixes = [
        "aditya birla sun life", "sbi", "icici prudential", "icici pru", "hdfc", "nippon india", "nippon",
        "tata", "dsp", "kotak", "axis", "uti", "mirae asset", "mirae", "bandhan", "idfc", "motilal oswal",
        "motilal", "franklin templeton", "franklin", "invesco", "sundaram", "lic mutual fund", "lic mf"
    ]
    core_str = " ".join(core_words)
    
    for amc in amc_prefixes:
        if core_str.startswith(amc + " "):
            core_str = core_str[len(amc)+1:].strip()
            break
            
    if not core_str:
        core_str = " ".join(core_words)
        
    return {
        "original_scheme_name": orig_name,
        "normalized_underlying_name": core_str,
        "normalization_rule_version": "1.0",
        "is_direct": is_direct,
        "is_regular": is_regular,
        "is_growth": is_growth,
        "is_idcw": is_idcw
    }

def classify_plan_strict(norm_info: dict) -> str:
    """Strictly assigns plan classification."""
    is_d = norm_info["is_direct"]
    is_r = norm_info["is_regular"]
    is_g = norm_info["is_growth"]
    is_i = norm_info["is_idcw"]

    if (is_d and is_r) or (is_g and is_i):
        return "Ambiguous"

    if is_i:
        if is_d and not is_r: return "Direct IDCW"
        if is_r and not is_d: return "Regular IDCW"
        return "Other"
    if is_g:
        if is_d and not is_r: return "Direct Growth"
        if is_r and not is_d: return "Regular Growth"
        return "Ambiguous"
    
    return "Other"

def build_normalized_universe(clean_df: pd.DataFrame) -> pd.DataFrame:
    schemes = clean_df.drop_duplicates("scheme_code")[
        ["scheme_code", "scheme_name"] + 
        (["isin_growth"] if "isin_growth" in clean_df.columns else [])
    ].copy()
    
    norm_results = []
    for _, row in schemes.iterrows():
        ninfo = normalize_scheme_name(row["scheme_name"])
        ninfo["scheme_code"] = row["scheme_code"]
        ninfo["plan_classification"] = classify_plan_strict(ninfo)
        norm_results.append(ninfo)
        
    norm_df = pd.DataFrame(norm_results)
    return schemes.merge(norm_df, on="scheme_code")

def manual_classification_sample(norm_df: pd.DataFrame, seed: int = 42, sample_size: int = 50) -> pd.DataFrame:
    np.random.seed(seed)
    sampled = []
    
    for cls, grp in norm_df.groupby("plan_classification"):
        n = min(len(grp), sample_size)
        samp = grp.sample(n=n, random_state=seed)
        sampled.append(samp)
        
    if sampled:
        res = pd.concat(sampled, ignore_index=True)
        res["manual_classification"] = ""
        res["reviewer_reason"] = ""
        res["audit_timestamp"] = ""
        return res
    return pd.DataFrame()

def find_direct_regular_pairs(norm_df: pd.DataFrame) -> dict:
    dg = norm_df[norm_df["plan_classification"] == "Direct Growth"].copy()
    rg = norm_df[norm_df["plan_classification"] == "Regular Growth"].copy()
    
    mappings = []
    dg_map = dg.groupby("normalized_underlying_name")["scheme_code"].apply(list).to_dict()
    rg_map = rg.groupby("normalized_underlying_name")["scheme_code"].apply(list).to_dict()
    
    for norm_name, d_codes in dg_map.items():
        r_codes = rg_map.get(norm_name, [])
        
        if len(d_codes) == 1 and len(r_codes) == 1:
            mappings.append({
                "normalized_underlying_name": norm_name,
                "direct_scheme_code": d_codes[0],
                "regular_scheme_code": r_codes[0],
                "mapping_status": "high_confidence_candidate"
            })
        elif len(d_codes) > 0 and len(r_codes) > 0:
            mappings.append({
                "normalized_underlying_name": norm_name,
                "direct_scheme_codes": str(d_codes),
                "regular_scheme_codes": str(r_codes),
                "mapping_status": "ambiguous"
            })
        else:
            mappings.append({
                "normalized_underlying_name": norm_name,
                "mapping_status": "rejected"
            })
            
    return pd.DataFrame(mappings)
