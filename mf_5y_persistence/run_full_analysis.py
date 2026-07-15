"""
mf_5y_persistence/run_full_analysis.py
Complete pipeline: download audit, reconcile, classify, windows, base rates, gate.
"""
import os, sys, re, hashlib, time, io, math, logging
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from pathlib import Path

PROJECT = Path(__file__).parent
RAW_DIR  = PROJECT / "data" / "raw" / "amfi"
PROC_DIR = PROJECT / "data" / "processed"
REP_DIR  = PROJECT / "reports" / "tables"
for d in (PROC_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
AMFI_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"

# ── helpers ──────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()

def parse_amfi_text(text, source_file=""):
    lines = [l for l in text.split("\n") if ";" in l and l.strip()]
    if not lines:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(io.StringIO("\n".join(lines)), sep=";",
                          low_memory=False, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()
    raw.columns = [c.strip() for c in raw.columns]
    rename = {"Scheme Code":"scheme_code","Scheme Name":"scheme_name",
               "Net Asset Value":"nav","Date":"nav_date"}
    for isin_col in ("ISIN Div Payout/ISIN Growth","ISIN Div Payout/ ISIN Growth","ISIN Growth"):
        if isin_col in raw.columns:
            rename[isin_col] = "isin_growth"; break
    raw = raw.rename(columns=rename)
    need = ["scheme_code","scheme_name","nav","nav_date"]
    if not all(c in raw.columns for c in need):
        return pd.DataFrame()
    df = raw[[c for c in raw.columns if c in need+["isin_growth"]]].copy()
    df["scheme_code"] = pd.to_numeric(df["scheme_code"], errors="coerce")
    df["nav"]         = pd.to_numeric(df["nav"], errors="coerce")
    df["nav_date"]    = pd.to_datetime(df["nav_date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["scheme_code","nav","nav_date"])
    df["scheme_code"] = df["scheme_code"].astype(int)
    df["source_file"] = source_file
    return df

def canonical_chunks(start, end):
    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=89), end)
        chunks.append((cur, chunk_end)); cur = chunk_end + timedelta(days=1)
    return chunks

def fetch_chunk(s, e):
    chunk_dir = RAW_DIR / str(s.year)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    fp = chunk_dir / f"{s.isoformat()}_to_{e.isoformat()}.txt"
    meta = dict(request_start_date=s.isoformat(), request_end_date=e.isoformat(),
                attempt_count=0, HTTP_status=None, content_type=None,
                download_status="pending", raw_file_path=str(fp),
                SHA256_checksum=None, raw_row_count=0, parsed_row_count=0,
                minimum_parsed_date=None, maximum_parsed_date=None, failure_reason=None)
    if fp.exists():
        meta["download_status"] = "cached"; meta["HTTP_status"] = 200
        content_bytes = fp.read_bytes()
    else:
        frmdt = s.strftime("%d-%b-%Y"); todt = e.strftime("%d-%b-%Y")
        url = f"{AMFI_URL}?frmdt={frmdt}&todt={todt}"
        for attempt in range(3):
            meta["attempt_count"] += 1
            try:
                resp = requests.get(url, timeout=120)
                meta["HTTP_status"] = resp.status_code
                meta["content_type"] = resp.headers.get("Content-Type","")
                if resp.status_code != 200:
                    meta["failure_reason"] = f"HTTP {resp.status_code}"; return meta
                content = resp.text
                if "<html" in content.lower() or "<!doctype" in content.lower():
                    meta["failure_reason"] = "HTML response"; return meta
                if len(content.strip()) < 50:
                    meta["failure_reason"] = "Empty response"; return meta
                content_bytes = content.encode("utf-8")
                fp.write_bytes(content_bytes)
                meta["download_status"] = "downloaded"; time.sleep(1.2); break
            except Exception as exc:
                meta["failure_reason"] = str(exc)
                if attempt < 2: time.sleep(5)
                else: return meta
    content_bytes = fp.read_bytes()
    text = content_bytes.decode("utf-8", errors="replace")
    meta["SHA256_checksum"] = hashlib.sha256(content_bytes).hexdigest()
    meta["raw_row_count"] = text.count("\n")
    df = parse_amfi_text(text, str(fp))
    meta["parsed_row_count"] = len(df)
    if not df.empty:
        min_d = df["nav_date"].min().date(); max_d = df["nav_date"].max().date()
        meta["minimum_parsed_date"] = min_d.isoformat()
        meta["maximum_parsed_date"] = max_d.isoformat()
        if min_d < s - timedelta(days=10) or max_d > e + timedelta(days=10):
            meta["failure_reason"] = "Dates outside requested range"
            meta["download_status"] = "range_error"
    else:
        meta["failure_reason"] = meta["failure_reason"] or "No parseable NAV rows"
        meta["download_status"] = "parse_failure"
    return meta

# ── download ──────────────────────────────────────────────────────────────────

def download_and_audit():
    start = date(2013,1,1); end = date.today() - timedelta(days=1)
    chunks = canonical_chunks(start, end)
    log.info(f"Canonical chunks {start}→{end}: {len(chunks)}")
    existing = {}
    for fp in sorted(RAW_DIR.rglob("*.txt")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.txt", fp.name)
        if m: existing[(date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2)))] = fp
    manifests, dfs = [], []
    for (s, e) in chunks:
        cached_fp = existing.get((s, e))
        if cached_fp is None:
            for (cs,ce),fp in existing.items():
                if cs == s: cached_fp = fp; break
        if cached_fp and cached_fp.exists():
            content_bytes = cached_fp.read_bytes()
            text = content_bytes.decode("utf-8", errors="replace")
            df = parse_amfi_text(text, str(cached_fp))
            m_dict = dict(request_start_date=s.isoformat(), request_end_date=e.isoformat(),
                          attempt_count=0, HTTP_status=200, content_type="text/plain",
                          download_status="cached", raw_file_path=str(cached_fp),
                          SHA256_checksum=hashlib.sha256(content_bytes).hexdigest(),
                          raw_row_count=text.count("\n"), parsed_row_count=len(df),
                          minimum_parsed_date=df["nav_date"].min().date().isoformat() if not df.empty else None,
                          maximum_parsed_date=df["nav_date"].max().date().isoformat() if not df.empty else None,
                          failure_reason=None if not df.empty else "parse_failure")
            manifests.append(m_dict)
            if not df.empty: dfs.append(df)
        else:
            log.info(f"  Downloading: {s} → {e}")
            m_dict = fetch_chunk(s, e)
            manifests.append(m_dict)
            if m_dict.get("parsed_row_count", 0) > 0:
                fp = Path(m_dict["raw_file_path"])
                if fp.exists():
                    text = fp.read_bytes().decode("utf-8", errors="replace")
                    df = parse_amfi_text(text, str(fp))
                    if not df.empty: dfs.append(df)
    manifest_df = pd.DataFrame(manifests)
    failures_df = manifest_df[manifest_df["failure_reason"].notna()].copy()
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    manifest_df.to_csv(REP_DIR / "raw_file_manifest.csv", index=False)
    failures_df.to_csv(REP_DIR / "download_failures.csv", index=False)
    log.info(f"Manifest: {len(manifest_df)} chunks, {len(failures_df)} failures, {len(combined_df):,} NAV rows")
    return manifest_df, failures_df, combined_df

# ── reconcile ─────────────────────────────────────────────────────────────────

CLEAN_CACHE = PROC_DIR / "full_nav_history.parquet"

def reconcile(combined):
    exact_dups = combined[combined.duplicated(subset=["scheme_code","nav_date","nav"], keep=False)].copy()
    exact_dups.to_csv(REP_DIR / "duplicate_rows.csv", index=False)
    deduped = combined.drop_duplicates(subset=["scheme_code","nav_date","nav"]).copy()
    nav_counts = deduped.groupby(["scheme_code","nav_date"])["nav"].nunique()
    conflicts = nav_counts[nav_counts > 1]
    if len(conflicts) > 0:
        ck = conflicts.reset_index()[["scheme_code","nav_date"]]
        conflict_df = deduped.merge(ck, on=["scheme_code","nav_date"])
        conflict_df.to_csv(REP_DIR / "conflicting_nav_rows.csv", index=False)
        log.warning(f"{len(conflicts)} scheme-date pairs have conflicting NAVs")
        clean = deduped[~deduped.set_index(["scheme_code","nav_date"]).index.isin(conflicts.index)].copy()
    else:
        pd.DataFrame().to_csv(REP_DIR / "conflicting_nav_rows.csv", index=False)
        conflict_df = pd.DataFrame(); clean = deduped.copy()
    audit = {"total_raw_rows":len(combined),"exact_duplicate_rows":len(exact_dups),
             "conflicting_rows":len(conflict_df),"clean_rows":len(clean),
             "unique_schemes":clean["scheme_code"].nunique(),
             "date_min":clean["nav_date"].min().date().isoformat() if len(clean) else "",
             "date_max":clean["nav_date"].max().date().isoformat() if len(clean) else ""}
    pd.DataFrame([audit]).to_csv(REP_DIR / "combined_data_audit.csv", index=False)
    # Cache clean data so restarts skip the 30M-row parse
    clean.to_parquet(CLEAN_CACHE, index=False)
    log.info(f"Clean NAV cached → {CLEAN_CACHE}")
    return clean, exact_dups, conflict_df

# ── max date audit ────────────────────────────────────────────────────────────

def audit_max_date(clean, combined):
    TARGET = pd.Timestamp("2026-07-12")
    fresh_rows = clean[clean["nav_date"] == TARGET].copy()
    fresh_idx = set(zip(fresh_rows["scheme_code"], fresh_rows["nav"]))
    records = []
    legacy_path = PROJECT / "data" / "raw" / "nav_history.parquet"
    if legacy_path.exists():
        legacy = pd.read_parquet(legacy_path)
        legacy.columns = [c.strip() for c in legacy.columns]
        date_col = next((c for c in ("date","nav_date") if c in legacy.columns), None)
        if date_col:
            legacy[date_col] = pd.to_datetime(legacy[date_col], errors="coerce")
            legacy_max = legacy[legacy[date_col] == TARGET].copy()
            for _, row in legacy_max.iterrows():
                sc = int(row.get("scheme_code",0)); nav = float(row.get("nav", float("nan")))
                fresh_scheme = fresh_rows[fresh_rows["scheme_code"] == sc]
                if (sc, nav) in fresh_idx:                 status = "verified_exact_match"
                elif len(fresh_scheme) > 0:                status = "verified_date_NAV_conflict"
                elif fresh_rows.empty:                     status = "unable_to_verify"
                else:                                      status = "not_present_in_download"
                records.append({"scheme_code":sc,"scheme_name":row.get("scheme_name",""),
                                 "legacy_nav":nav,"legacy_nav_date":"2026-07-12",
                                 "download_nav":float(fresh_scheme["nav"].values[0]) if len(fresh_scheme)>0 else None,
                                 "verification_status":status})
    else:
        for _, row in fresh_rows.iterrows():
            records.append({"scheme_code":int(row["scheme_code"]),"scheme_name":row.get("scheme_name",""),
                             "legacy_nav":None,"legacy_nav_date":"2026-07-12",
                             "download_nav":float(row["nav"]),"verification_status":"unable_to_verify"})
    audit_df = pd.DataFrame(records)
    audit_df.to_csv(REP_DIR / "max_date_audit.csv", index=False)
    log.info(f"Max-date audit: {len(audit_df)} rows on 2026-07-12")
    return audit_df

# ── plan classification ───────────────────────────────────────────────────────

def classify_plan(name, isin=""):
    if not isinstance(name, str): return "Ambiguous"
    n = name.lower()
    is_idcw    = any(t in n for t in ("idcw","dividend","div ","reinvestment","payout"))
    is_direct  = "direct" in n
    is_regular = any(t in n for t in ("regular","retail","advisor"))
    is_growth  = "growth" in n and not is_idcw
    if is_idcw:
        if is_direct: return "Direct IDCW"
        if is_regular: return "Regular IDCW"
        return "Other"
    if is_growth:
        if is_direct and not is_regular: return "Direct Growth"
        if is_regular and not is_direct: return "Regular Growth"
        if is_direct and is_regular: return "Ambiguous"
        return "Ambiguous"
    return "Ambiguous" if (is_direct or is_regular) else "Other"

def classify_schemes(clean):
    schemes = clean.drop_duplicates("scheme_code")[
        ["scheme_code","scheme_name"]+
        (["isin_growth"] if "isin_growth" in clean.columns else [])
    ].copy()
    isin_col = "isin_growth" if "isin_growth" in schemes.columns else None
    schemes["plan_class"] = schemes.apply(
        lambda r: classify_plan(r["scheme_name"], r[isin_col] if isin_col else ""), axis=1)
    return schemes

# ── scheme identity ───────────────────────────────────────────────────────────

def scheme_identity_audit(clean):
    latest = (clean.sort_values("nav_date").groupby("scheme_code")["scheme_name"]
                   .last().reset_index().rename(columns={"scheme_name":"latest_name"}))
    all_names = (clean.groupby("scheme_code")["scheme_name"]
                      .apply(lambda x: list(x.unique())).reset_index()
                      .rename(columns={"scheme_name":"all_names"}))
    all_names["name_change_count"] = all_names["all_names"].apply(lambda x: len(x)-1)
    name_changes = all_names[all_names["name_change_count"] > 0].copy()
    name_changes["all_names"] = name_changes["all_names"].apply(str)
    name_changes.to_csv(REP_DIR / "scheme_name_changes.csv", index=False)
    # Use 60-char prefix; exclude category-header rows (short names without AMC-style tokens)
    latest["name_prefix"] = latest["latest_name"].str[:60].str.lower().str.strip()
    # Only flag as duplicate-candidate when prefix is at least 20 chars (avoids degenerate matches)
    dup_candidates = (latest[latest["name_prefix"].str.len() >= 20]
                          .groupby("name_prefix").filter(lambda g: len(g) > 1))
    dup_candidates.to_csv(REP_DIR / "possible_scheme_duplicates.csv", index=False)
    last_nav = (clean.groupby("scheme_code")["nav_date"].max().reset_index()
                     .rename(columns={"nav_date":"last_nav_date"}))
    cutoff = pd.Timestamp(date.today() - timedelta(days=3*365))
    discontinued = last_nav[last_nav["last_nav_date"] < cutoff].copy()
    discontinued.to_csv(REP_DIR / "unresolved_discontinuations.csv", index=False)
    identity_audit = latest.merge(last_nav, on="scheme_code")
    identity_audit.to_csv(REP_DIR / "scheme_identity_audit.csv", index=False)
    schemes = classify_schemes(clean)
    unresolved = schemes[schemes["plan_class"] == "Ambiguous"].copy()
    unresolved.to_csv(REP_DIR / "unresolved_scheme_mappings.csv", index=False)
    dc = set(schemes[schemes["plan_class"]=="Direct Growth"]["scheme_code"])
    rc = set(schemes[schemes["plan_class"]=="Regular Growth"]["scheme_code"])
    overlap_df = schemes[schemes["scheme_code"].isin(dc & rc)].copy()
    overlap_df.to_csv(REP_DIR / "direct_regular_overlap.csv", index=False)
    return dict(identity_audit=identity_audit, name_changes=name_changes,
                dup_candidates=dup_candidates, discontinued=discontinued,
                unresolved=unresolved, overlap=overlap_df)

# ── five-year windows ─────────────────────────────────────────────────────────

TOL = timedelta(days=10)

def find_nav_near(nav_s, target):
    tgt = pd.Timestamp(target)
    window = nav_s[(nav_s.index >= tgt - pd.Timedelta(TOL)) & (nav_s.index <= tgt + pd.Timedelta(TOL))]
    if window.empty: return None, None
    diffs = pd.Series(window.index - tgt, index=window.index).abs()
    idx = diffs.idxmin()
    return float(window[idx]), idx.date()

def cagr(s, e, yrs):
    if s <= 0 or e <= 0 or yrs < 0.5: return None
    return (pow(e/s, 1.0/yrs) - 1.0) * 100.0

def infer_category(name):
    n = name.lower()
    if any(t in n for t in ("elss","equity","flexi cap","large cap","mid cap","small cap","multi cap")): return "Equity"
    if any(t in n for t in ("debt","bond","credit","gilt","duration","income","banking and psu")): return "Debt"
    if any(t in n for t in ("liquid","overnight","money market","ultra short")): return "Liquid/Money Market"
    if any(t in n for t in ("hybrid","balanced","aggressive","conservative","multi asset")): return "Hybrid"
    if any(t in n for t in ("arbitrage",)): return "Arbitrage"
    if any(t in n for t in ("fund of fund","fof","international","overseas","global")): return "FoF/International"
    return "Other"

def build_5y_windows(clean, schemes, obs_month=3):
    dg_codes = set(schemes[schemes["plan_class"]=="Direct Growth"]["scheme_code"])
    clean_dg = clean[clean["scheme_code"].isin(dg_codes)].copy().sort_values(["scheme_code","nav_date"])
    records = []
    for sc, grp in clean_dg.groupby("scheme_code"):
        nav_s = grp.set_index("nav_date")["nav"]
        sname = grp["scheme_name"].iloc[-1]
        cat   = infer_category(sname)
        first_d = nav_s.index.min().date(); last_d = nav_s.index.max().date()
        for yr in range(2018, date.today().year + 1):
            m_last = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
            try: pred_int = date(yr, obs_month, m_last[obs_month])
            except: pred_int = date(yr, obs_month, 28)
            try: past_int = date(yr-5, obs_month, m_last[obs_month])
            except: past_int = date(yr-5, obs_month, 28)
            try: fut_int = date(yr+5, obs_month, m_last[obs_month])
            except: fut_int = date(yr+5, obs_month, 28)
            if fut_int > last_d or past_int < first_d: continue
            past_nav, past_act = find_nav_near(nav_s, past_int)
            pred_nav, pred_act = find_nav_near(nav_s, pred_int)
            fut_nav,  fut_act  = find_nav_near(nav_s, fut_int)
            if any(v is None for v in [past_nav, pred_nav, fut_nav]):
                status = "missing_nav_endpoint"; pc = fc = None
            else:
                py = (pred_act - past_act).days / 365.25
                fy = (fut_act  - pred_act).days / 365.25
                pc = cagr(past_nav, pred_nav, py); fc = cagr(pred_nav, fut_nav, fy)
                status = "complete" if (pc is not None and fc is not None) else "invalid_cagr"
            records.append({"scheme_code":sc,"scheme_name":sname,"category":cat,
                             "prediction_date":pred_int.isoformat(),
                             "past_start_intended":past_int.isoformat(),
                             "past_start_actual":past_act.isoformat() if past_act else None,
                             "prediction_NAV_date":pred_act.isoformat() if pred_act else None,
                             "future_end_intended":fut_int.isoformat(),
                             "future_end_actual":fut_act.isoformat() if fut_act else None,
                             "past_5y_CAGR":round(pc,4) if pc is not None else None,
                             "future_5y_CAGR":round(fc,4) if fc is not None else None,
                             "window_status":status})
    win_df = pd.DataFrame(records)
    win_df.to_csv(REP_DIR / "five_year_window_audit.csv", index=False)
    log.info(f"Windows: {len(win_df)} total, {(win_df['window_status']=='complete').sum()} complete")
    return win_df

# ── base rates ────────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0.0, centre-margin), min(1.0, centre+margin))

def base_rate(win_df, lo, hi, thr=12.0):
    sub = win_df[(win_df["window_status"]=="complete") &
                 (win_df["past_5y_CAGR"] >= lo) & (win_df["past_5y_CAGR"] < hi)].copy()
    n = len(sub); k = int((sub["future_5y_CAGR"] >= thr).sum())
    wlo, whi = wilson_ci(k, n)
    pcts = sub["future_5y_CAGR"].quantile([.10,.25,.50,.75,.90]).to_dict() if n > 0 else {}
    return {"past_cagr_band":f"{lo}%-{hi}%","case_count":n,"success_count":k,
            "success_probability":round(k/n,4) if n>0 else None,
            "wilson_ci_lo":round(wlo,4),"wilson_ci_hi":round(whi,4),
            "median_future_cagr":round(pcts.get(.50,float("nan")),4) if n>0 else None,
            "p10_future_cagr":round(pcts.get(.10,float("nan")),4) if n>0 else None,
            "p25_future_cagr":round(pcts.get(.25,float("nan")),4) if n>0 else None,
            "p75_future_cagr":round(pcts.get(.75,float("nan")),4) if n>0 else None,
            "p90_future_cagr":round(pcts.get(.90,float("nan")),4) if n>0 else None}

def compute_base_rates(win_df):
    rows = [base_rate(win_df, lo, hi) for (lo,hi) in [(11,13),(10,14),(8,16)]]
    df = pd.DataFrame(rows)
    df.to_csv(REP_DIR / "base_rate_by_cagr_band.csv", index=False)
    return df

# ── subgroups ─────────────────────────────────────────────────────────────────

def subgroup_results(win_df):
    complete = win_df[win_df["window_status"]=="complete"].copy()
    MIN_N = 30; rows = []
    for cat, grp in complete.groupby("category"):
        n=len(grp); k=int((grp["future_5y_CAGR"]>=12).sum())
        lo,hi=wilson_ci(k,n)
        rows.append({"subgroup_type":"category","subgroup_value":cat,"case_count":n,
                     "success_count":k,"success_probability":round(k/n,4) if n>0 else None,
                     "wilson_ci_lo":round(lo,4),"wilson_ci_hi":round(hi,4),"adequate_sample":n>=MIN_N})
    complete["cohort_year"] = pd.to_datetime(complete["prediction_date"]).dt.year
    for yr, grp in complete.groupby("cohort_year"):
        n=len(grp); k=int((grp["future_5y_CAGR"]>=12).sum())
        lo,hi=wilson_ci(k,n)
        rows.append({"subgroup_type":"cohort_year","subgroup_value":str(yr),"case_count":n,
                     "success_count":k,"success_probability":round(k/n,4) if n>0 else None,
                     "wilson_ci_lo":round(lo,4),"wilson_ci_hi":round(hi,4),"adequate_sample":n>=MIN_N})
    if len(complete) >= 4:
        complete["past_cagr_q"] = pd.qcut(complete["past_5y_CAGR"],4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
        for q, grp in complete.groupby("past_cagr_q", observed=True):
            n=len(grp); k=int((grp["future_5y_CAGR"]>=12).sum())
            lo,hi=wilson_ci(k,n)
            rows.append({"subgroup_type":"past_cagr_quartile","subgroup_value":str(q),"case_count":n,
                         "success_count":k,"success_probability":round(k/n,4) if n>0 else None,
                         "wilson_ci_lo":round(lo,4),"wilson_ci_hi":round(hi,4),"adequate_sample":n>=MIN_N})
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(REP_DIR / "category_results.csv", index=False)
    return sub_df

# ── training gate ─────────────────────────────────────────────────────────────

def evaluate_gate(win_df):
    comp = win_df[win_df["window_status"]=="complete"]
    n=len(comp); n_pos=int((comp["future_5y_CAGR"]>=12).sum()); n_neg=int((comp["future_5y_CAGR"]<12).sum())
    cohorts=comp["prediction_date"].nunique()
    cats = comp.groupby("category").size(); cat_ok=int((cats>=30).sum())
    gate = {"history_gate":"pass" if not win_df.empty else "fail",
            "complete_rows_gate":f"pass" if n>=500 else f"fail ({n}<500)",
            "cohort_count_gate":f"pass" if cohorts>=5 else f"fail ({cohorts}<5)",
            "positive_target_gate":f"pass" if n_pos>=50 else f"fail ({n_pos}<50)",
            "negative_target_gate":f"pass" if n_neg>=50 else f"fail ({n_neg}<50)",
            "category_sample_gate":f"pass" if cat_ok>=2 else f"fail ({cat_ok} cats ≥30 obs)"}
    all_pass = all(v=="pass" for v in gate.values())
    gate["pipeline_status"] = "empirical_analysis_only" if (n>=30 and n_pos>=5) else "insufficient_effective_sample"
    gate["training_executed"] = False
    return gate

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("="*70)
    log.info("mf_5y_persistence — full analysis pipeline")
    log.info("="*70)

    # 1. Download audit + resume
    manifest, failures, combined = download_and_audit()
    n_total  = len(manifest); n_failed = int(manifest["failure_reason"].notna().sum())
    n_cached = int((manifest["download_status"]=="cached").sum())
    n_parsed = int(manifest["parsed_row_count"].sum())
    min_date = manifest["minimum_parsed_date"].dropna().replace("",pd.NA).dropna().min()
    max_date = manifest["maximum_parsed_date"].dropna().replace("",pd.NA).dropna().max()
    total_raw = int(manifest["raw_row_count"].sum())
    total_bytes = manifest["raw_file_path"].apply(lambda p: Path(p).stat().st_size if Path(p).exists() else 0).sum()

    print("\n── DOWNLOAD STATE ──────────────────────────────────────────────────")
    print(f"total chunks planned:               {n_total}")
    print(f"chunks completed (cached+downloaded): {n_total - n_failed}")
    print(f"chunks failed:                      {n_failed}")
    print(f"chunks pending:                     0")
    print(f"chunks with parse failures:         {int(manifest['failure_reason'].str.contains('parse', na=False).sum())}")
    print(f"earliest successfully downloaded date: {min_date}")
    print(f"latest successfully downloaded date:   {max_date}")
    print(f"total raw files:                    {n_total - n_failed}")
    print(f"total raw bytes:                    {int(total_bytes):,}")
    print(f"total raw lines:                    {total_raw:,}")
    print(f"total parsed NAV rows:              {n_parsed:,}")
    if len(failures):
        print("\nFailed chunks:")
        print(failures[["request_start_date","request_end_date","failure_reason"]].to_string(index=False))

    if combined.empty:
        print("ERROR: No NAV data parsed. Aborting."); sys.exit(1)

    # 2. Reconcile
    clean, dups, conflicts = reconcile(combined)
    print(f"\n── RECONCILIATION ──────────────────────────────────────────────────")
    print(f"raw rows:             {len(combined):,}")
    print(f"exact duplicate rows: {len(dups):,}")
    print(f"conflicting NAV rows: {len(conflicts):,}")
    print(f"clean rows:           {len(clean):,}")
    print(f"unique schemes:       {clean['scheme_code'].nunique():,}")
    print(f"date_min:             {clean['nav_date'].min().date()}")
    print(f"date_max:             {clean['nav_date'].max().date()}")

    # 3. Max-date audit
    max_audit = audit_max_date(clean, combined)
    vc = max_audit["verification_status"].value_counts() if len(max_audit) else pd.Series()
    print(f"\n── MAX-DATE AUDIT (2026-07-12) ─────────────────────────────────────")
    if len(max_audit):
        for k,v in vc.items(): print(f"  {k}: {v}")
    else:
        print("  No rows dated 2026-07-12 found in fresh download.")

    # Exclude unverified legacy rows from model dataset
    if not max_audit.empty:
        bad_codes = set(max_audit[max_audit["verification_status"].isin(
            ["not_present_in_download","unable_to_verify","verified_date_NAV_conflict"]
        )]["scheme_code"].astype(int))
        clean_model = clean[~(
            (clean["nav_date"]==pd.Timestamp("2026-07-12")) &
            (clean["scheme_code"].isin(bad_codes))
        )].copy()
    else:
        clean_model = clean.copy()

    # 4. Plan classification
    schemes = classify_schemes(clean_model)
    pc = schemes["plan_class"].value_counts()
    print(f"\n── PLAN CLASSIFICATION ─────────────────────────────────────────────")
    for k,v in pc.items(): print(f"  {k}: {v}")
    schemes[schemes["plan_class"]=="Ambiguous"].to_csv(
        REP_DIR / "ambiguous_scheme_classifications.csv", index=False)

    # 5. Scheme identity
    identity = scheme_identity_audit(clean_model)
    print(f"\n── SCHEME IDENTITY ─────────────────────────────────────────────────")
    print(f"  schemes with name changes:       {len(identity['name_changes'])}")
    print(f"  possible duplicates:             {len(identity['dup_candidates'])}")
    print(f"  discontinued schemes:            {len(identity['discontinued'])}")
    print(f"  unresolved / ambiguous:          {len(identity['unresolved'])}")
    print(f"  direct-regular overlaps:         {len(identity['overlap'])}")

    # 6. Five-year windows
    win_df = build_5y_windows(clean_model, schemes)
    complete_wins = win_df[win_df["window_status"]=="complete"]
    n_complete = len(complete_wins)

    funds_5y = int((clean_model.groupby("scheme_code")["nav_date"]
                     .apply(lambda x: (x.max()-x.min()).days/365.25 >= 5)).sum())
    funds_10y = int((clean_model.groupby("scheme_code")["nav_date"]
                      .apply(lambda x: (x.max()-x.min()).days/365.25 >= 10)).sum())

    print(f"\n── EFFECTIVE SAMPLE SIZE ───────────────────────────────────────────")
    print(f"  complete labelled windows:           {n_complete}")
    print(f"  unique funds in complete windows:    {complete_wins['scheme_code'].nunique()}")
    print(f"  unique annual prediction dates:      {complete_wins['prediction_date'].nunique()}")
    print(f"  funds with ≥5y history:              {funds_5y}")
    print(f"  funds with ≥10y complete transition: {funds_10y}")
    if n_complete:
        cats_n = complete_wins.groupby("category").size()
        print("  observations per category:")
        for cat,cnt in cats_n.items(): print(f"    {cat}: {cnt}")

    # 7. Base rates
    br_df = compute_base_rates(win_df)
    print(f"\n── PRIMARY BASE RATE (past CAGR band → future ≥12%) ──────────────────")
    print(br_df.to_string(index=False))

    # 8. Subgroups
    sub_df = subgroup_results(win_df)
    print(f"\n── SUBGROUP RESULTS ─────────────────────────────────────────────────")
    print(sub_df.to_string(index=False))

    # 9. Training gate
    gate = evaluate_gate(win_df)
    print(f"\n── ML TRAINING GATE ─────────────────────────────────────────────────")
    for k,v in gate.items(): print(f"  {k}: {v}")

    print(f"\n── OUTPUT FILES ─────────────────────────────────────────────────────")
    for fp in sorted(REP_DIR.glob("*.csv")): print(f"  {fp}")

if __name__ == "__main__":
    main()
