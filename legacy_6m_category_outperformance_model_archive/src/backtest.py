import pandas as pd
import numpy as np
import yaml
import json
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def block_bootstrap(df, metric_func, n_bootstraps=1000):
    # df must have 'date' column representing observation date
    dates = np.sort(df['date'].unique())
    # Block size of 6 months to account for overlapping targets
    block_size = 6
    if len(dates) < block_size * 2:
        print("WARNING: Not enough independent time blocks for robust bootstrap.")
        return None
        
    metrics = []
    for _ in range(n_bootstraps):
        # Sample starting indices for blocks
        # We sample enough blocks to roughly equal the original dataset size
        n_blocks = len(dates) // block_size + 1
        starts = np.random.choice(len(dates) - block_size + 1, size=n_blocks, replace=True)
        
        boot_dates = []
        for start in starts:
            boot_dates.extend(dates[start:start+block_size])
            
        boot_dates = boot_dates[:len(dates)] # Truncate to original length
        
        # Build bootstrap dataframe
        # To avoid duplicating dates exactly (which breaks groupbys if we just merge), 
        # it's better to just sample the data for those dates
        boot_df = pd.concat([df[df['date'] == d] for d in boot_dates])
        
        try:
            val = metric_func(boot_df)
            if not np.isnan(val):
                metrics.append(val)
        except:
            pass
            
    if not metrics:
        return None
        
    return {
        'mean': float(np.mean(metrics)),
        'ci_lower': float(np.percentile(metrics, 2.5)),
        'ci_upper': float(np.percentile(metrics, 97.5))
    }

def run_advanced_backtest():
    config = load_config()
    date_col = config['data']['date_column']
    
    print("Loading modeling dataset...")
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    df[date_col] = pd.to_datetime(df[date_col])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df = df.sort_values(date_col)
    
    # We also need the raw NAV end date to ensure target_end_date <= raw_nav_end_date
    navs = pd.read_parquet("data/raw/nav_history.parquet")
    navs[date_col] = pd.to_datetime(navs[date_col], format='mixed', errors='coerce')
    raw_nav_end_date = navs[date_col].max()
    print(f"Raw NAV End Date: {raw_nav_end_date.date()}")
    del navs # free memory
    
    features = [
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
        'mom_12_1', 'mom_6_1', 'vol_12m', 'vol_6m',
        'dist_ma_6m', 'dist_ma_12m',
        'rank_ret_6m', 'rank_ret_12m', 'rank_vol_12m', 'excess_ret_6m'
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)
    
    dates = np.sort(df[date_col].unique())
    start_idx = 24
    if len(dates) <= start_idx:
        print("Not enough data.")
        return
        
    step_size = 6
    test_starts = dates[start_idx::step_size]
    
    # Define explicitly evaluated dates for holdout per user request
    holdout_start = pd.to_datetime('2025-07-31')
    holdout_end_exclusive = pd.to_datetime('2026-01-01')
    
    results = {
        'folds': [],
        'audit_table': [],
        'holdout_turnover': {},
        'holdout_clustered_metrics': {},
        'probability_buckets': {},
        'calibration_curves': {}
    }
    
    holdout_predictions = []
    
    for test_start in test_starts:
        test_start = pd.to_datetime(test_start)
        test_end = test_start + pd.DateOffset(months=step_size)
        
        test_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)
        test_df_full = df[test_mask].copy()
        if len(test_df_full) == 0:
            continue
            
        # Is this fold fully within the new strictly defined holdout period?
        is_holdout = (test_start >= holdout_start) and (test_start < holdout_end_exclusive)
        
        # 1. EVALUATION INVARIANTS: Strictly evaluated rows only
        # Target must not be missing, forward_return_6m not missing, cat median not missing, target_end <= raw_nav_end
        eval_mask = (
            test_df_full['target'].notna() & 
            test_df_full['forward_ret_6m'].notna() &
            test_df_full['target_end_date'].le(raw_nav_end_date)
        )
        test_df = test_df_full[eval_mask].copy()
        
        if len(test_df) == 0:
            print(f"Skipping {test_start.date()} - No fully evaluated rows.")
            continue
            
        assert test_df['target'].notna().all(), "Invariant failed: NaNs in target"
        assert test_df['target'].isin([0, 1]).all(), "Invariant failed: Target not strictly 0 or 1"
        assert test_df['target_end_date'].le(raw_nav_end_date).all(), "Invariant failed: Target date leaks past raw data"
        assert test_df[date_col].lt(test_df['target_end_date']).all(), "Invariant failed: Obs date >= Target date"
        
        pos_rate = test_df['target'].mean()
        assert pos_rate > 0.40 and pos_rate < 0.60, f"Invariant failed: Target rate {pos_rate:.3f} outside bounds (0.4, 0.6)"
            
        # 2. CALIBRATION (ROLLING BASE MODEL)
        # To satisfy calibration_max_target_end < test_start:
        # Calibration observation must end 6 months before test_start.
        calib_end = test_start - pd.DateOffset(months=6)
        calib_start = calib_end - pd.DateOffset(months=6)
        calib_dates = np.sort(df[(df[date_col] >= calib_start) & (df[date_col] < calib_end)][date_col].unique())
        
        if len(calib_dates) == 0:
            print(f"Skipping {test_start.date()} - No calibration dates.")
            continue
            
        
        out_of_sample_probs = []
        out_of_sample_y = []
        
        for c in calib_dates:
            c = pd.to_datetime(c)
            # Train base model ONLY on rows where target_end_date < c
            # This ensures ABSOLUTELY NO overlap between training targets and the calibration observation date
            c_train_mask = (df['target_end_date'] < c) & df['target'].notna()
            c_train = df[c_train_mask]
            if len(c_train) < 500:
                continue
                
            c_hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
            c_hgb.fit(c_train[features], c_train['target'])
            
            cls_idx = list(c_hgb.classes_).index(1.0)
            
            c_test = df[df[date_col] == c].copy()
            c_test = c_test[c_test['target'].notna()]
            if len(c_test) > 0:
                out_of_sample_probs.extend(c_hgb.predict_proba(c_test[features])[:, cls_idx])
                out_of_sample_y.extend(c_test['target'].values)
                
        if len(out_of_sample_y) < 100:
            print(f"Skipping {test_start.date()} - Not enough calibration samples.")
            continue
            
        # Fit calibrators on perfectly out-of-sample calibration predictions
        from sklearn.linear_model import LogisticRegression as SkLogReg
        iso_calibrator = IsotonicRegression(out_of_bounds='clip')
        iso_calibrator.fit(out_of_sample_probs, out_of_sample_y)
        
        sig_calibrator = SkLogReg()
        # reshape for logreg
        sig_calibrator.fit(np.array(out_of_sample_probs).reshape(-1, 1), out_of_sample_y)
        
        # 3. TEST SET PREDICTIONS
        # Final base model for test set: trained on target_end_date < test_start
        final_train_mask = (df['target_end_date'] < test_start) & df['target'].notna()
        final_train = df[final_train_mask]
        
        hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
        hgb.fit(final_train[features], final_train['target'])
        
        # Test positive class extraction invariant
        assert 1.0 in hgb.classes_, "Target class 1 not found in model classes."
        cls_idx = list(hgb.classes_).index(1.0)
        
        raw_preds = hgb.predict_proba(test_df[features])[:, cls_idx]
        iso_preds = iso_calibrator.predict(raw_preds)
        sig_preds = sig_calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, list(sig_calibrator.classes_).index(1.0)]
        
        # Calibrated Momentum 12M
        rank_calib = df[df[date_col].isin(calib_dates)]['excess_ret_12m'].rank(pct=True).values
        y_calib = df[df[date_col].isin(calib_dates)]['target'].values
        valid_idx = ~np.isnan(rank_calib) & ~np.isnan(y_calib)
        iso_mom = IsotonicRegression(out_of_bounds='clip')
        if len(rank_calib[valid_idx]) > 10:
            iso_mom.fit(rank_calib[valid_idx], y_calib[valid_idx])
            mom_preds = iso_mom.predict(test_df['excess_ret_12m'].rank(pct=True).values)
        else:
            mom_preds = test_df['excess_ret_12m'].rank(pct=True).values
        
        models = {
            'Raw_HistGBM': raw_preds,
            'Calibrated_Sigmoid': sig_preds,
            'Calibrated_Isotonic': iso_preds,
            'Calibrated_Mom12M': mom_preds
        }
        
        # 4. AUDIT TABLE
        # The user requested verification that base_train_max_target_end < calibration_start
        # This refers to the strictly out-of-sample boundary for the INITIAL base model of the calibration rolling window.
        initial_base_mask = (df['target_end_date'] < calib_dates[0]) & df['target'].notna()
        initial_base = df[initial_base_mask]
        
        audit_row = {
            'Fold': str(test_start.date()),
            'Base Train Obs Start': str(initial_base[date_col].min().date()),
            'Base Train Obs End': str(initial_base[date_col].max().date()),
            'Base Train Max Target End': str(initial_base['target_end_date'].max().date()),
            'Calibration Obs Start': str(pd.to_datetime(calib_dates[0]).date()),
            'Calibration Obs End': str(pd.to_datetime(calib_dates[-1]).date()),
            'Calibration Max Target End': str(df[df[date_col].isin(calib_dates)]['target_end_date'].max().date()),
            'Test Obs Start': str(test_df[date_col].min().date()),
            'Test Obs End': str(test_df[date_col].max().date()),
            'Test Max Target End': str(test_df['target_end_date'].max().date()),
            'Raw NAV End Date': str(raw_nav_end_date.date())
        }
        results['audit_table'].append(audit_row)
        
        # Programmatic Assertions for Audit
        assert pd.to_datetime(audit_row['Base Train Max Target End']) < pd.to_datetime(audit_row['Calibration Obs Start']), "Audit Failed: Train target leaks into Calib Obs"
        assert pd.to_datetime(audit_row['Calibration Max Target End']) < pd.to_datetime(audit_row['Test Obs Start']), "Audit Failed: Calib target leaks into Test Obs"
        assert pd.to_datetime(audit_row['Test Max Target End']) <= raw_nav_end_date, "Audit Failed: Test target exceeds Raw NAV data"
        
        fold_name = f"Holdout_{test_start.date()}" if is_holdout else f"CV_{test_start.date()}"
        print(f"Evaluated {fold_name}: {len(test_df)} rows. Target rate: {pos_rate:.3f}")
        
        fold_meta = {
            'fold_name': fold_name,
            'is_holdout': is_holdout,
            'num_funds': int(test_df['scheme_code'].nunique()),
            'num_categories': int(test_df['category'].nunique()),
            'num_months': int(test_df[date_col].nunique())
        }
        
        def calc_calibration_metrics(y_true, y_prob, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            binids = np.digitize(y_prob, bins) - 1
            bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
            bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
            bin_total = np.bincount(binids, minlength=n_bins)
            nonzero = bin_total > 0
            if not np.any(nonzero): return np.nan, np.nan, np.nan, np.nan
            mean_pred = bin_sums[nonzero] / bin_total[nonzero]
            mean_true = bin_true[nonzero] / bin_total[nonzero]
            abs_diff = np.abs(mean_pred - mean_true)
            ece = np.sum(abs_diff * (bin_total[nonzero] / len(y_true)))
            mce = np.max(abs_diff)
            eps = 1e-15
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
            lr = LogisticRegression()
            lr.fit(log_odds.reshape(-1, 1), y_true)
            return float(ece), float(mce), float(lr.coef_[0][0]), float(lr.intercept_[0])
            
        fold_metrics = {}
        for m_name, preds in models.items():
            auc = roc_auc_score(test_df['target'], preds)
            brier = brier_score_loss(test_df['target'], preds)
            ll = log_loss(test_df['target'], preds)
            
            try:
                ece, mce, slope, intercept = calc_calibration_metrics(test_df['target'].values, preds)
            except:
                ece, mce, slope, intercept = np.nan, np.nan, np.nan, np.nan
            
            test_df['pred'] = preds
            
            # Quintiles Q1-Q5
            test_df['quintile'] = test_df.groupby(date_col)['pred'].transform(
                lambda x: pd.qcut(x + np.random.normal(0, 1e-8, len(x)), 5, labels=[1,2,3,4,5])
            )
            q_rets = test_df.groupby('quintile')['forward_excess_ret_6m'].mean()
            top_min_bot = q_rets.get(5, np.nan) - q_rets.get(1, np.nan)
            
            fold_metrics[m_name] = {
                'roc_auc': float(auc),
                'brier_score': float(brier),
                'log_loss': float(ll),
                'ece': float(ece),
                'mce': float(mce),
                'calib_slope': float(slope),
                'calib_intercept': float(intercept),
                'q_top_minus_bottom': float(top_min_bot)
            }
            
            if is_holdout and m_name == 'Calibrated_Isotonic':
                ho_df = test_df[['scheme_code', date_col, 'category', 'target', 'forward_excess_ret_6m', 'pred', 'quintile']].copy()
                ho_df['model'] = m_name
                holdout_predictions.append(ho_df)
                
        results['folds'].append({'metadata': fold_meta, 'metrics': fold_metrics})
        
    # --- POST PROCESSING HOLDOUT ---
    if holdout_predictions:
        ho_df = pd.concat(holdout_predictions).rename(columns={date_col: 'date'})
        
        # Overall Holdout Checks
        results['holdout_summary'] = {
            'evaluated_rows': len(ho_df),
            'target_positive_rate': float(ho_df['target'].mean())
        }
        
        # Block Bootstrap CIs
        def _auc(df):
            if len(np.unique(df['target'])) < 2: return np.nan
            return roc_auc_score(df['target'], df['pred'])
            
        def _spread(df):
            q = df.groupby('date')['pred'].transform(lambda x: pd.qcut(x + np.random.normal(0, 1e-8, len(x)), 5, labels=[1,2,3,4,5]))
            g = df.copy(); g['q'] = q
            means = g.groupby('q')['forward_excess_ret_6m'].mean()
            if 5 in means and 1 in means:
                return means[5] - means[1]
            return np.nan
            
        results['holdout_clustered_metrics']['AUC'] = block_bootstrap(ho_df, _auc)
        results['holdout_clustered_metrics']['Q5_Q1_Spread'] = block_bootstrap(ho_df, _spread)
        
        # Probability Bucketing
        bins = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.0]
        labels = ['<40%', '40-45%', '45-50%', '50-55%', '55-60%', '60-70%', '>70%']
        ho_df['prob_bucket'] = pd.cut(ho_df['pred'], bins=bins, labels=labels)
        bucket_stats = ho_df.groupby('prob_bucket').agg(
            count=('target', 'count'),
            success_rate=('target', 'mean')
        ).reset_index()
        
        results['probability_buckets']['Calibrated_Isotonic'] = bucket_stats.to_dict('records')

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/advanced_backtest.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Backtest complete. Results saved.")

if __name__ == "__main__":
    run_advanced_backtest()
