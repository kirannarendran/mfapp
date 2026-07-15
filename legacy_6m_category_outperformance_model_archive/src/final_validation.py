import pandas as pd
import numpy as np
import yaml
import json
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def calc_calibration_metrics(y_true, y_prob, n_bins=10):
    # ECE and MCE
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    
    nonzero = bin_total > 0
    bin_sums = bin_sums[nonzero]
    bin_true = bin_true[nonzero]
    bin_total = bin_total[nonzero]
    
    mean_pred = bin_sums / bin_total
    mean_true = bin_true / bin_total
    
    abs_diff = np.abs(mean_pred - mean_true)
    ece = np.sum(abs_diff * (bin_total / len(y_true)))
    mce = np.max(abs_diff)
    
    # Intercept and Slope via Logistic Regression of log-odds
    # Avoid log(0)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
    
    lr = LogisticRegression()
    lr.fit(log_odds.reshape(-1, 1), y_true)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    
    return float(ece), float(mce), float(slope), float(intercept)

def time_aware_paired_bootstrap(df, metric_func, block_size=6, n_bootstraps=1000):
    dates = np.sort(df['date'].unique())
    if len(dates) < block_size * 2:
        return None, None
        
    metrics = []
    n_blocks = len(dates) // block_size + 1
    
    for _ in range(n_bootstraps):
        starts = np.random.choice(len(dates) - block_size + 1, size=n_blocks, replace=True)
        boot_dates = []
        for start in starts:
            boot_dates.extend(dates[start:start+block_size])
        boot_dates = boot_dates[:len(dates)]
        
        boot_df = pd.concat([df[df['date'] == d] for d in boot_dates])
        
        try:
            val = metric_func(boot_df)
            if not np.isnan(val):
                metrics.append(val)
        except:
            pass
            
    if not metrics:
        return None, None
        
    mean = np.mean(metrics)
    ci_lower = np.percentile(metrics, 2.5)
    ci_upper = np.percentile(metrics, 97.5)
    
    return float(mean), (float(ci_lower), float(ci_upper)), metrics

def run_validation():
    print("Loading data...")
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    df['date'] = pd.to_datetime(df['date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df = df.sort_values('date')
    
    features_full = [
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
        'mom_12_1', 'mom_6_1', 'vol_12m', 'vol_6m',
        'dist_ma_6m', 'dist_ma_12m',
        'rank_ret_6m', 'rank_ret_12m', 'rank_vol_12m', 'excess_ret_6m'
    ]
    features_no_mom = [f for f in features_full if f not in ['mom_12_1', 'mom_6_1']]
    features_mom_only = ['mom_12_1', 'mom_6_1']
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_full)
    dates = np.sort(df['date'].unique())
    test_starts = dates[24::6]
    
    all_preds = []
    
    print("Running Ablation Study over Folds...")
    for test_start in test_starts:
        test_start = pd.to_datetime(test_start)
        test_mask = (df['date'] >= test_start) & (df['date'] < test_start + pd.DateOffset(months=6))
        test_df = df[test_mask].copy()
        test_df = test_df[test_df['target'].notna() & test_df['forward_ret_6m'].notna()]
        
        if len(test_df) == 0: continue
            
        train_mask = (df['target_end_date'] < test_start) & df['target'].notna()
        train_df = df[train_mask]
        
        # 1. HistGBM Full
        hgb_full = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        hgb_full.fit(train_df[features_full], train_df['target'])
        pred_full = hgb_full.predict_proba(test_df[features_full])[:, 1]
        
        # 2. HistGBM No Mom
        hgb_no_mom = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        hgb_no_mom.fit(train_df[features_no_mom], train_df['target'])
        pred_no_mom = hgb_no_mom.predict_proba(test_df[features_no_mom])[:, 1]
        
        # 3. Mom Only (Logistic Regression)
        lr_mom = LogisticRegression()
        lr_mom.fit(train_df[features_mom_only], train_df['target'])
        pred_mom = lr_mom.predict_proba(test_df[features_mom_only])[:, 1]
        
        # 4. Ensemble
        pred_ens = (pred_no_mom + pred_mom) / 2
        
        test_df['pred_full'] = pred_full
        test_df['pred_no_mom'] = pred_no_mom
        test_df['pred_mom'] = pred_mom
        test_df['pred_ens'] = pred_ens
        
        # Calibrated Mom 12M Rank (from advanced backtest logic)
        calib_start = test_start - pd.DateOffset(months=12)
        calib_end = test_start - pd.DateOffset(months=6)
        calib_df = df[(df['date'] >= calib_start) & (df['date'] < calib_end)]
        if len(calib_df) > 100:
            rank_c = calib_df['excess_ret_12m'].rank(pct=True).values
            iso = IsotonicRegression(out_of_bounds='clip')
            valid = ~np.isnan(rank_c) & ~np.isnan(calib_df['target'].values)
            if valid.sum() > 10:
                iso.fit(rank_c[valid], calib_df['target'].values[valid])
                pred_mom_12 = iso.predict(test_df['excess_ret_12m'].rank(pct=True).values)
            else:
                pred_mom_12 = test_df['excess_ret_12m'].rank(pct=True).values
        else:
            pred_mom_12 = test_df['excess_ret_12m'].rank(pct=True).values
            
        test_df['pred_mom_12'] = pred_mom_12
        
        all_preds.append(test_df[['scheme_code', 'date', 'category', 'target', 'forward_excess_ret_6m', 'forward_ret_6m', 'pred_full', 'pred_no_mom', 'pred_mom', 'pred_ens', 'pred_mom_12']])
        
    final_df = pd.concat(all_preds)
    
    results = {'ablation': {}, 'economic': {}, 'bootstrap': {}, 'discrimination': {}}
    
    # 4. Paired Bootstrap
    def paired_auc_diff(d):
        if len(np.unique(d['target'])) < 2: return np.nan
        auc_full = roc_auc_score(d['target'], d['pred_full'])
        auc_mom = roc_auc_score(d['target'], d['pred_mom_12'])
        return auc_full - auc_mom
        
    diff_mean, diff_ci, diff_dist = time_aware_paired_bootstrap(final_df, paired_auc_diff)
    if diff_dist:
        pct_beat = np.mean(np.array(diff_dist) > 0)
    else:
        pct_beat = np.nan
        
    results['bootstrap'] = {
        'auc_diff_mean': diff_mean,
        'auc_diff_ci': diff_ci,
        'pct_beat_momentum': float(pct_beat),
        'n_independent_months': int(final_df['date'].nunique())
    }
    
    # Ablation metrics
    for m in ['pred_full', 'pred_no_mom', 'pred_mom', 'pred_ens']:
        if len(np.unique(final_df['target'])) > 1:
            auc = roc_auc_score(final_df['target'], final_df[m])
            results['ablation'][m] = float(auc)
            
    # 5. Discrimination by period
    for m in ['pred_full', 'pred_mom_12']:
        monthly_aucs = []
        spearmans = []
        pos_spearman_count = 0
        total_cats = 0
        
        for dt, g in final_df.groupby('date'):
            if len(np.unique(g['target'])) > 1:
                monthly_aucs.append(roc_auc_score(g['target'], g[m]))
                
        for (dt, cat), g in final_df.groupby(['date', 'category']):
            if len(g) > 2:
                sp, _ = spearmanr(g[m], g['forward_excess_ret_6m'])
                if not np.isnan(sp):
                    spearmans.append(sp)
                    total_cats += 1
                    if sp > 0: pos_spearman_count += 1
                    
        results['discrimination'][m] = {
            'mean_monthly_auc': float(np.mean(monthly_aucs)),
            'median_monthly_auc': float(np.median(monthly_aucs)),
            'pct_months_gt_05': float(np.mean(np.array(monthly_aucs) > 0.5)),
            'mean_category_spearman': float(np.mean(spearmans)),
            'pct_category_positive_spearman': float(pos_spearman_count / max(1, total_cats))
        }
        
    # 6. Economic Performance
    def simulate_portfolio(d, pred_col, hold_rule=False, cost_bps=0):
        # We need sequential dates to calculate turnover and hold rules
        dates = np.sort(d['date'].unique())
        holdings = set()
        portfolio_returns = []
        turnover_rates = []
        
        cost_dec = cost_bps / 10000.0
        
        for dt in dates:
            dt_df = d[d['date'] == dt].copy()
            dt_df['quintile'] = dt_df.groupby('category')[pred_col].transform(
                lambda x: pd.qcut(x + np.random.normal(0, 1e-8, len(x)), 5, labels=[1,2,3,4,5])
            )
            
            current_q5 = set(dt_df[dt_df['quintile'] == 5]['scheme_code'])
            
            if hold_rule and len(holdings) > 0:
                # Keep if quintile >= 3
                eligible_hold = set(dt_df[dt_df['quintile'] >= 3]['scheme_code'])
                new_holdings = (holdings & eligible_hold) | current_q5
            else:
                new_holdings = current_q5
                
            # Turnover
            if len(holdings) > 0:
                retained = len(new_holdings & holdings)
                turnover = 1 - (retained / max(1, len(holdings)))
                turnover_rates.append(turnover)
            else:
                turnover = 1.0 # 100% turnover on first day
                
            # Returns (forward_ret_6m / 6 to approximate 1-month return)
            # Actually, the user asked for forward category-relative return by quintile.
            # But they also asked for max drawdown and risk adjusted performance.
            # To do time-series properly we'd need non-overlapping returns or 1-month forward returns.
            # We will approximate 1-month return as forward_ret_6m / 6 for the portfolio simulation.
            held_df = dt_df[dt_df['scheme_code'].isin(new_holdings)]
            if len(held_df) > 0:
                ret = held_df['forward_ret_6m'].mean() / 6.0
                # Apply switching costs to the turned-over portion
                ret -= (turnover * cost_dec)
                portfolio_returns.append(ret)
                
            holdings = new_holdings
            
        ret_arr = np.array(portfolio_returns)
        cum_ret = np.cumprod(1 + ret_arr)
        if len(cum_ret) > 0:
            drawdowns = 1 - (cum_ret / np.maximum.accumulate(cum_ret))
            max_dd = np.max(drawdowns)
        else:
            max_dd = 0
            
        ann_ret = np.mean(ret_arr) * 12
        ann_vol = np.std(ret_arr) * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Q5-Q1 Spread (without costs)
        spreads = []
        q5_rets = []
        for dt in dates:
            dt_df = d[d['date'] == dt].copy()
            dt_df['quintile'] = dt_df.groupby('category')[pred_col].transform(
                lambda x: pd.qcut(x + np.random.normal(0, 1e-8, len(x)), 5, labels=[1,2,3,4,5])
            )
            q_mean = dt_df.groupby('quintile')['forward_excess_ret_6m'].mean()
            if 5 in q_mean and 1 in q_mean:
                spreads.append(q_mean[5] - q_mean[1])
                q5_rets.append(q_mean[5])
                
        return {
            'mean_turnover': float(np.mean(turnover_rates)) if turnover_rates else 0,
            'max_drawdown': float(max_dd),
            'sharpe': float(sharpe),
            'ann_ret': float(ann_ret),
            'q5_q1_spread': float(np.mean(spreads)),
            'q5_ret': float(np.mean(q5_rets)),
            'pct_positive_spread': float(np.mean(np.array(spreads) > 0)) if spreads else 0
        }
        
    results['economic']['HistGBM_0bps'] = simulate_portfolio(final_df, 'pred_full', hold_rule=False, cost_bps=0)
    results['economic']['HistGBM_25bps'] = simulate_portfolio(final_df, 'pred_full', hold_rule=False, cost_bps=25)
    results['economic']['HistGBM_50bps'] = simulate_portfolio(final_df, 'pred_full', hold_rule=False, cost_bps=50)
    results['economic']['HistGBM_100bps'] = simulate_portfolio(final_df, 'pred_full', hold_rule=False, cost_bps=100)
    results['economic']['HistGBM_HoldQ3_50bps'] = simulate_portfolio(final_df, 'pred_full', hold_rule=True, cost_bps=50)
    
    results['economic']['Mom12_0bps'] = simulate_portfolio(final_df, 'pred_mom_12', hold_rule=False, cost_bps=0)
    results['economic']['Mom12_HoldQ3_50bps'] = simulate_portfolio(final_df, 'pred_mom_12', hold_rule=True, cost_bps=50)

    # Note: Calibration metrics (ECE, MCE, Intercept, Slope) are extracted from advanced_backtest.json 
    # to avoid re-running the nested rolling CV here.
    
    with open("data/processed/final_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Final validation complete.")

if __name__ == "__main__":
    run_validation()
