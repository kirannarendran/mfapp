import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

df = pd.read_parquet('data/processed/model_dataset.parquet')
df['date'] = pd.to_datetime(df['date'])
df['target_end_date'] = pd.to_datetime(df['target_end_date'])

features = [
    'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
    'mom_12_1', 'mom_6_1', 'vol_12m', 'vol_6m',
    'dist_ma_6m', 'dist_ma_12m',
    'rank_ret_6m', 'rank_ret_12m', 'rank_vol_12m', 'excess_ret_6m'
]

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
df = df.sort_values('date')

cv_dates = ['2024-07-31', '2025-01-31']
results = []

for cv_d in cv_dates:
    cv_start = pd.to_datetime(cv_d)
    test_mask = (df['date'] >= cv_start) & (df['date'] < cv_start + pd.DateOffset(months=6))
    test_df = df[test_mask]
    test_df = test_df[test_df['target'].notna()]
    if len(test_df) == 0: continue
    
    # Base model train
    train_mask = (df['target_end_date'] < cv_start) & df['target'].notna()
    train_df = df[train_mask]
    
    hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
    hgb.fit(train_df[features], train_df['target'])
    
    cls_idx = list(hgb.classes_).index(1.0)
    raw_preds = hgb.predict_proba(test_df[features])[:, cls_idx]
    
    # Constant baseline
    const_pred = np.full(len(test_df), train_df['target'].mean())
    
    # Calibration
    calib_end = cv_start - pd.DateOffset(months=6)
    calib_start = calib_end - pd.DateOffset(months=6)
    calib_dates = np.sort(df[(df['date'] >= calib_start) & (df['date'] < calib_end)]['date'].unique())
    
    out_of_sample_probs = []
    out_of_sample_margins = []
    out_of_sample_y = []
    
    for c in calib_dates:
        c = pd.to_datetime(c)
        c_train_mask = (df['target_end_date'] < c) & df['target'].notna()
        c_train = df[c_train_mask]
        if len(c_train) < 500: continue
            
        c_hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
        c_hgb.fit(c_train[features], c_train['target'])
        
        c_test = df[df['date'] == c].copy()
        c_test = c_test[c_test['target'].notna()]
        if len(c_test) > 0:
            out_of_sample_probs.extend(c_hgb.predict_proba(c_test[features])[:, 1])
            out_of_sample_margins.extend(c_hgb.decision_function(c_test[features]))
            out_of_sample_y.extend(c_test['target'].values)
            
    iso_preds = raw_preds.copy()
    sig_preds = raw_preds.copy()
    
    if len(out_of_sample_y) > 100:
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(out_of_sample_probs, out_of_sample_y)
        iso_preds = iso.predict(raw_preds)
        
        lr = LogisticRegression()
        lr.fit(np.array(out_of_sample_margins).reshape(-1, 1), out_of_sample_y)
        raw_margins = hgb.decision_function(test_df[features])
        sig_preds = lr.predict_proba(raw_margins.reshape(-1, 1))[:, 1]
        
    y_test = test_df['target'].values
    
    results.append({
        'fold': cv_d,
        'brier_raw': brier_score_loss(y_test, raw_preds),
        'log_raw': log_loss(y_test, raw_preds),
        'brier_const': brier_score_loss(y_test, const_pred),
        'log_const': log_loss(y_test, const_pred),
        'brier_iso': brier_score_loss(y_test, iso_preds),
        'log_iso': log_loss(y_test, iso_preds),
        'brier_sig': brier_score_loss(y_test, sig_preds),
        'log_sig': log_loss(y_test, sig_preds)
    })

print(pd.DataFrame(results).to_string())
