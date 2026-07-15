import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from backtest import compute_metrics
import warnings
warnings.filterwarnings('ignore')

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    date_col = config['data']['date_column']
    
    print("Loading modeling dataset...")
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    df[date_col] = pd.to_datetime(df[date_col])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    
    features = [
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
        'mom_12_1', 'mom_6_1', 'vol_12m', 'vol_6m',
        'dist_ma_6m', 'dist_ma_12m',
        'rank_ret_6m', 'rank_ret_12m', 'rank_vol_12m', 'excess_ret_6m'
    ]
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features)
    
    # We only train on rows that have a valid target (i.e. not the most recent 6 months)
    train_df = df.dropna(subset=['target']).copy()
    
    if len(train_df) == 0:
        print("No training data available.")
        return
        
    train_df = train_df.sort_values(date_col)
    
    # Validation set for calibration: last 6 months of available training data
    val_start = train_df[date_col].max() - pd.DateOffset(months=6)
    
    t_mask = train_df[date_col] < val_start
    v_mask = train_df[date_col] >= val_start
    
    X_t, y_t = train_df.loc[t_mask, features], train_df.loc[t_mask, 'target']
    X_v, y_v = train_df.loc[v_mask, features], train_df.loc[v_mask, 'target']
    
    print(f"Training on {len(X_t)} rows, calibrating on {len(X_v)} rows...")
    
    # Train base model
    hgb = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_depth=6, early_stopping=False, random_state=42
    )
    hgb.fit(X_t, y_t)
    
    # Calibrate
    iso = IsotonicRegression(out_of_bounds='clip')
    probs_v = hgb.predict_proba(X_v)[:, 1]
    iso.fit(probs_v, y_v)
    
    # Quick metrics on validation set
    metrics = compute_metrics(y_v, iso.predict(probs_v), train_df.loc[v_mask, 'forward_excess_ret_6m'])
    print(f"Validation Metrics: AUC={metrics['roc_auc']:.3f}, Brier={metrics['brier_score']:.3f}, Spearman={metrics['spearman']:.3f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(hgb, "models/model.pkl")
    joblib.dump(iso, "models/calibrator.pkl")
    joblib.dump(features, "models/features.pkl")
    print("Model and calibrator saved.")

if __name__ == "__main__":
    train_model()
