import pandas as pd
import joblib
import sqlite3
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def predict():
    config = load_config()
    date_col = config['data']['date_column']
    
    print("Loading features and models...")
    df = pd.read_parquet("data/processed/features.parquet")
    df[date_col] = pd.to_datetime(df[date_col])
    
    model = joblib.load("models/model.pkl")
    calibrator = joblib.load("models/calibrator.pkl")
    features = joblib.load("models/features.pkl")
    
    latest_date = df[date_col].max()
    print(f"Latest feature date: {latest_date.date()}")
    
    latest_df = df[df[date_col] == latest_date].copy()
    latest_df = latest_df.dropna(subset=features)
    
    print(f"Generating predictions for {len(latest_df)} funds...")
    X = latest_df[features]
    
    raw_probs = model.predict_proba(X)[:, 1]
    calib_probs = calibrator.predict(raw_probs)
    
    latest_df['raw_model_score'] = calib_probs
    latest_df['prediction_date'] = latest_date.strftime('%Y-%m-%d')
    
    # Calculate ranking within category and date
    latest_df["ml_ranking_score"] = (
        latest_df.groupby(["prediction_date", "category"])["raw_model_score"]
        .rank(pct=True, method="average")
        .mul(100)
    )
    
    # Peer count and status logic
    latest_df['ml_category_peer_count'] = latest_df.groupby(["prediction_date", "category"])['scheme_code'].transform('count')
    latest_df['ml_score_status'] = 'current'
    
    # Check for metadata
    try:
        model_metadata = joblib.load("models/metadata.pkl")
        ml_training_cutoff_date = model_metadata.get('training_cutoff_date', None)
        metadata_status = "ok"
    except Exception:
        ml_training_cutoff_date = None
        metadata_status = "missing"
        
    latest_df['ml_training_cutoff_date'] = ml_training_cutoff_date
    latest_df['metadata_status'] = metadata_status
    
    # Calculate expiration date
    stale_after_days = config.get('legacy_ml', {}).get('stale_after_days', 45)
    latest_df['ml_expires_at'] = (pd.to_datetime(latest_df['prediction_date']) + pd.Timedelta(days=stale_after_days)).dt.strftime('%Y-%m-%d')
    
    # Status precedence
    latest_df['ml_score_status'] = 'current'
    
    # Check stale
    today_str = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
    stale_mask = today_str > latest_df['ml_expires_at']
    latest_df.loc[stale_mask, 'ml_score_status'] = 'stale'
    
    # Check metadata
    if ml_training_cutoff_date is None or metadata_status == "missing":
        latest_df['ml_score_status'] = 'model_metadata_incomplete'
        latest_df['ml_training_cutoff_date'] = None
        
    # Check peers
    insufficient_mask = latest_df['ml_category_peer_count'] < 10
    latest_df.loc[insufficient_mask, 'ml_score_status'] = 'insufficient_category_peers'
    
    # Suppress ranking score
    latest_df.loc[latest_df['ml_score_status'] != 'current', 'ml_ranking_score'] = None
    
    # Add metadata fields
    latest_df['ml_score_as_of_date'] = latest_df['prediction_date']
    latest_df['ml_model_version'] = "legacy-6m-category-outperformance"
    latest_df['ml_status'] = "experimental"
    
    db_path = config['data']['database_path']
    conn = sqlite3.connect(db_path)
    
    print("Writing predictions to mf_tracker.db...")
    preds_to_save = latest_df[[
        'scheme_code', 
        'raw_model_score', 
        'ml_ranking_score', 
        'prediction_date', 
        'ml_score_as_of_date',
        'ml_expires_at',
        'ml_model_version',
        'ml_training_cutoff_date',
        'ml_status',
        'ml_score_status',
        'ml_category_peer_count'
    ]].copy()
    
    # Atomic migration pattern
    # 1. Write to new table
    preds_to_save.to_sql('ml_predictions_new', conn, if_exists='replace', index=False)
    
    # 2. Validate
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ml_predictions_new")
    new_count = cursor.fetchone()[0]
    
    if new_count == 0:
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_new")
        conn.close()
        raise Exception("Validation failed: No rows in ml_predictions_new")
        
    cursor.execute("SELECT COUNT(*) FROM ml_predictions_new GROUP BY scheme_code, prediction_date HAVING COUNT(*) > 1")
    if cursor.fetchone():
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_new")
        conn.close()
        raise Exception("Validation failed: Duplicates found for scheme_code x prediction_date")
        
    cursor.execute("SELECT COUNT(*) FROM ml_predictions_new WHERE ml_ranking_score IS NOT NULL AND (ml_ranking_score <= 0 OR ml_ranking_score > 100)")
    row = cursor.fetchone()
    if row and row[0] > 0:
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_new")
        conn.close()
        raise Exception("Validation failed: Scores out of bounds (0, 100]")
        
    cursor.execute("SELECT COUNT(*) FROM ml_predictions_new WHERE date(ml_expires_at) IS NULL")
    if cursor.fetchone()[0] > 0:
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_new")
        conn.close()
        raise Exception("Validation failed: Invalid date format in ml_expires_at")
        
    cursor.execute("SELECT COUNT(*) FROM ml_predictions_new WHERE ml_score_status NOT IN ('current', 'stale', 'model_metadata_incomplete', 'insufficient_category_peers')")
    if cursor.fetchone()[0] > 0:
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_new")
        conn.close()
        raise Exception("Validation failed: Invalid status value")
        
    try:
        # 3. Begin transaction
        conn.execute("BEGIN TRANSACTION")
        
        # 4. Rename existing to backup
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_backup")
        try:
            cursor.execute("ALTER TABLE ml_predictions RENAME TO ml_predictions_backup")
        except sqlite3.OperationalError:
            pass # Table might not exist yet
            
        # Check if a failure is injected for testing
        if os.environ.get('INJECT_MIGRATION_FAILURE') == '1':
            raise Exception("Injected failure during migration")
            
        # 5. Rename new to live
        cursor.execute("ALTER TABLE ml_predictions_new RENAME TO ml_predictions")
        
        # 6. Recreate required indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_preds_scheme ON ml_predictions(scheme_code)")
        
        # 7. Commit
        conn.commit()
        print("Predictions successfully saved to DB (Atomic migration).")
        
        # 8. Remove backup
        cursor.execute("DROP TABLE IF EXISTS ml_predictions_backup")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Migration failed, rolling back: {e}")
        try:
            cursor.execute("DROP TABLE IF EXISTS ml_predictions_new")
            # If we renamed to backup but failed before live, restore it
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions_backup'")
            if cursor.fetchone():
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE ml_predictions_backup RENAME TO ml_predictions")
            conn.commit()
        except Exception as rollback_err:
            print(f"Secondary rollback error: {rollback_err}")
        finally:
            conn.close()
        raise e
    finally:
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    import numpy as np
    predict()
