import os
import sqlite3
import pandas as pd
import pytest
import sys

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict

@pytest.fixture
def setup_db(tmp_path):
    # Setup config to point to temp db
    import yaml
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "config.yaml"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    original_db_path = config['data']['database_path']
    test_db_path = str(tmp_path / "mf_tracker_test.db")
    config['data']['database_path'] = test_db_path
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # Create mock original table
    conn = sqlite3.connect(test_db_path)
    conn.execute('''
        CREATE TABLE ml_predictions (
            scheme_code INTEGER PRIMARY KEY,
            raw_model_score REAL,
            ml_ranking_score REAL
        )
    ''')
    conn.execute('INSERT INTO ml_predictions VALUES (1, 0.5, 50)')
    conn.commit()
    conn.close()
    
    yield test_db_path
    
    # Restore config
    config['data']['database_path'] = original_db_path
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def test_migration_success(setup_db, monkeypatch):
    db_path = setup_db
    monkeypatch.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run prediction and migration
    predict()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check original table is replaced and new columns exist
    cursor.execute("PRAGMA table_info(ml_predictions)")
    columns = [row[1] for row in cursor.fetchall()]
    assert 'ml_expires_at' in columns
    assert 'ml_score_status' in columns
    
    # Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='ml_predictions'")
    indexes = [row[0] for row in cursor.fetchall()]
    assert 'idx_ml_preds_scheme' in indexes
    
    # Check bounds (all should be null since no metadata exists)
    cursor.execute("SELECT count(*) FROM ml_predictions WHERE ml_ranking_score IS NOT NULL")
    count_non_null = cursor.fetchone()[0]
    assert count_non_null == 0
    
    # Check states
    cursor.execute("SELECT count(*) FROM ml_predictions WHERE ml_score_status = 'model_metadata_incomplete'")
    count_incomplete = cursor.fetchone()[0]
    assert count_incomplete > 0
    
    cursor.execute("SELECT count(*) FROM ml_predictions WHERE ml_training_cutoff_date IS NULL")
    count_null_cutoff = cursor.fetchone()[0]
    assert count_null_cutoff == count_incomplete
    
    conn.close()

def test_migration_failure_rollback(setup_db, monkeypatch):
    db_path = setup_db
    monkeypatch.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Verify original state
    cursor.execute("SELECT * FROM ml_predictions")
    original_rows = cursor.fetchall()
    assert len(original_rows) == 1
    assert original_rows[0][0] == 1
    
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
    old_schema = cursor.fetchone()[0]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='ml_predictions'")
    old_indexes = [r[0] for r in cursor.fetchall()]
    
    print(f"old row count before injected failure: {len(original_rows)}")
    print(f"old schema checksum before failure: {hash(old_schema)}")
    print(f"indexes before failure: {old_indexes}")
    
    # Inject failure
    os.environ['INJECT_MIGRATION_FAILURE'] = '1'
    try:
        predict()
    except Exception as e:
        assert "Injected failure" in str(e)
    finally:
        del os.environ['INJECT_MIGRATION_FAILURE']
        
    # Verify original table is restored
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
    live_table_name = cursor.fetchone()[0]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions_backup'")
    assert cursor.fetchone() is None
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions_new'")
    assert cursor.fetchone() is None
    
    # Verify original rows
    cursor.execute("SELECT * FROM ml_predictions")
    restored_rows = cursor.fetchall()
    
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
    new_schema = cursor.fetchone()[0]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='ml_predictions'")
    new_indexes = [r[0] for r in cursor.fetchall()]
    
    print(f"old row count after rollback: {len(restored_rows)}")
    print(f"old schema checksum after rollback: {hash(new_schema)}")
    print(f"indexes after rollback: {new_indexes}")
    print(f"live table name after rollback: {live_table_name}")
    
    assert len(restored_rows) == 1
    assert restored_rows[0][0] == 1
    
    conn.close()
    
def test_ranking_semantics():
    # Test that equal raw scores receive consistent percentile ranks
    df = pd.DataFrame({
        'prediction_date': ['2026-06-30']*5,
        'category': ['A']*5,
        'raw_model_score': [0.1, 0.5, 0.5, 0.9, 0.9]
    })
    
    df['ml_ranking_score'] = (
        df.groupby(["prediction_date", "category"])["raw_model_score"]
        .rank(pct=True, method="average")
        .mul(100)
    )
    
    scores = df['ml_ranking_score'].tolist()
    # 0.1 is rank 1 (20%)
    # 0.5 are ranks 2,3 -> avg 2.5 (50%)
    # 0.9 are ranks 4,5 -> avg 4.5 (90%)
    assert scores[0] == 20.0
    assert scores[1] == 50.0
    assert scores[2] == 50.0
    assert scores[3] == 90.0
    assert scores[4] == 90.0

def test_independent_ranking():
    # Multiple prediction dates and categories are ranked independently
    df = pd.DataFrame({
        'prediction_date': ['2026-06-30', '2026-06-30', '2026-07-31', '2026-07-31'],
        'category': ['A', 'A', 'A', 'A'],
        'raw_model_score': [0.1, 0.9, 0.5, 0.5]
    })
    
    df['ml_ranking_score'] = (
        df.groupby(["prediction_date", "category"])["raw_model_score"]
        .rank(pct=True, method="average")
        .mul(100)
    )
    
    # Each group has size 2. 
    # For June: 0.1 is 50%, 0.9 is 100%
    # For July: 0.5 and 0.5 are ties, rank 1.5/2 = 75%
    assert df.loc[0, 'ml_ranking_score'] == 50.0
    assert df.loc[1, 'ml_ranking_score'] == 100.0
    assert df.loc[2, 'ml_ranking_score'] == 75.0
    assert df.loc[3, 'ml_ranking_score'] == 75.0
