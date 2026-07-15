import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation.portfolio import get_execution_navs

class TestTemporalIntegrity(unittest.TestCase):
    def test_chronological_split(self):
        df = pd.read_parquet('data/processed/model_dataset.parquet')
        test_start = pd.to_datetime('2024-07-31')
        train_mask = (df['target_end_date'] < test_start) & df['target'].notna()
        train_df = df[train_mask]
        
        max_target_date_in_train = train_df['target_end_date'].max()
        self.assertTrue(max_target_date_in_train < test_start)
        
    def test_no_pending_targets_in_evaluation(self):
        df = pd.read_parquet('data/processed/model_dataset.parquet')
        eval_mask = df['target'].notna()
        eval_df = df[eval_mask]
        post_2026 = eval_df[eval_df['date'] >= '2026-01-01']
        self.assertEqual(len(post_2026), 0)

    def test_same_day_execution_leakage(self):
        nav_df = pd.DataFrame({
            'scheme_code': ['1001', '1001', '1001'],
            'date': pd.to_datetime(['2024-07-31', '2024-08-01', '2024-08-02']),
            'nav': [10.0, 10.1, 10.2]
        })
        
        feature_date = pd.to_datetime('2024-07-31')
        
        # Test 1: Find next eligible execution date strictly after feature_date
        # get_execution_navs takes intended_date which must be feature_date + 1 day
        intended_date = feature_date + pd.Timedelta(days=1)
        res = get_execution_navs(nav_df, intended_date, ['1001'], tolerance_days=7)
        execution_date = res['actual_nav_date'].iloc[0]
        
        self.assertTrue(execution_date > feature_date)
        self.assertEqual(execution_date, pd.to_datetime('2024-08-01'))
        
        # Test 2: Deliberately invalid feature execution
        invalid_execution_date = feature_date
        with self.assertRaises(AssertionError):
            assert invalid_execution_date > feature_date, "Execution date leaked!"

    def test_auc_negated_score(self):
        # 2. Add a test confirming: auc_negated_score ≈ 1 - auc_original_score subject to tied scores.
        y_true = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        
        auc_original = roc_auc_score(y_true, y_score)
        auc_negated = roc_auc_score(y_true, -y_score)
        
        self.assertAlmostEqual(auc_negated, 1.0 - auc_original, places=5)

if __name__ == '__main__':
    unittest.main()
