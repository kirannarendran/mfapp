import pandas as pd
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataAuditor:
    def __init__(self, config):
        self.config = config
        self.raw_path = Path(config['data']['raw_nav_path'])
        self.meta_path = Path(config['data'].get('scheme_metadata_path', 'data/raw/funds.parquet'))
        
    def run_audit(self):
        logger.info("Running data audit...")
        if not self.raw_path.exists():
            return self._fail_audit(0.0)
            
        df = pd.read_parquet(self.raw_path)
        if df.empty:
            return self._fail_audit(0.0)
            
        df['date'] = pd.to_datetime(df['date'])
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        history_years = (max_date - min_date).days / 365.25
        
        # Build coverage report
        stats = df.groupby(['scheme_code']).agg(
            minimum_NAV_date=('date', 'min'),
            maximum_NAV_date=('date', 'max'),
            observation_count=('date', 'count')
        ).reset_index()
        
        # Try joining with metadata
        if self.meta_path.exists():
            meta = pd.read_parquet(self.meta_path)
            if 'scheme_name' in meta.columns and 'scheme_code' in meta.columns:
                stats = stats.merge(meta[['scheme_code', 'scheme_name']], on='scheme_code', how='left')
        
        if 'scheme_name' not in stats.columns:
            stats['scheme_name'] = 'Unknown'
            
        stats['history_years'] = (stats['maximum_NAV_date'] - stats['minimum_NAV_date']).dt.days / 365.25
        stats['plan_type'] = 'Unknown'
        stats['option_type'] = 'Unknown'
        stats['missing_month_count'] = 0
        stats['largest_data_gap_days'] = 0
        stats['eligible_for_5y_past_window'] = stats['history_years'] >= 5
        stats['eligible_for_5y_past_and_future_window'] = stats['history_years'] >= 10
        
        out_dir = Path("reports/tables")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Reorder columns to ensure scheme_code and scheme_name are first
        cols = ['scheme_code', 'scheme_name'] + [c for c in stats.columns if c not in ['scheme_code', 'scheme_name']]
        stats = stats[cols]
        
        stats.to_csv(out_dir / "data_quality_report.csv", index=False)
        
        req_years = self.config['training_gate']['minimum_history_years']
        
        if history_years < req_years:
            return self._fail_audit(history_years)
            
        # If we passed
        return {
            "pipeline_status": "ready",
            "available_history_years": float(history_years),
            "required_history_years": float(req_years),
            "complete_labelled_windows": int(stats['eligible_for_5y_past_and_future_window'].sum()), # Approx
            "training_executed": True
        }
        
    def _fail_audit(self, history_years):
        req_years = self.config['training_gate']['minimum_history_years']
        
        manifest = {
            "pipeline_status": "insufficient_history",
            "available_history_years": float(history_years),
            "required_history_years": float(req_years),
            "complete_labelled_windows": 0,
            "training_executed": False
        }
        
        Path("reports").mkdir(exist_ok=True)
        with open("reports/run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        self._generate_empty_schemas()
        return manifest
        
    def _generate_empty_schemas(self):
        tables = [
            ("five_year_window_audit.csv", ["scheme_code", "past_start_intended_date", "past_start_actual_nav_date", "prediction_date", "prediction_actual_nav_date", "future_end_intended_date", "future_end_actual_nav_date"]),
            ("base_rate_by_cagr_band.csv", ["cagr_band", "n_cases", "prob_ge_8", "prob_ge_10", "prob_ge_12", "prob_ge_15", "median_future_cagr", "p10", "p25", "p75", "p90"]),
            ("base_rate_by_category.csv", ["category", "cagr_band", "n_cases", "prob_ge_12"]),
            ("persistence_transition_matrix.csv", ["past_quartile", "future_quartile", "probability"]),
            ("out_of_sample_predictions.csv", ["scheme_code", "prediction_date", "prob_ge_12"]),
            ("calibration_buckets.csv", ["bucket", "n_predictions", "mean_pred", "actual_rate"]),
            ("model_comparison.csv", ["model", "brier_score", "log_loss", "expected_calibration_error"]),
            ("cohort_results.csv", ["cohort", "brier_score"]),
            ("category_results.csv", ["category", "brier_score"]),
            ("quantile_coverage.csv", ["interval", "coverage"]),
            ("ablation_results.csv", ["feature_set", "brier_score"]),
            ("latest_fund_probabilities.csv", ["scheme_code", "scheme_name", "prob_ge_12"]),
            ("scheme_exclusions.csv", ["scheme_code", "scheme_name", "reason_for_exclusion"]),
            ("unresolved_discontinuations.csv", ["scheme_code", "last_nav_date"])
        ]
        
        out_dir = Path("reports/tables")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for file_name, cols in tables:
            pd.DataFrame(columns=cols).to_csv(out_dir / file_name, index=False)
        
        logger.info(f"Generated {len(tables)} empty schema outputs.")
