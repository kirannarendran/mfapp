import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from collections import deque
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.portfolio import init_portfolio, rebalance, step_forward, get_execution_navs

def run_evidence_appendix():
    out_dir = "reports/final"
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    nav = pd.read_parquet("data/raw/nav_history.parquet")
    
    df['date'] = pd.to_datetime(df['date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    nav['date'] = pd.to_datetime(nav['date'])
    
    raw_nav_end_date = nav['date'].max()
    data_extraction_timestamp = pd.Timestamp("2026-07-12")
    
    last_nav_dates = nav.groupby('scheme_code')['date'].max().to_dict()
    
    features_full = ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m', 'mom_12_1', 'mom_6_1', 
                     'vol_12m', 'vol_6m', 'dist_ma_6m', 'dist_ma_12m',
                     'rank_ret_6m', 'rank_ret_12m', 'rank_vol_12m', 'excess_ret_6m']
                     
    eval_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_full)
    eval_df = eval_df.sort_values('date')
    dates = np.sort(eval_df['date'].unique())
    
    test_starts = pd.to_datetime(['2024-07-31', '2025-01-31', '2025-07-31', '2026-01-31'])
    
    scenarios = [
        {'name': 'HistGBM_1m_0bps', 'model': 'hgb', 'cost': 0, 'stress': 'none', 'freq': '1m'},
        {'name': 'HistGBM_1m_25bps', 'model': 'hgb', 'cost': 25, 'stress': 'none', 'freq': '1m'},
        {'name': 'HistGBM_1m_50bps', 'model': 'hgb', 'cost': 50, 'stress': 'none', 'freq': '1m'},
        {'name': 'HistGBM_1m_100bps', 'model': 'hgb', 'cost': 100, 'stress': 'none', 'freq': '1m'},
        {'name': 'HistGBM_6m_Avg_0bps', 'model': 'hgb', 'cost': 0, 'stress': 'none', 'freq': '6m'},
        {'name': 'HistGBM_6m_Avg_25bps', 'model': 'hgb', 'cost': 25, 'stress': 'none', 'freq': '6m'},
        {'name': 'HistGBM_6m_Avg_50bps', 'model': 'hgb', 'cost': 50, 'stress': 'none', 'freq': '6m'},
        {'name': 'HistGBM_6m_Avg_100bps', 'model': 'hgb', 'cost': 100, 'stress': 'none', 'freq': '6m'},
        
        {'name': 'Mom_1m_0bps', 'model': 'mom', 'cost': 0, 'stress': 'none', 'freq': '1m'},
        {'name': 'Mom_1m_25bps', 'model': 'mom', 'cost': 25, 'stress': 'none', 'freq': '1m'},
        {'name': 'Mom_1m_50bps', 'model': 'mom', 'cost': 50, 'stress': 'none', 'freq': '1m'},
        {'name': 'Mom_1m_100bps', 'model': 'mom', 'cost': 100, 'stress': 'none', 'freq': '1m'},
        {'name': 'Mom_6m_Avg_0bps', 'model': 'mom', 'cost': 0, 'stress': 'none', 'freq': '6m'},
        {'name': 'Mom_6m_Avg_25bps', 'model': 'mom', 'cost': 25, 'stress': 'none', 'freq': '6m'},
        {'name': 'Mom_6m_Avg_50bps', 'model': 'mom', 'cost': 50, 'stress': 'none', 'freq': '6m'},
        {'name': 'Mom_6m_Avg_100bps', 'model': 'mom', 'cost': 100, 'stress': 'none', 'freq': '6m'},
        
        {'name': 'Benchmark_1m_0bps', 'model': 'bmk', 'cost': 0, 'stress': 'none', 'freq': '1m'},
        {'name': 'Benchmark_1m_25bps', 'model': 'bmk', 'cost': 25, 'stress': 'none', 'freq': '1m'},
        {'name': 'Benchmark_1m_50bps', 'model': 'bmk', 'cost': 50, 'stress': 'none', 'freq': '1m'},
        {'name': 'Benchmark_1m_100bps', 'model': 'bmk', 'cost': 100, 'stress': 'none', 'freq': '1m'},
        {'name': 'Benchmark_6m_Avg_0bps', 'model': 'bmk', 'cost': 0, 'stress': 'none', 'freq': '6m'},
        {'name': 'Benchmark_6m_Avg_25bps', 'model': 'bmk', 'cost': 25, 'stress': 'none', 'freq': '6m'},
        {'name': 'Benchmark_6m_Avg_50bps', 'model': 'bmk', 'cost': 50, 'stress': 'none', 'freq': '6m'},
        {'name': 'Benchmark_6m_Avg_100bps', 'model': 'bmk', 'cost': 100, 'stress': 'none', 'freq': '6m'},
        
        {'name': 'HistGBM_1m_Exclusion', 'model': 'hgb', 'cost': 50, 'stress': 'exclusion', 'freq': '1m'},
        {'name': 'HistGBM_1m_CatWorst', 'model': 'hgb', 'cost': 50, 'stress': 'cat_worst', 'freq': '1m'},
        {'name': 'HistGBM_1m_Cat10th', 'model': 'hgb', 'cost': 50, 'stress': 'cat_10th', 'freq': '1m'},
        {'name': 'HistGBM_1m_-20pct', 'model': 'hgb', 'cost': 50, 'stress': '-20pct', 'freq': '1m'},
        {'name': 'HistGBM_1m_-50pct', 'model': 'hgb', 'cost': 50, 'stress': '-50pct', 'freq': '1m'},
        {'name': 'HistGBM_1m_-100pct', 'model': 'hgb', 'cost': 50, 'stress': '-100pct', 'freq': '1m'}
    ]
    
    trackers = {}
    for s in scenarios:
        trackers[s['name']] = {
            'port': init_portfolio(), 
            'returns': [], 
            'gross_returns': [],
            'turnover': [],
            'cost_paid': 0.0,
            'targets_queue': deque(maxlen=6) if s['freq'] == '6m' else deque(maxlen=1),
            'suspended_stats': [],
            'suspension_start_dates': {},
            'unique_schemes_suspended': set(),
            'total_suspension_events': 0,
            'peak_drawdown': 0.0,
            'max_value': 1.0,
            'current_value': 1.0
        }
            
    auc_reconciliation = []
    
    latest_6m_auc_date = None
    latest_6m_target_end = None
    
    latest_live_signal_date = None
    latest_live_execution_date = None
    
    latest_completed_signal_date = None
    latest_completed_execution_start_date = None
    latest_completed_return_end_date = None
    
    latest_pending_end = None
    
    np.random.seed(42)
    
    for block_id, t_start in enumerate(test_starts):
        t_end = t_start + pd.DateOffset(months=6)
        if block_id == len(test_starts) - 1:
            t_end = dates[-1] + pd.Timedelta(days=1)
        train_mask = (eval_df['target_end_date'] < t_start) & eval_df['target'].notna()
        train_df = eval_df[train_mask]
        
        hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
        hgb.fit(train_df[features_full], train_df['target'])
        
        block_months = [d for d in dates if t_start <= d < t_end]
        
        for m_start in block_months:
            intended_purchase = pd.Timestamp(m_start) + pd.Timedelta(days=1)
            intended_return_end = pd.Timestamp(m_start) + pd.DateOffset(months=1) + pd.Timedelta(days=1)
            intended_6m_end = pd.Timestamp(m_start) + pd.DateOffset(months=6)
            
            is_completed = (intended_return_end <= raw_nav_end_date)
            
            if not is_completed:
                latest_pending_end = intended_return_end
                latest_live_signal_date = pd.Timestamp(m_start)
                latest_live_execution_date = intended_purchase
                continue
                
            m_mask = (eval_df['date'] == m_start)
            m_df = eval_df[m_mask].copy()
            if len(m_df) == 0: continue
            
            latest_completed_signal_date = pd.Timestamp(m_start)
            latest_completed_execution_start_date = intended_purchase
            latest_completed_return_end_date = intended_return_end
            
            m_df['score_hgb'] = hgb.predict_proba(m_df[features_full])[:, 1]
            m_df['score_mom'] = m_df.groupby('category')['excess_ret_12m'].rank(pct=True, method='average')
            
            if intended_6m_end <= raw_nav_end_date:
                latest_6m_auc_date = pd.Timestamp(m_start)
                latest_6m_target_end = intended_6m_end
                b_eval = m_df[m_df['target'].notna()]
                if len(np.unique(b_eval['target'])) > 1:
                    hgb_auc = roc_auc_score(b_eval['target'], b_eval['score_hgb'])
                    mom_auc = roc_auc_score(b_eval['target'], b_eval['score_mom'])
                    auc_reconciliation.append({
                        'Block': str(t_start.date()),
                        'prediction_date': str(pd.Timestamp(m_start).date()),
                        'classification_target_horizon': '6-months',
                        'row_count': len(b_eval),
                        'HistGBM_AUC': hgb_auc,
                        'Momentum_AUC': mom_auc,
                    })
                
            all_schemes = m_df['scheme_code'].unique().tolist()
            held_schemes = set()
            for s in scenarios:
                p = trackers[s['name']]['port']
                held_schemes.update(p['tradeable'].keys())
                held_schemes.update(p['suspended'].keys())
            
            all_schemes = list(set(all_schemes).union(held_schemes))
            
            exec_start_df = get_execution_navs(nav, intended_purchase, all_schemes, tolerance_days=7)
            exec_end_df = get_execution_navs(nav, intended_return_end, all_schemes, tolerance_days=7)
            
            start_navs = exec_start_df.set_index('scheme_code')
            end_navs = exec_end_df.set_index('scheme_code')
            
            rets = {}
            for sc in all_schemes:
                sn = start_navs.loc[sc, 'nav'] if sc in start_navs.index else np.nan
                en = end_navs.loc[sc, 'nav'] if sc in end_navs.index else np.nan
                if pd.notnull(sn) and pd.notnull(en):
                    rets[sc] = (en - sn) / sn
                    
            m_df['ret_1m'] = m_df['scheme_code'].map(rets)
            
            cats = m_df['category'].unique()
            cat_weight = 1.0 / len(cats)
            targets = {'hgb': {}, 'mom': {}, 'bmk': {}}
            
            cat_worst = {}
            cat_10th = {}
            for c in cats:
                c_df = m_df[m_df['category'] == c].copy()
                if len(c_df) < 5: continue
                
                c_min = np.nanmin(c_df['ret_1m']) if not c_df['ret_1m'].isna().all() else 0.0
                cat_worst[c] = c_min
                c_10th = np.nanquantile(c_df['ret_1m'], 0.1) if not c_df['ret_1m'].isna().all() else 0.0
                cat_10th[c] = c_10th
                
                fw = cat_weight / len(c_df)
                for sc in c_df['scheme_code']: targets['bmk'][sc] = fw
                
                c_df['q_hgb'] = pd.qcut(c_df['score_hgb'] + np.random.normal(0, 1e-8, len(c_df)), 5, labels=[1,2,3,4,5])
                q5 = c_df[c_df['q_hgb'] == 5]
                fw = cat_weight / len(q5)
                for sc in q5['scheme_code']: targets['hgb'][sc] = fw
                    
                c_df['q_mom'] = pd.qcut(c_df['score_mom'] + np.random.normal(0, 1e-8, len(c_df)), 5, labels=[1,2,3,4,5])
                q5 = c_df[c_df['q_mom'] == 5]
                fw = cat_weight / len(q5)
                for sc in q5['scheme_code']: targets['mom'][sc] = fw
                    
            global_worst = min(cat_worst.values()) if cat_worst else 0.0
            global_10th = min(cat_10th.values()) if cat_10th else 0.0
            
            for s in scenarios:
                t = trackers[s['name']]
                p = t['port']
                
                t['targets_queue'].append(targets[s['model']])
                
                agg_target = {}
                num_vintages = len(t['targets_queue'])
                for v_targets in t['targets_queue']:
                    for sc, w in v_targets.items():
                        agg_target[sc] = agg_target.get(sc, 0.0) + (w / num_vintages)
                        
                stress_map = {
                    'cat_worst': global_worst, 
                    'cat_10th': global_10th, 
                    'exclusion': 0.0,
                    '-20pct': -0.20,
                    '-50pct': -0.50,
                    '-100pct': -1.00
                }
                
                if s['stress'] != 'none':
                    stress_val = stress_map.get(s['stress'], global_worst)
                    for sc in list(p['suspended'].keys()):
                        last_date = last_nav_dates.get(sc)
                        if last_date and last_date < intended_return_end:
                            w_exposed = p['suspended'][sc]
                            loss = w_exposed * abs(stress_val) if stress_val < 0 else 0
                            
                            p['cash'] += w_exposed * (1 + stress_val)
                            del p['suspended'][sc]
                            if sc in t['suspension_start_dates']: del t['suspension_start_dates'][sc]
                            
                            total_remaining = 1.0 - loss
                            if total_remaining > 0:
                                p['cash'] /= total_remaining
                                for k in p['tradeable']: p['tradeable'][k] /= total_remaining
                                for k in p['suspended']: p['suspended'][k] /= total_remaining
                            
                            if len(t['returns']) > 0:
                                t['returns'][-1] -= loss
                                t['gross_returns'][-1] -= loss
                                t['current_value'] *= (1.0 - loss)

                for sc in p['suspended']:
                    if sc not in t['suspension_start_dates']:
                        t['suspension_start_dates'][sc] = intended_purchase
                        t['unique_schemes_suspended'].add(sc)
                        t['total_suspension_events'] += 1
                        
                current_date = intended_return_end
                ages_days = [(current_date - sd).days for sd in t['suspension_start_dates'].values()]
                
                current_w = {k: v for k, v in p['tradeable'].items()}
                total_tvr = 0.0
                all_keys = set(current_w.keys()).union(set(agg_target.keys()))
                for k in all_keys:
                    total_tvr += abs(agg_target.get(k, 0.0) - current_w.get(k, 0.0))
                one_way = total_tvr / 2.0
                t['turnover'].append(one_way)
                
                cost_deducted = one_way * (s['cost'] / 10000.0)
                t['cost_paid'] += (cost_deducted * t['current_value'])

                rebalance(p, agg_target, cost_bps=s['cost'])
                gross, net = step_forward(p, rets)
                t['returns'].append(net)
                t['gross_returns'].append(gross)
                t['current_value'] *= (1 + net)
                
                if t['current_value'] > t['max_value']:
                    t['max_value'] = t['current_value']
                drawdown = 1.0 - (t['current_value'] / t['max_value'])
                if drawdown > t['peak_drawdown']:
                    t['peak_drawdown'] = drawdown
                
                for sc in list(t['suspension_start_dates'].keys()):
                    if sc not in p['suspended']:
                        del t['suspension_start_dates'][sc]
                        
                susp_weight = sum(p['suspended'].values())
                t['suspended_stats'].append({
                    'date': str(current_date.date()),
                    'weight': susp_weight,
                    'count': len(p['suspended']),
                    'avg_age_days': np.mean(ages_days) if ages_days else 0,
                    'max_age_days': max(ages_days) if ages_days else 0
                })
        
    date_table = [
        {"Metric": "data_extraction_timestamp", "Date": str(data_extraction_timestamp.date())},
        {"Metric": "maximum_observed_NAV_date", "Date": str(raw_nav_end_date.date())},
        {"Metric": "latest_6m_AUC_prediction_date", "Date": str(latest_6m_auc_date.date()) if latest_6m_auc_date else "None"},
        {"Metric": "latest_6m_target_end_NAV_date", "Date": str(latest_6m_target_end.date()) if latest_6m_target_end else "None"},
        {"Metric": "latest_completed_signal_date", "Date": str(latest_completed_signal_date.date()) if latest_completed_signal_date else "None"},
        {"Metric": "latest_completed_execution_start_date", "Date": str(latest_completed_execution_start_date.date()) if latest_completed_execution_start_date else "None"},
        {"Metric": "latest_completed_return_end_date", "Date": str(latest_completed_return_end_date.date()) if latest_completed_return_end_date else "None"},
        {"Metric": "latest_live_signal_date", "Date": str(latest_live_signal_date.date()) if latest_live_signal_date else "None"},
        {"Metric": "latest_live_execution_date", "Date": str(latest_live_execution_date.date()) if latest_live_execution_date else "None"},
        {"Metric": "latest_pending_return_end_date", "Date": str(latest_pending_end.date()) if latest_pending_end else "None"}
    ]
    pd.DataFrame(date_table).to_csv(f"{out_dir}/dates.csv", index=False)
    
    auc_df = pd.DataFrame(auc_reconciliation)
    if not auc_df.empty:
        auc_df.to_csv(f"{out_dir}/auc_reconciliation.csv", index=False)
    
    summary_metrics = []
    for s in scenarios:
        t = trackers[s['name']]
        gross_cum = np.prod([1+x for x in t['gross_returns']]) - 1
        net_cum = np.prod([1+x for x in t['returns']]) - 1
        cum_turnover = np.sum(t['turnover'])
        avg_turnover = np.mean(t['turnover']) if t['turnover'] else 0
        
        gross_to_net_drag = gross_cum - net_cum
        
        summary_metrics.append({
            'strategy': s['name'],
            'model': s['model'],
            'freq': s['freq'],
            'cost_bps': s['cost'],
            'gross_cumulative_return': gross_cum,
            'net_cumulative_return': net_cum,
            'cumulative_one_way_turnover': cum_turnover,
            'average_monthly_one_way_turnover': avg_turnover,
            'total_cost_paid_as_pct_of_initial_portfolio_value': t['cost_paid'],
            'gross_to_net_cumulative_return_drag': gross_to_net_drag,
            'maximum_drawdown': t['peak_drawdown'],
            'total_suspension_events': t['total_suspension_events'],
            'number_of_unique_schemes_suspended': len(t['unique_schemes_suspended']),
            'maximum_concurrent_suspended_positions': max([st['count'] for st in t['suspended_stats']]) if t['suspended_stats'] else 0,
            'average_suspension_age_days': np.mean([st['avg_age_days'] for st in t['suspended_stats']]) if t['suspended_stats'] else 0,
            'maximum_suspension_age_days': max([st['max_age_days'] for st in t['suspended_stats']]) if t['suspended_stats'] else 0,
            'average_suspended_weight': np.mean([st['weight'] for st in t['suspended_stats']]) if t['suspended_stats'] else 0,
            'maximum_suspended_weight': max([st['weight'] for st in t['suspended_stats']]) if t['suspended_stats'] else 0
        })
        
    pd.DataFrame(summary_metrics).to_csv(f"{out_dir}/summary_metrics.csv", index=False)
    
if __name__ == "__main__":
    run_evidence_appendix()
