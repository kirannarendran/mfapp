import pandas as pd
import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def run_simulation():
    print("Loading datasets...")
    df = pd.read_parquet("data/processed/model_dataset.parquet")
    nav = pd.read_parquet("data/raw/nav_history.parquet")
    
    df['date'] = pd.to_datetime(df['date'])
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    nav['date'] = pd.to_datetime(nav['date'])
    
    features_full = [
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
        'mom_12_1', 'mom_6_1', 'vol_12m', 'vol_6m',
        'dist_ma_6m', 'dist_ma_12m',
        'rank_ret_6m', 'rank_ret_12m', 'rank_vol_12m', 'excess_ret_6m'
    ]
    features_no_mom = [f for f in features_full if f not in ['mom_12_1', 'mom_6_1']]
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features_full)
    df = df.sort_values('date')
    df_dates = np.sort(df['date'].unique())
    
    # Pre-calculate 1-month returns
    nav = nav.sort_values(['scheme_code', 'date'])
    nav_returns = {}
    for i in range(len(df_dates)-1):
        d_start = df_dates[i]
        d_end = df_dates[i+1]
        start_navs = nav[nav['date'] == d_start].set_index('scheme_code')['nav']
        end_navs = nav[nav['date'] == d_end].set_index('scheme_code')['nav']
        common = start_navs.index.intersection(end_navs.index)
        rets = (end_navs[common] - start_navs[common]) / start_navs[common]
        nav_returns[pd.Timestamp(d_start)] = rets.to_dict()

    test_starts = pd.to_datetime(['2024-07-31', '2025-01-31', '2025-07-31'])
    block_metrics = []
    
    # We will track portfolio weights
    def init_portfolio(): return {'weights': {}, 'cash': 1.0}
    
    def rebalance(port, target_weights, cost_bps=0):
        # target_weights is a dict {scheme_code: weight}
        drifted = port['weights']
        all_assets = set(drifted.keys()).union(target_weights.keys())
        
        turnover_sum = 0.0
        for a in all_assets:
            w_d = drifted.get(a, 0.0)
            w_t = target_weights.get(a, 0.0)
            turnover_sum += abs(w_t - w_d)
            
        turnover = 0.5 * turnover_sum
        cost = turnover * (cost_bps / 10000.0)
        
        # New port
        port['weights'] = target_weights.copy()
        port['turnover'] = turnover
        port['cost'] = cost
        return port

    def step_forward(port, rets_dict, missing_scenario='cat_worst', cat_worst_ret=0.0):
        port_ret = 0.0
        new_weights = {}
        for a, w in port['weights'].items():
            if a in rets_dict:
                r = rets_dict[a]
            else:
                # Missing fund resolution
                if missing_scenario == 'cat_worst': r = cat_worst_ret
                elif missing_scenario == '-20pct': r = -0.20
                elif missing_scenario == '-50pct': r = -0.50
                elif missing_scenario == '-100pct': r = -1.00
                elif missing_scenario == 'exclusion': r = 0.0 # effectively cash
                else: r = cat_worst_ret
                
            port_ret += w * r
            new_weights[a] = w * (1 + r)
            
        # Normalize drifted weights
        if port_ret > -1.0:
            for a in new_weights:
                new_weights[a] /= (1 + port_ret)
        else:
            new_weights = {}
            
        port['weights'] = new_weights
        return port_ret
        
    scenarios = [
        {'name': 'HistGBM_0bps', 'model': 'hgb', 'cost': 0, 'missing': 'cat_worst'},
        {'name': 'HistGBM_50bps', 'model': 'hgb', 'cost': 50, 'missing': 'cat_worst'},
        {'name': 'HistGBM_Missing50', 'model': 'hgb', 'cost': 50, 'missing': '-50pct'},
        {'name': 'Mom12_0bps', 'model': 'mom', 'cost': 0, 'missing': 'cat_worst'},
        {'name': 'Mom12_50bps', 'model': 'mom', 'cost': 50, 'missing': 'cat_worst'},
        {'name': 'Benchmark_0bps', 'model': 'bmk', 'cost': 0, 'missing': 'cat_worst'},
    ]
    
    trackers = {s['name']: {'port': init_portfolio(), 'returns': []} for s in scenarios}
    
    for block_id, t_start in enumerate(test_starts):
        t_end = t_start + pd.DateOffset(months=6)
        
        train_mask = (df['target_end_date'] < t_start) & df['target'].notna()
        train_df = df[train_mask]
        
        hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
        hgb.fit(train_df[features_full], train_df['target'])
        
        hgb_no = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=6, random_state=42)
        hgb_no.fit(train_df[features_no_mom], train_df['target'])
        
        block_months = [d for d in df_dates if t_start <= d < t_end]
        b_aucs = {'hgb': [], 'mom': [], 'no_mom': []}
        
        for m_start in block_months:
            m_mask = (df['date'] == m_start)
            m_df = df[m_mask].copy()
            if len(m_df) == 0: continue
            
            m_df['score_hgb'] = hgb.predict_proba(m_df[features_full])[:, 1]
            m_df['score_hgb_nomom'] = hgb_no.predict_proba(m_df[features_no_mom])[:, 1]
            m_df['score_mom'] = m_df['excess_ret_12m'].rank(pct=True).values
            
            m_rets = nav_returns.get(pd.Timestamp(m_start), {})
            m_df['ret_1m'] = m_df['scheme_code'].map(m_rets)
            
            # AUC
            b_eval = m_df[m_df['target'].notna()]
            if len(np.unique(b_eval['target'])) > 1:
                b_aucs['hgb'].append(roc_auc_score(b_eval['target'], b_eval['score_hgb']))
                b_aucs['mom'].append(roc_auc_score(b_eval['target'], b_eval['score_mom']))
                b_aucs['no_mom'].append(roc_auc_score(b_eval['target'], b_eval['score_hgb_nomom']))
                
            # Construct target portfolios
            cats = m_df['category'].unique()
            cat_weight = 1.0 / len(cats)
            
            targets = {'hgb': {}, 'mom': {}, 'bmk': {}}
            cat_worst = {}
            
            for c in cats:
                c_df = m_df[m_df['category'] == c].copy()
                if len(c_df) < 5: continue
                
                cat_worst[c] = c_df['ret_1m'].min()
                
                # Bmk: equal weight all
                fw = cat_weight / len(c_df)
                for sc in c_df['scheme_code']: targets['bmk'][sc] = fw
                
                # HGB Q5
                c_df['q_hgb'] = pd.qcut(c_df['score_hgb'] + np.random.normal(0, 1e-8, len(c_df)), 5, labels=[1,2,3,4,5])
                q5_hgb = c_df[c_df['q_hgb'] == 5]
                fw = cat_weight / len(q5_hgb)
                for sc in q5_hgb['scheme_code']: targets['hgb'][sc] = fw
                    
                # Mom Q5
                c_df['q_mom'] = pd.qcut(c_df['score_mom'] + np.random.normal(0, 1e-8, len(c_df)), 5, labels=[1,2,3,4,5])
                q5_mom = c_df[c_df['q_mom'] == 5]
                fw = cat_weight / len(q5_mom)
                for sc in q5_mom['scheme_code']: targets['mom'][sc] = fw
                    
            # Rebalance and step forward
            global_worst = min(cat_worst.values()) if cat_worst else 0
            
            for s in scenarios:
                t = trackers[s['name']]
                rebalance(t['port'], targets[s['model']], cost_bps=s['cost'])
                r = step_forward(t['port'], m_rets, missing_scenario=s['missing'], cat_worst_ret=global_worst)
                t['returns'].append(r - t['port']['cost'])
                
        block_metrics.append({
            'block': str(t_start.date()),
            'auc_hgb': float(np.nanmean(b_aucs['hgb'])),
            'auc_mom': float(np.nanmean(b_aucs['mom'])),
            'auc_no_mom': float(np.nanmean(b_aucs['no_mom']))
        })
        
    print("\\nEconomic Returns Summary (Annualized):")
    res = {'blocks': block_metrics, 'economic': {}}
    for s in scenarios:
        arr = np.array(trackers[s['name']]['returns'])
        ann = np.nanmean(arr) * 12
        print(f"{s['name']}: {ann*100:.2f}%")
        res['economic'][s['name']] = {'ann_ret': float(ann)}
        
    with open("data/processed/final_validation_complete.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_simulation()
