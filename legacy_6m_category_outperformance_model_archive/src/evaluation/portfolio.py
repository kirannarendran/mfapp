import numpy as np
import pandas as pd

def init_portfolio():
    return {
        'cash': 1.0,
        'tradeable': {},
        'suspended': {},
        'turnover': 0.0,
        'cost': 0.0
    }

def get_execution_navs(nav_df, intended_date, scheme_codes, tolerance_days=7):
    """
    Select the earliest available NAV on or after intended_date, up to tolerance_days.
    Returns DataFrame: scheme_code, actual_nav_date, nav, days_delayed, lookup_status
    """
    end_date = intended_date + pd.Timedelta(days=tolerance_days)
    mask = (nav_df['scheme_code'].isin(scheme_codes)) & (nav_df['date'] >= intended_date) & (nav_df['date'] <= end_date)
    subset = nav_df[mask].sort_values('date')
    
    # Get first available date per scheme
    first_navs = subset.groupby('scheme_code').first().reset_index()
    
    results = []
    for sc in scheme_codes:
        row = first_navs[first_navs['scheme_code'] == sc]
        if len(row) > 0:
            actual_date = row['date'].iloc[0]
            nav = row['nav'].iloc[0]
            days_delayed = (actual_date - intended_date).days
            results.append({
                'scheme_code': sc,
                'actual_nav_date': actual_date,
                'nav': nav,
                'days_delayed': days_delayed,
                'lookup_status': 'Resolved'
            })
        else:
            results.append({
                'scheme_code': sc,
                'actual_nav_date': None,
                'nav': np.nan,
                'days_delayed': np.nan,
                'lookup_status': 'Unresolved'
            })
            
    return pd.DataFrame(results)

def rebalance(port, target_tradeable_weights, cost_bps=0):
    """
    Rebalances the tradeable portion of the portfolio. 
    Suspended positions cannot be sold or reallocated.
    Target tradeable weights are rescaled so that sum(target_tradeable) + sum(suspended) = 1.0 - cash (if cash target).
    Assume target_tradeable_weights sum to 1.0 * (1.0 - sum(suspended)).
    Wait, the simulation rule says: "Suspended positions cannot be sold or reallocated until a valid NAV is available."
    So capital locked in suspended remains locked.
    The rebalance allocates ONLY the remaining available capital (cash + sum(tradeable)).
    """
    suspended_val = sum(port['suspended'].values())
    available_capital = port['cash'] + sum(port['tradeable'].values())
    
    # Target weights typically sum to 1.0 (desired total portfolio). 
    # We must restrict them to available_capital.
    # If target wants 100% in asset A, but available is 0.9, asset A gets 0.9.
    target_sum = sum(target_tradeable_weights.values())
    
    scaled_targets = {}
    if target_sum > 0:
        for k, v in target_tradeable_weights.items():
            scaled_targets[k] = v * available_capital
            
    # Calculate turnover
    # Turnover is the absolute difference in the available portion.
    all_assets = set(port['tradeable'].keys()).union(scaled_targets.keys())
    turnover_sum = 0.0
    for a in all_assets:
        w_old = port['tradeable'].get(a, 0.0)
        w_new = scaled_targets.get(a, 0.0)
        turnover_sum += abs(w_new - w_old)
        
    cash_old = port['cash']
    cash_new = available_capital - sum(scaled_targets.values())
    turnover_sum += abs(cash_new - cash_old)
    
    turnover = 0.5 * turnover_sum
    cost = turnover * (cost_bps / 10000.0)
    
    port['tradeable'] = scaled_targets
    port['cash'] = cash_new
    port['turnover'] = turnover
    port['cost'] = cost
    
    # Assert constraint
    total = port['cash'] + sum(port['tradeable'].values()) + sum(port['suspended'].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Portfolio weights sum to {total}, expected 1.0")
        
    return port

def step_forward(port, rets_dict, cash_return=0.0):
    """
    Steps the portfolio forward by one period using realized returns.
    Returns gross_growth_factor and deducts cost at the start.
    If a tradeable asset has no return (unresolved), it becomes suspended.
    If a suspended asset gets a return, it becomes tradeable again.
    """
    new_tradeable = {}
    new_suspended = {}
    
    gross_growth_factor = port['cash'] * (1 + cash_return)
    new_cash = port['cash'] * (1 + cash_return)
    
    for a, w in port['tradeable'].items():
        if a in rets_dict and not pd.isna(rets_dict[a]):
            val = w * (1 + rets_dict[a])
            new_tradeable[a] = new_tradeable.get(a, 0.0) + val
            gross_growth_factor += val
        else:
            new_suspended[a] = new_suspended.get(a, 0.0) + w
            gross_growth_factor += w # No return applied while suspended during the live run
            
    for a, w in port['suspended'].items():
        if a in rets_dict and not pd.isna(rets_dict[a]):
            val = w * (1 + rets_dict[a])
            new_tradeable[a] = new_tradeable.get(a, 0.0) + val
            gross_growth_factor += val
        else:
            new_suspended[a] = new_suspended.get(a, 0.0) + w
            gross_growth_factor += w
            
    gross_return = gross_growth_factor - 1.0
    
    # Normalize drifted weights
    if gross_growth_factor > 1e-8:
        port['cash'] = new_cash / gross_growth_factor
        port['tradeable'] = {k: v / gross_growth_factor for k, v in new_tradeable.items()}
        port['suspended'] = {k: v / gross_growth_factor for k, v in new_suspended.items()}
    else:
        port['cash'] = 1.0
        port['tradeable'] = {}
        port['suspended'] = {}
        
    # Deduct transaction cost from net return
    net_return = gross_return - port['cost']
    
    total = port['cash'] + sum(port['tradeable'].values()) + sum(port['suspended'].values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Normalized weights do not sum to 1.0")
        
    return gross_return, net_return
