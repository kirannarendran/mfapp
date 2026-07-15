import pandas as pd
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def normalize_scheme_name(name):
    if not isinstance(name, str):
        return ""
        
    name = name.lower()
    # Remove common structural noise
    name = re.sub(r'\(.*?\)', '', name)
    name = name.replace('-', ' ')
    name = name.replace('  ', ' ')
    
    # Remove plan/option text
    removals = ['direct', 'regular', 'plan', 'growth', 'option', 'idcw', 'dividend', 'reinvestment']
    for r in removals:
        name = name.replace(r, '')
        
    return name.strip()

def classify_plan_and_option(name):
    if not isinstance(name, str):
        return "Unclassified", "Unclassified"
        
    name_low = name.lower()
    
    is_direct = 'direct' in name_low
    is_regular = 'regular' in name_low
    
    is_growth = 'growth' in name_low
    is_idcw = 'idcw' in name_low or 'dividend' in name_low or 'div' in name_low
    is_bonus = 'bonus' in name_low
    
    if is_direct and not is_regular:
        plan = "Direct"
    elif is_regular and not is_direct:
        plan = "Regular"
    else:
        plan = "Unclassified"
        
    if is_growth and not is_idcw:
        option = "Growth"
    elif is_idcw and not is_growth:
        option = "IDCW"
    elif is_bonus:
        option = "Bonus"
    else:
        option = "Unclassified"
        
    return plan, option

def map_scheme_identities(funds_df, nav_df, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if funds_df.empty:
        return pd.DataFrame()
        
    df = funds_df.copy()
    
    # Identify types
    df['normalized_underlying_name'] = df['scheme_name'].apply(normalize_scheme_name)
    df['plan_type'], df['option_type'] = zip(*df['scheme_name'].apply(classify_plan_and_option))
    
    ambiguous = df[(df['plan_type'] == 'Unclassified') | (df['option_type'] == 'Unclassified')]
    if not ambiguous.empty:
        ambiguous.to_csv(out_dir / "scheme_name_changes.csv", index=False)
        
    # Match candidate Direct-Regular pairs
    directs = df[(df['plan_type'] == 'Direct') & (df['option_type'] == 'Growth')]
    regulars = df[(df['plan_type'] == 'Regular') & (df['option_type'] == 'Growth')]
    
    pairs = pd.merge(
        directs, regulars, 
        on=['fund_house', 'normalized_underlying_name'], 
        suffixes=('_direct', '_regular')
    )
    
    # To check correlation, we need nav overlap
    results = []
    
    for _, row in pairs.iterrows():
        d_code = row['scheme_code_direct']
        r_code = row['scheme_code_regular']
        
        d_nav = nav_df[nav_df['scheme_code'] == d_code].set_index('parsed_NAV_date')['nav'].pct_change().dropna()
        r_nav = nav_df[nav_df['scheme_code'] == r_code].set_index('parsed_NAV_date')['nav'].pct_change().dropna()
        
        overlap, _ = d_nav.align(r_nav, join='inner')
        
        obs = len(overlap)
        
        if obs > 30:
            corr = overlap.corr(r_nav.loc[overlap.index])
            med_diff = (overlap - r_nav.loc[overlap.index]).median()
            est_drag = med_diff * 252 # approx annualized
            
            if corr > 0.99 and est_drag > 0:
                status = "verified_by_identifiers"
            elif corr > 0.95:
                status = "high_confidence_candidate"
            else:
                status = "ambiguous"
        else:
            corr = None
            med_diff = None
            est_drag = None
            status = "rejected"
            
        results.append({
            'direct_scheme_code': d_code,
            'regular_scheme_code': r_code,
            'direct_ISIN': row.get('isin_direct', 'Unknown'),
            'regular_ISIN': row.get('isin_regular', 'Unknown'),
            'normalized_underlying_name': row['normalized_underlying_name'],
            'overlap_start': overlap.index.min() if obs > 0 else pd.NaT,
            'overlap_end': overlap.index.max() if obs > 0 else pd.NaT,
            'overlap_observations': obs,
            'return_correlation': corr,
            'median_return_difference': med_diff,
            'estimated_annual_expense_drag': est_drag,
            'mapping_confidence': status,
            'mapping_status': status
        })
        
    pairs_df = pd.DataFrame(results)
    if not pairs_df.empty:
        pairs_df.to_csv(out_dir / "direct_regular_overlap.csv", index=False)
        
    return df
