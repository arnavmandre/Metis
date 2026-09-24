import numpy as np
import pandas as pd

def make_features(prices, windows=(5,20,60)):
    r = prices.pct_change(fill_method=None)
    x = pd.DataFrame(index=prices.index)
    x['return_1'] = r.equity
    for w in windows:
        # Cumulative returns already express momentum; avoid duplicate columns.
        x[f'return_{w}'] = prices.equity.pct_change(w, fill_method=None)
        x[f'vol_{w}'] = r.equity.rolling(w).std()*np.sqrt(252)
    x['vol_ratio'] = x[f'vol_{windows[0]}']/x[f'vol_{windows[-1]}']
    x['drawdown'] = prices.equity/prices.equity.rolling(252, min_periods=60).max()-1
    for c in prices.columns:
        if c != 'equity':
            x[f'{c}_change'] = r[c]
        if c in ('bonds','gold','oil'):
            x[f'corr_{c}'] = r.equity.rolling(60).corr(r[c])
    if 'vix' in prices:
        x['vix_level'] = prices.vix
    return x.replace([np.inf,-np.inf], np.nan).dropna()
