"""Trailing risk features; optional OHLCV/cross-asset features are never fabricated."""
import numpy as np
import pandas as pd

def make_features_v2(prices,include_optional=False):
    p=prices.equity;r=p.pct_change(fill_method=None)
    f=pd.DataFrame(index=p.index)
    for w in [1,5,10,20,60,120]:f[f'return_{w}']=p.pct_change(w,fill_method=None)
    for w in [5,10,20,60,120]:
        f[f'vol_{w}']=r.rolling(w).std()*np.sqrt(252)
    f['ewma_vol']=np.sqrt(r.pow(2).ewm(alpha=.06,adjust=False,min_periods=60).mean()*252)
    f['downside_vol_20']=np.sqrt(r.clip(upper=0).pow(2).rolling(20).mean()*252)
    f['upside_vol_20']=np.sqrt(r.clip(lower=0).pow(2).rolling(20).mean()*252)
    f['downside_fraction']=f.downside_vol_20/(f.downside_vol_20+f.upside_vol_20).replace(0,np.nan)
    f['vol_ratio_5_20']=f.vol_5/f.vol_20.replace(0,np.nan)
    f['vol_ratio_20_60']=f.vol_20/f.vol_60.replace(0,np.nan)
    f['vol_acceleration']=f.vol_20-f.vol_20.shift(5)
    f['vol_of_vol']=f.vol_20.rolling(20).std()
    f['drawdown_60']=p/p.rolling(60).max()-1
    f['drawdown_252']=p/p.rolling(252).max()-1
    f['distance_ma_20']=p/p.rolling(20).mean()-1
    f['distance_ma_200']=p/p.rolling(200).mean()-1
    f['skew_60']=r.rolling(60).skew()
    f['kurtosis_60']=r.rolling(60).kurt()
    f['worst_return_20']=r.rolling(20).min()
    f['best_return_20']=r.rolling(20).max()
    f['negative_fraction_20']=(r<0).rolling(20).mean()
    f['absolute_return_1']=r.abs()
    f['return_autocorr_20']=r.rolling(20).corr(r.shift(1))
    f['absolute_autocorr_20']=r.abs().rolling(20).corr(r.abs().shift(1))
    if include_optional:
        for asset in ['bonds','gold','oil','dollar']:
            if asset in prices:
                ar=prices[asset].pct_change(fill_method=None)
                f[f'{asset}_return_20']=prices[asset].pct_change(20,fill_method=None)
                f[f'corr_equity_{asset}']=r.rolling(60).corr(ar)
        if 'vix' in prices:
            f['vix']=prices.vix;f['vix_change_5']=prices.vix.pct_change(5,fill_method=None)
        if 'equity_volume' in prices:
            v=prices.equity_volume
            f['relative_volume']=v/v.rolling(20).mean().replace(0,np.nan)
            f['dollar_volume_20']=(v*p).rolling(20).mean()
        if {'equity_high','equity_low'}.issubset(prices.columns):
            f['range_vol_20']=np.sqrt(np.log(prices.equity_high/prices.equity_low).pow(2).rolling(20).mean()*252/(4*np.log(2)))
        if 'equity_open' in prices:f['overnight_gap']=prices.equity_open/p.shift(1)-1
    return f.replace([np.inf,-np.inf],np.nan).dropna()
