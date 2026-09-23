"""CSV adapter, optional Yahoo download, and explicitly synthetic smoke data."""
from pathlib import Path
import numpy as np
import pandas as pd

TICKERS = {'equity':'SPY', 'bonds':'IEF', 'gold':'GLD', 'oil':'USO', 'dollar':'UUP', 'vix':'^VIX'}

def load_csv(path):
    d = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    if d.index.has_duplicates:
        raise ValueError('Duplicate dates are not permitted')
    required = ['equity', 'bonds']
    if not set(required).issubset(d.columns):
        raise ValueError('CSV requires date, equity, bonds; adjusted positive prices')
    if d[required].isna().any().any() or (d[required] <= 0).any().any():
        raise ValueError('Equity/bond prices must be complete and positive; no price filling')
    if not np.isfinite(d.to_numpy(dtype=float)).all() or (d <= 0).any().any():
        raise ValueError('Use complete, finite, positive observations; inspect missing sessions first')
    return d

def download(path, start='2005-01-01', end=None):
    import yfinance as yf
    series = {}
    for name, ticker in TICKERS.items():
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f'No data returned for {ticker}')
        series[name] = raw['Close'].squeeze().rename(name)
    d = pd.concat(series.values(), axis=1).dropna()
    d.index.name = 'date'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path)
    Path(str(path)+'.source.txt').write_text('Yahoo Finance via yfinance; adjusted Close; ETFs '+str(TICKERS)+'\nCommon complete sessions; inspect inception dates and data licensing before redistribution.\n')
    return d

def synthetic(n=1800, seed=42):
    """Software demonstration only. Not evidence about actual markets."""
    rng = np.random.default_rng(seed)
    states = np.zeros(n, int)
    for i in range(1, n):
        states[i] = states[i-1] if rng.random()<.975 else rng.integers(0, 3)
    vol = np.array([.006, .014, .03])[states]
    r = rng.normal(0, 1, (n, 5))*vol[:, None]*np.array([1, .3, .65, 1.3, .25])
    r[:, 0] += np.array([.0005, 0, -.001])[states]
    d = pd.DataFrame(100*np.exp(np.cumsum(r, axis=0)), columns=['equity','bonds','gold','oil','dollar'], index=pd.bdate_range('2015-01-01', periods=n))
    d['vix'] = 10 + 650*vol + rng.uniform(0, 3, n)
    d.index.name = 'date'
    return d
