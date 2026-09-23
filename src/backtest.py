"""Close-to-close execution, full L1 traded notional, drift-aware turnover."""
import numpy as np
import pandas as pd

def run_backtest(prices, equity_targets, cost=.001):
    if not 0 <= cost < .1:
        raise ValueError('Invalid cost rate')
    targets = equity_targets.reindex(prices.index)
    if targets.isna().any() or not targets.between(0,1).all():
        raise ValueError('Targets must cover every date and lie in [0,1]')
    returns = prices[['equity','bonds']].pct_change(fill_method=None).fillna(0)
    held = np.array([0., 0.]); cash = 1.
    records = []
    for i, date in enumerate(prices.index):
        # Decision at close t-1, execution at close t, first earning return t+1.
        gross = float(held @ returns.iloc[i].to_numpy())
        drift = held*(1+returns.iloc[i].to_numpy())/(1+gross)
        drift_cash = cash/(1+gross)
        if i == 0:
            target = drift
            turnover = 0.
        else:
            eq = float(targets.iloc[i-1]); target = np.array([eq, 1-eq])
            # Exact post-fee NAV target: solve fee = c * |target*(NAV-fee)-holdings|.
            fee = 0.
            for _ in range(30):
                fee = cost*np.abs(target*(1-fee)-drift).sum()
            turnover = float(np.abs(target*(1-fee)-drift).sum())
            drift_cash = 0.
        fee = cost*turnover
        net = (1+gross)*(1-fee)-1
        held, cash = target, drift_cash
        records.append((gross, net, turnover, fee, held[0]))
    return pd.DataFrame(records, index=prices.index, columns=['gross','net','turnover','cost_fraction','equity_weight'])
