"""Delayed close execution with drift, costs/slippage, rebalance cadence and bands."""
import numpy as np
import pandas as pd

def validate_prices(prices):
    defensive='defensive' if 'defensive' in prices else 'bonds'
    if 'equity' not in prices or defensive not in prices: raise ValueError('Require equity and defensive (or legacy bonds) prices')
    p=prices[['equity',defensive]]
    if len(p)<2 or not p.index.is_monotonic_increasing or p.index.has_duplicates: raise ValueError('Require sorted unique sessions')
    if not isinstance(p.index,pd.DatetimeIndex): raise ValueError('Require DatetimeIndex')
    if not np.isfinite(p.to_numpy(float)).all() or (p<=0).any().any(): raise ValueError('Prices must be finite, positive and complete')
    return p

def run_backtest(prices,equity_targets,cost=.001,slippage_bps=0.,rebalance_every=1,min_trade=0.,buy_and_hold=False,execution_delay=1):
    """Signal t executes at t+delay; first earns return t+delay+1. Both legs incur fees.

    Starts in zero-yield cash. After entry the defensive proxy earns its own return.
    No leverage, terminal liquidation, taxes or nonlinear market-impact model.
    """
    p=validate_prices(prices);rate=cost+slippage_bps/10000.
    if not 0<=cost<.1 or not 0<=slippage_bps<1000 or not 0<=rate<.1: raise ValueError('Invalid costs')
    if rebalance_every<1 or int(rebalance_every)!=rebalance_every or execution_delay<1 or int(execution_delay)!=execution_delay: raise ValueError('Cadence/delay must be positive integer sessions')
    if not 0<=min_trade<=1: raise ValueError('Invalid trade band')
    targets=equity_targets.reindex(p.index)
    if not np.isfinite(targets).all() or not targets.between(0,1).all(): raise ValueError('Invalid or missing targets')
    returns=p.pct_change(fill_method=None).fillna(0).to_numpy()
    held=np.zeros(2);cash=1.;records=[];entered=False
    for i,date in enumerate(p.index):
        gross=float(held@returns[i]);drift=held*(1+returns[i])/(1+gross);cash/=1+gross
        eligible=i>=execution_delay and (i-execution_delay)%rebalance_every==0
        trade=False;turnover=0.;fee=0.;target=drift
        if eligible and (not buy_and_hold or not entered):
            eq=float(targets.iloc[i-execution_delay]);proposed=np.array([eq,1-eq])
            if not entered or abs(eq-drift[0])>=min_trade:
                trade=True;target=proposed
                for _ in range(40):fee=rate*np.abs(target*(1-fee)-drift).sum()
                turnover=float(np.abs(target*(1-fee)-drift).sum());cash=0.;entered=True
        net=(1+gross)*(1-fee)-1;held=target
        records.append((gross,net,turnover,fee,held[0],trade,turnover*cost,turnover*slippage_bps/10000.))
    return pd.DataFrame(records,index=p.index,columns=['gross','net','turnover','cost_fraction','equity_weight','traded','commission_fraction','slippage_fraction'])
