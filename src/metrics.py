import numpy as np

def summarize(bt):
    r = bt.net
    nav = (1+r).cumprod()
    annual = nav.iloc[-1]**(252/len(r))-1
    vol = r.std()*np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252)
    dd = (nav/nav.cummax().clip(lower=1)-1).min()
    return {'annualized_return':annual, 'annualized_volatility':vol,
            'sharpe_rf_zero':r.mean()*252/vol if vol else np.nan,
            'sortino_target_zero':r.mean()*252/downside if downside else np.nan,
            'max_drawdown':dd, 'calmar':annual/abs(dd) if dd else np.nan,
            'annual_turnover':bt.turnover.mean()*252,
            'sum_cost_fractions':bt.cost_fraction.sum(), 'ending_nav':nav.iloc[-1]}
