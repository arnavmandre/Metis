"""Read-only paper signal from a frozen artifact; never submits orders."""
import argparse,json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from src.backtest import validate_prices
from src.reliability import equity_features,restore_hmm,logistic_predict,sha

def paper_signal(prices,artifact,as_of,max_stale_days=5,z_limit=8.):
    validate_prices(prices)
    as_of=pd.Timestamp(as_of);last=prices.index[-1]
    if last>as_of:raise ValueError('Input contains dates after as-of')
    cutoff=pd.Timestamp(artifact['last_development_date'])
    history=prices.loc[:cutoff]
    if len(history)<252:raise ValueError('Need at least 252 history sessions through model cursor date')
    if history.index[-1]!=cutoff:raise ValueError('Missing model cursor date; cannot continue frozen filter')
    if prices.index.to_series().diff().dt.days.max()>7:raise ValueError('Unexpected gap: review source calendar; never forward-fill silently')
    f=equity_features(prices);f=f.loc[f.index>cutoff]
    if f.empty:raise ValueError('No new observations after frozen model cursor')
    x=(f.to_numpy()-np.array(artifact['scaler_mean']))/np.array(artifact['scaler_scale'])
    probs=restore_hmm(artifact['hmm']).filter(x,np.array(artifact['last_development_probability']))
    stale=(as_of-last).days>max_stale_days;ood=float(np.abs(x[-1]).max())>z_limit
    flags=[]
    if stale:flags.append('stale_data')
    if ood:flags.append('out_of_distribution')
    return {'mode':'paper_research_only','as_of':str(as_of.date()),'data_through':str(last.date()),
            'status':'review_required' if flags else 'paper_signal_available','flags':flags,
            'probability_next_20_sessions_high_vol':float(logistic_predict(artifact['risk_head'],probs[-1:])[0]),
            'state_probabilities':probs[-1].tolist(),'max_abs_standardized_feature':float(np.abs(x[-1]).max()),
            'paper_equity_target':None if flags else float(probs[-1]@np.array(artifact['weights'])),
            'execution':'At earliest the next session close; no order is sent',
            'limitations':'Research-proxy model. A different live instrument/feed needs separate validation; probability is not crash probability.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',required=True);p.add_argument('--model',default='outputs/blind/model.json');p.add_argument('--as-of',default=str(date.today()));a=p.parse_args()
    prices=pd.read_csv(a.csv,index_col='date',parse_dates=['date']);artifact=json.loads(Path(a.model).read_text())
    result=paper_signal(prices,artifact,a.as_of);result['model_sha256']=sha(a.model)
    print(json.dumps(result,indent=2))
