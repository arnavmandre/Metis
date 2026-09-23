"""Explicit forecast target, purging, paired block uncertainty, portable models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,brier_score_loss,log_loss
from .features import make_features
from .hmm_model import GaussianHMM
FEATURES=['return_1','return_5','return_20','return_60','vol_5','vol_20','vol_60','vol_ratio','drawdown']
def equity_features(prices):return make_features(prices[['equity']])[FEATURES]
def future_volatility(prices,horizon=20):
    """Standard deviation of returns t+1,...,t+h; scoring labels, never features."""
    return prices.equity.pct_change(fill_method=None).rolling(horizon).std().shift(-horizon)*np.sqrt(252)
def label_end_dates(index,horizon):return pd.Series(index,index=index).shift(-horizon)
def purged_mask(index,end_dates,start,end):
    return (index>=pd.Timestamp(start))&(index<=pd.Timestamp(end))&(end_dates.reindex(index)<=pd.Timestamp(end)).to_numpy()
def classifier_metrics(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=.5;both=len(np.unique(y))==2
    return {'n':len(y),'positive_rate':float(y.mean()),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred) if both else np.nan,'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p) if both else np.nan,'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1])}
def block_indices(n,block,rng):
    block=min(block,n);starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
    return np.concatenate([np.arange(i,i+block) for i in starts])[:n]
def uncertainty(y,probabilities,reps=1000,block=60,seed=1907):
    rng=np.random.default_rng(seed);rows=[]
    for rep in range(reps):
        ix=block_indices(len(y),block,rng)
        values={name:classifier_metrics(np.asarray(y)[ix],np.asarray(p)[ix]) for name,p in probabilities.items()}
        for name,v in values.items():rows.append({'replicate':rep,'model':name,**{k:v[k] for k in ('accuracy','balanced_accuracy','roc_auc','brier')}})
        for name in probabilities:
            if name!='hmm':rows.append({'replicate':rep,'model':'hmm_minus_'+name,**{k:values['hmm'][k]-values[name][k] for k in ('accuracy','balanced_accuracy','roc_auc','brier')}})
    d=pd.DataFrame(rows)
    return d.groupby('model')[['accuracy','balanced_accuracy','roc_auc','brier']].quantile([.025,.975]).unstack().rename_axis(columns=['metric','quantile'])
def logistic_payload(model):return {'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist()}
def logistic_predict(payload,x):return expit(np.asarray(x)@np.array(payload['coef'])[0]+payload['intercept'][0])
def hmm_payload(model):return {k:getattr(model,k).tolist() for k in ('means','var','start','transition')}
def restore_hmm(payload):
    m=GaussianHMM(len(payload['start']))
    for k,v in payload.items():setattr(m,k,np.array(v,float))
    return m

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def source_digest(root):
    paths=['blind.py','blind_protocol.json','src/reliability.py','src/blind_research.py','src/historical.py','src/hmm_model.py','src/features.py','src/backtest.py','src/metrics.py']
    return {p:sha(Path(root)/p) for p in paths}
