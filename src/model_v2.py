"""Baseline-first model tournament and separate chronological calibration."""
import numpy as np
from scipy.special import expit,logit
from scipy.stats import norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score,precision_score,confusion_matrix

def candidate(name,seed):
    if name in ['trailing_vol','logistic_regularized','logistic']:
        return make_pipeline(StandardScaler(),LogisticRegression(C=.1 if name=='logistic_regularized' else 1.,max_iter=3000))
    if name.startswith('boosting'):
        return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7 if name=='boosting_shallow' else 15,max_depth=3,learning_rate=.05,l2_regularization=10,min_samples_leaf=80,early_stopping=False,random_state=seed)
    if name=='har_volatility':return make_pipeline(StandardScaler(),Ridge(alpha=10))
    raise ValueError(name)

def inputs(name,f):
    if name=='trailing_vol':return f[['vol_20']].to_numpy()
    if name=='har_volatility':return np.log(f[['vol_5','vol_20','vol_60','ewma_vol']].clip(lower=1e-5)).to_numpy()
    return f.to_numpy()

def predict_raw(name,model,f,threshold,residual_scale=None):
    if name=='har_volatility':
        mu=model.predict(inputs(name,f))
        return norm.sf((np.log(threshold)-mu)/residual_scale)
    return model.predict_proba(inputs(name,f))[:,1]

def calibrator(p,y):
    counts=np.bincount(np.asarray(y,int),minlength=2)
    if counts.min()<10:return None
    return LogisticRegression(C=1.,max_iter=1000).fit(logit(np.clip(p,1e-6,1-1e-6))[:,None],y)

def calibrated(p,cal):
    if cal is None:return np.asarray(p)
    return cal.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1]

def choose_alert_threshold(y,p):
    options=[]
    for t in np.linspace(.05,.85,33):
        pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        precision=precision_score(y,pred,zero_division=0);fpr=fp/max(tn+fp,1)
        if precision>=.35 and fpr<=.25:options.append((fbeta_score(y,pred,beta=2,zero_division=0),float(t)))
    return max(options)[1] if options else .5
