import json
import hashlib
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .features import make_features
from .regimes import compare
from .hmm_model import GaussianHMM
from .backtest import run_backtest
from .metrics import summarize
from .visualization import plots

def run(prices, cfg, out='outputs', demo=False):
    out=Path(out); (out/'results').mkdir(parents=True,exist_ok=True); (out/'models').mkdir(exist_ok=True)
    f=make_features(prices, cfg.get('windows',[5,20,60]))
    if demo:
        a,b=int(len(f)*.6),int(len(f)*.8)
    else:
        a=int((f.index<=cfg['train_end']).sum()); b=int((f.index<=cfg['validation_end']).sum())
    if min(a,b-a,len(f)-b)<100:
        raise ValueError('Need at least 100 complete observations in each chronological split')
    scaler=StandardScaler().fit(f.iloc[:a]); x=scaler.transform(f)
    comparison,model,all_models=compare(x[:a],x[a:b],cfg['states'],cfg['seed'])
    comparison.to_csv(out/'results/model_comparison.csv',index=False)
    probs=model.filter(x)
    for (kind,k), fitted in all_models.items():
        if kind == "gmm":
            pd.DataFrame(fitted.predict_proba(x),index=f.index).to_csv(out/f"results/gmm_{k}_probabilities.csv")
    # State naming/risk order is learned from training data ONLY.
    train_labels=probs[:a].argmax(1)
    profiles=f.iloc[:a].groupby(train_labels).mean().reindex(range(model.k))
    vol_col=f"vol_{cfg.get('windows',[5,20,60])[1]}"
    fallback=pd.Series(scaler.inverse_transform(model.means)[:,f.columns.get_loc(vol_col)],index=range(model.k))
    risk=profiles[vol_col].fillna(fallback).sort_values().index
    weights=np.empty(model.k); weights[risk]=np.interp(np.linspace(0,1,model.k),np.linspace(0,1,len(cfg['risk_weights'])),cfg['risk_weights'])
    profiles['risk_rank']=pd.Series(np.argsort(np.argsort(profiles[vol_col].fillna(fallback))),index=profiles.index)
    profiles.to_csv(out/'results/training_state_profiles.csv')
    pd.DataFrame(model.transition).to_csv(out/'results/transition_matrix.csv',index=False)
    pd.DataFrame(probs,index=f.index,columns=[f'state_{i}' for i in range(model.k)]).to_csv(out/'results/filtered_probabilities.csv')
    test=f.index[b:]; p=prices.loc[test]; target=pd.Series(probs[b:]@weights,index=test)
    portfolios={'buy_hold':run_backtest(p,pd.Series(1.,index=test),cfg['transaction_cost']),
                'fixed_60_40':run_backtest(p,pd.Series(.6,index=test),cfg['transaction_cost']),
                'regime':run_backtest(p,target,cfg['transaction_cost'])}
    metrics=pd.DataFrame({name:summarize(bt) for name,bt in portfolios.items()}).T
    metrics.to_csv(out/'results/portfolio_metrics.csv')
    for name,bt in portfolios.items(): bt.to_csv(out/f'results/{name}_backtest.csv')
    sensitivity=[]
    for cost in [0,.0005,.001,.002,.005]:
        for scale in [.8,1.,1.2]:
            result=summarize(run_backtest(p,(target*scale).clip(0,1),cost))
            sensitivity.append({'cost':cost,'allocation_scale':scale,**result})
    pd.DataFrame(sensitivity).to_csv(out/'results/cost_allocation_sensitivity.csv',index=False)
    # Validation-only structural robustness: do not select on test outcomes.
    structural=[]
    for window_set in [[3,15,45],[5,20,60],[10,30,90]]:
        alt=make_features(prices,window_set)
        for trim in [0,.15]:
            train=alt.loc[alt.index<=f.index[a-1]]; train=train.iloc[int(len(train)*trim):]
            val=alt.loc[(alt.index>f.index[a-1]) & (alt.index<=f.index[b-1])]
            sc=StandardScaler().fit(train); tr=sc.transform(train); va=sc.transform(val)
            hm=GaussianHMM(model.k,cfg['seed'],n_iter=60).fit(tr)
            pr=hm.filter(va,hm.filter(tr)[-1])
            structural.append({'windows':str(window_set),'training_trim':trim,'validation_loglik_per_day':hm.score(va,hm.filter(tr)[-1])/len(va),'switch_fraction':float(np.mean(np.diff(pr.argmax(1))!=0))})
    pd.DataFrame(structural).to_csv(out/'results/window_training_sensitivity.csv',index=False)
    state=probs.argmax(1); starts=np.r_[0,np.flatnonzero(np.diff(state))+1]; ends=np.r_[starts[1:],len(state)]
    pd.DataFrame({'state':state[starts],'start':f.index[starts],'duration':ends-starts,'boundary_censored':(starts==0)|(ends==len(state))}).to_csv(out/'results/durations.csv',index=False)
    entropy=-(probs*np.log(np.maximum(probs,1e-15))).sum(1)
    pd.DataFrame({'entropy':entropy,'confidence':probs.max(1)},index=f.index).to_csv(out/'results/uncertainty.csv')
    perstate=[]
    for name,bt in portfolios.items():
        # State known for the decision underlying that day's earned return.
        decision=pd.Series(state[b:],index=test).shift(2)
        for s,g in bt.groupby(decision):
            perstate.append({'portfolio':name,'decision_state':int(s),'sessions':len(g),'mean_daily_net':g.net.mean(),'daily_volatility':g.net.std()})
    pd.DataFrame(perstate).to_csv(out/'results/performance_by_state.csv',index=False)
    title='SYNTHETIC DEMO — no financial conclusions' if demo else 'Historical research — filtered states; frozen training fit'
    plots(prices,f,probs,model,profiles.drop(columns='risk_rank'),PCA(2).fit(x[:a]).transform(x),portfolios,out/'figures',title)
    with open(out/'models/fitted.pkl','wb') as stream: pickle.dump({'model':model,'scaler':scaler,'features':list(f),'weights':weights},stream)
    manifest={'input_sha256':hashlib.sha256(prices.to_csv().encode()).hexdigest(),'synthetic':demo,'selected_states':model.k,'train_end':str(f.index[a-1]),'validation_end':str(f.index[b-1]),'test_start':str(test[0]),'test_end':str(test[-1]),'rows':len(f),'config':cfg,'hmm_iterations':len(model.history),'hmm_loglik_history':model.history}
    (out/'results/run_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'results/REPORT.md').write_text('# Metis run\n\n'+title+'\n\n```\n'+metrics.to_string()+'\n```\n\nModel selection uses validation likelihood; test is untouched until evaluation. Sensitivities are exploratory, not independent confirmation. Labels describe training statistics, not confirmed economic regimes.\n')
    return metrics
