"""Separate fit/freeze and reveal/evaluate stages for retrospective blind research."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score,adjusted_rand_score
from .historical import french_prices
from .reliability import (FEATURES,equity_features,future_volatility,label_end_dates,purged_mask,
    classifier_metrics,uncertainty,logistic_payload,logistic_predict,hmm_payload,restore_hmm,
    sha,dump_json,source_digest,block_indices)
from .hmm_model import GaussianHMM
from .backtest import run_backtest,validate_prices
from .metrics import summarize

ROOT=Path(__file__).resolve().parents[1]

def prepare(archive,work,protocol):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    p,provenance=french_prices(archive,protocol['data_start'],protocol['test_end'])
    p.loc[:protocol['validation_end']].to_csv(work/'development.csv')
    # Kept separate: fit() has no path to, and never reads, holdout.csv.
    p.loc[p.index>protocol['validation_end']].to_csv(work/'holdout.csv')
    dump_json(work/'provenance.json',provenance)
    dump_json(work/'data_seal.json',{'development_sha256':sha(work/'development.csv'),'holdout_sha256':sha(work/'holdout.csv')})
    print('Prepared sealed chronological partitions; no test values or metrics displayed.')

def read_prices(path):
    p=pd.read_csv(path,index_col='date',parse_dates=['date']);validate_prices(p);return p

def synthetic_truth(seed,n=1800):
    rng=np.random.default_rng(seed);state=np.zeros(n,dtype=int)
    for i in range(1,n):state[i]=state[i-1] if rng.random()<.98 else rng.integers(0,3)
    r=rng.normal(np.array([.0003,0.,-.0004])[state],np.array([.004,.012,.030])[state])
    p=pd.DataFrame({'equity':100*np.exp(np.cumsum(r)),'defensive':100*np.exp(np.arange(n)*.00005)},index=pd.bdate_range('2000-01-03',periods=n))
    return p,pd.Series(state,index=p.index)

def fit(work,out,protocol):
    work=Path(work);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists():raise ValueError('Frozen run already exists; use a new run directory')
    seal=json.loads((work/'data_seal.json').read_text())
    if sha(work/'development.csv')!=seal['development_sha256']:raise ValueError('Development data changed')
    p=read_prices(work/'development.csv')
    if p.index.max()>pd.Timestamp(protocol['validation_end']):raise ValueError('Development includes holdout')
    f=equity_features(p);f=f.loc[protocol['train_start']:]
    train=f.index<=pd.Timestamp(protocol['train_end']);val=~train
    if min(train.sum(),val.sum())<500:raise ValueError('Insufficient train/validation history')
    scaler=StandardScaler().fit(f.loc[train]);x=scaler.transform(f)
    candidates=[];models={}
    for k in protocol['state_counts']:
        fitted=[]
        for seed in protocol['fit_seeds']:
            m=GaussianHMM(k,seed,n_iter=protocol['hmm_iterations']).fit(x[train])
            score=m.score(x[train])/train.sum();fitted.append((score,m,seed))
        score,m,seed=max(fitted,key=lambda item:item[0])
        trainprob=m.filter(x[train]);vscore=m.score(x[val],trainprob[-1])/val.sum()
        occupancy=np.bincount(trainprob.argmax(1),minlength=k)/train.sum()
        candidates.append({'k':k,'seed':seed,'train_loglik':score,'validation_loglik':vscore,'min_training_occupancy':float(occupancy.min()),'iterations':len(m.history),'last_training_ll_change':m.history[-1]-m.history[-2] if len(m.history)>1 else 0.})
        models[k]=m;print('Fitted candidate',k,'states',flush=True)
    selection=pd.DataFrame(candidates);selection.to_csv(out/'selection.csv',index=False)
    winner=selection.sort_values('validation_loglik',ascending=False).iloc[0];m=models[int(winner.k)]
    probs=m.filter(x)
    forward=future_volatility(p,protocol['horizon']).reindex(f.index)
    end_dates=label_end_dates(p.index,protocol['horizon'])
    eligible=purged_mask(f.index,end_dates,protocol['train_start'],protocol['train_end']) & forward.notna().to_numpy()
    threshold=float(forward.loc[eligible].quantile(protocol['high_vol_training_quantile']))
    labels=(forward>threshold).astype(int)
    risk_head=LogisticRegression(C=1.,max_iter=1000).fit(probs[eligible],labels.loc[eligible])
    vmean=float(f.loc[eligible,'vol_20'].mean());vscale=float(f.loc[eligible,'vol_20'].std())
    base_x=((f[['vol_20']]-vmean)/vscale).to_numpy()
    baseline=LogisticRegression(C=1.,max_iter=1000).fit(base_x[eligible],labels.loc[eligible])
    state_vol=(probs[train].T@f.loc[train,'vol_20'].to_numpy())/probs[train].sum(0)
    order=np.argsort(state_vol);weights=np.empty(m.k)
    weights[order]=np.linspace(*protocol['allocation_equity_range'],m.k)
    targets=probs@weights
    artifact={'features':FEATURES,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
              'hmm':hmm_payload(m),'risk_head':logistic_payload(risk_head),'vol_baseline':logistic_payload(baseline),
              'vol_mean':vmean,'vol_scale':vscale,'high_vol_threshold':threshold,'training_event_prevalence':float(labels.loc[eligible].mean()),
              'weights':weights.tolist(),'last_development_probability':probs[-1].tolist(),'last_development_date':str(f.index[-1].date()),
              'validation_mean_equity_target':float(targets[val].mean()),'state_training_volatility':state_vol.tolist(),
              'selected_k':m.k,'selected_seed':int(winner.seed),'hmm_likelihood_history':m.history}
    dump_json(out/'model.json',artifact)
    valid_eval=purged_mask(f.index,end_dates,str(pd.Timestamp(protocol['train_end'])+pd.Timedelta(days=1)),protocol['validation_end']) & forward.notna().to_numpy()
    validation={'hmm':classifier_metrics(labels.loc[valid_eval],logistic_predict(artifact['risk_head'],probs[valid_eval])),
                'trailing_vol':classifier_metrics(labels.loc[valid_eval],logistic_predict(artifact['vol_baseline'],base_x[valid_eval]))}
    dump_json(out/'validation_metrics.json',validation)
    # An independent known-state synthetic experiment, trained without any test-seed labels.
    sp,truth=synthetic_truth(protocol['synthetic_train_seed'],protocol['synthetic_rows'])
    sf=equity_features(sp);ss=StandardScaler().fit(sf);sx=ss.transform(sf)
    sm=GaussianHMM(3,protocol['fit_seeds'][0],n_iter=protocol['hmm_iterations']).fit(sx)
    labels_s=sm.filter(sx).argmax(1);counts=confusion_matrix(truth.loc[sf.index],labels_s,labels=range(3))
    rows,cols=linear_sum_assignment(-counts);mapping=np.empty(3,int);mapping[cols]=rows
    synth={'hmm':hmm_payload(sm),'mean':ss.mean_.tolist(),'scale':ss.scale_.tolist(),'training_label_mapping':mapping.tolist()}
    dump_json(out/'synthetic_model.json',synth)
    freeze={'protocol':protocol,'data_seal':seal,'source_hashes':source_digest(ROOT),
            'model_sha256':sha(out/'model.json'),'synthetic_model_sha256':sha(out/'synthetic_model.json'),
            'provenance':json.loads((work/'provenance.json').read_text()),'status':'FROZEN BEFORE HOLDOUT EVALUATION',
            'scope':'Retrospective historical holdout; researcher knows history. Not prospective or independent third-party validation.'}
    dump_json(out/'freeze.json',freeze)
    print('Frozen models, source hashes and protocol. Test has not been evaluated.')

def validate_freeze(work,out):
    freeze=json.loads((out/'freeze.json').read_text())
    if source_digest(ROOT)!=freeze['source_hashes']:raise ValueError('Evaluation source/protocol changed since freeze')
    for file,key in [('model.json','model_sha256'),('synthetic_model.json','synthetic_model_sha256')]:
        if sha(out/file)!=freeze[key]:raise ValueError('Frozen model was altered')
    for file,key in [('development.csv','development_sha256'),('holdout.csv','holdout_sha256')]:
        if sha(work/file)!=freeze['data_seal'][key]:raise ValueError('Sealed data was altered')
    return freeze

def portfolio_metrics(bt,rf):
    result=summarize(bt);excess=bt.net-rf.reindex(bt.index).fillna(0)
    result['sharpe_vs_tbill']=float(excess.mean()/excess.std()*np.sqrt(252)) if excess.std()>0 else 0.
    result['trades']=int(bt.traded.sum());result['mean_equity_weight']=float(bt.equity_weight.mean())
    return result

def evaluate(work,out):
    work=Path(work);out=Path(out);freeze=validate_freeze(work,out);c=freeze['protocol']
    if (out/'blind_metrics.csv').exists():raise ValueError('Holdout already revealed. Do not silently retune and rerun.')
    art=json.loads((out/'model.json').read_text());m=restore_hmm(art['hmm'])
    development=read_prices(work/'development.csv');holdout=read_prices(work/'holdout.csv')
    if holdout.index.min()<=development.index.max():raise ValueError('Overlapping development/holdout')
    prices=pd.concat([development,holdout]);f=equity_features(prices).loc[holdout.index]
    x=(f.to_numpy()-np.array(art['scaler_mean']))/np.array(art['scaler_scale'])
    probs=m.filter(x,np.array(art['last_development_probability']))
    pred=logistic_predict(art['risk_head'],probs)
    baseline=logistic_predict(art['vol_baseline'],((f[['vol_20']]-art['vol_mean'])/art['vol_scale']).to_numpy())
    future=future_volatility(prices,c['horizon']).reindex(f.index);valid=future.notna()
    y=(future.loc[valid]>art['high_vol_threshold']).astype(int).to_numpy()
    pp={'hmm':pred[valid],'trailing_vol':baseline[valid],
        'constant_prevalence':np.full(valid.sum(),art['training_event_prevalence'])}
    metrics=pd.DataFrame({name:classifier_metrics(y,p) for name,p in pp.items()}).T
    metrics.to_csv(out/'blind_metrics.csv')
    uncertainty(y,pp,c['bootstrap_replicates'],c['bootstrap_block_sessions'],c['bootstrap_seed']).to_csv(out/'blind_confidence_intervals.csv')
    frame=pd.DataFrame({'actual_high_vol':y,'forecast_high_vol':pp['hmm'],'trailing_vol_forecast':pp['trailing_vol'],'forward_20d_vol':future.loc[valid]},index=f.index[valid])
    frame.to_csv(out/'blind_predictions.csv')
    yearly=[]
    for year,g in frame.groupby(frame.index.year):
        for name,col in [('hmm','forecast_high_vol'),('trailing_vol','trailing_vol_forecast')]:yearly.append({'year':int(year),'model':name,**classifier_metrics(g.actual_high_vol,g[col])})
    pd.DataFrame(yearly).to_csv(out/'blind_yearly.csv',index=False)
    pd.DataFrame(confusion_matrix(y,pp['hmm']>=.5,labels=[0,1]),index=['actual_low','actual_high'],columns=['pred_low','pred_high']).to_csv(out/'blind_confusion_matrix.csv')
    bins=pd.cut(frame.forecast_high_vol,np.linspace(0,1,11),include_lowest=True)
    calibration=frame.groupby(bins,observed=False).agg(n=('actual_high_vol','size'),predicted=('forecast_high_vol','mean'),observed=('actual_high_vol','mean'))
    calibration.to_csv(out/'calibration.csv')
    ood=np.max(np.abs(x),axis=1)>c['ood_z_limit']
    pd.DataFrame({'ood':ood,'max_abs_z':np.max(np.abs(x),axis=1),'state_confidence':probs.max(1)},index=f.index).to_csv(out/'data_drift.csv')
    target=pd.Series(probs@np.array(art['weights']),index=f.index)
    allocations={'buy_hold':pd.Series(1.,index=f.index),'fixed_60_40':pd.Series(.6,index=f.index),
                 'exposure_matched':pd.Series(art['validation_mean_equity_target'],index=f.index),
                 'vol_target_10pct':(.10/f.vol_20).clip(.1,.8),'regime':target,
                 'guarded_regime':target.where(~ood,.1)}
    kwargs={'cost':c['commission_bps']/10000.,'slippage_bps':c['slippage_bps'],'rebalance_every':c['rebalance_every'],'min_trade':c['min_trade'],'execution_delay':c['execution_delay']}
    portfolios={name:run_backtest(holdout,t,**kwargs,buy_and_hold=name=='buy_hold') for name,t in allocations.items()}
    rf=prices.defensive.pct_change(fill_method=None).reindex(f.index)
    pm=pd.DataFrame({name:portfolio_metrics(bt,rf) for name,bt in portfolios.items()}).T
    pm.to_csv(out/'blind_portfolios.csv')
    curves=pd.DataFrame({name:(1+bt.net).cumprod() for name,bt in portfolios.items()});curves.to_csv(out/'blind_equity_curves.csv')
    for name,bt in portfolios.items():bt.to_csv(out/f'ledger_{name}.csv')
    stress=[]
    for bps in [0,5,10,25]:
        for delay in [1,2]:
            bt=run_backtest(holdout,target,cost=c['commission_bps']/10000.,slippage_bps=bps,rebalance_every=c['rebalance_every'],min_trade=c['min_trade'],execution_delay=delay)
            stress.append({'slippage_bps':bps,'delay':delay,**portfolio_metrics(bt,rf)})
    pd.DataFrame(stress).to_csv(out/'execution_stress.csv',index=False)
    # Paired block resampling preserves local dependence and uses identical dates per strategy.
    rng=np.random.default_rng(c['bootstrap_seed']);differences=[]
    for _ in range(c['bootstrap_replicates']):
        ix=block_indices(len(holdout),c['bootstrap_block_sessions'],rng)
        rr=rf.to_numpy()[ix]
        def sharpe(r):
            ex=r-rr;return ex.mean()/ex.std(ddof=1)*np.sqrt(252)
        ours=portfolios['regime'].net.to_numpy()[ix]
        row={}
        for name in ['fixed_60_40','exposure_matched','vol_target_10pct']:
            other=portfolios[name].net.to_numpy()[ix]
            row[name+'_annual_mean_diff']=float((ours-other).mean()*252)
            row[name+'_sharpe_diff']=float(sharpe(ours)-sharpe(other))
        differences.append(row)
    pd.DataFrame(differences).quantile([.025,.975]).T.to_csv(out/'portfolio_difference_intervals.csv')
    synth=json.loads((out/'synthetic_model.json').read_text());sm=restore_hmm(synth['hmm']);srows=[]
    for seed in c['synthetic_test_seeds']:
        sp,truth=synthetic_truth(seed,c['synthetic_rows']);sf=equity_features(sp)
        sx=(sf.to_numpy()-np.array(synth['mean']))/np.array(synth['scale'])
        raw=sm.filter(sx).argmax(1);guess=np.array(synth['training_label_mapping'])[raw];true=truth.loc[sf.index].to_numpy()
        srows.append({'seed':seed,'n':len(true),'state_accuracy':accuracy_score(true,guess),'balanced_accuracy':balanced_accuracy_score(true,guess),'adjusted_rand_index':adjusted_rand_score(true,raw)})
    pd.DataFrame(srows).to_csv(out/'synthetic_blind_accuracy.csv',index=False)
    summary={'test_start':str(f.index[0].date()),'test_end':str(f.index[-1].date()),'scored_sessions':int(valid.sum()),'positive_sessions':int(y.sum()),'high_vol_threshold_annualized':art['high_vol_threshold'],'selected_states':art['selected_k'],'ood_sessions':int(ood.sum()),'ood_fraction':float(ood.mean()),'freeze_sha256':sha(out/'freeze.json'),'real_regime_accuracy':'Not identifiable: no ground-truth financial regime labels','live_trading_ready':False}
    dump_json(out/'blind_summary.json',summary)
    charts(frame,curves,out)
    report(metrics,pm,pd.DataFrame(srows),summary,out)
    print(metrics.to_string());print('Blind results saved; no retuning performed.')

def charts(frame,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.figsize':(11,4),'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots();ax.plot(frame.index,frame.forecast_high_vol,label='HMM forecast');ax.plot(frame.index,frame.trailing_vol_forecast,label='Trailing-vol baseline',alpha=.6);ax.set_ylabel('Probability');ax.set_title('Frozen 2020–2025 holdout · next-20-session high volatility');ax.legend();fig.tight_layout();fig.savefig(out/'blind_forecasts.svg');plt.close(fig)
    fig,ax=plt.subplots();curves.plot(ax=ax);ax.set_title('Historical research proxies · costs + next-close execution');ax.set_ylabel('NAV from 1');fig.tight_layout();fig.savefig(out/'blind_portfolios.svg');plt.close(fig)
    fig,ax=plt.subplots();cal=pd.read_csv(out/'calibration.csv');ax.plot(cal.predicted,cal.observed,'o-');ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed event fraction',title='Holdout calibration · sparse bins are uncertain');fig.tight_layout();fig.savefig(out/'blind_calibration.svg');plt.close(fig)

def report(metrics,pm,synthetic,summary,out):
    ci=pd.read_csv(out/'blind_confidence_intervals.csv',header=[0,1],index_col=0)
    lines=['# Metis: frozen historical holdout results','',
           '**Research validation only. Not a live-trading approval.**','',
           'The fitted model and evaluation protocol were frozen before this test was revealed. This is retrospective: the researcher knows historical events, and current revised research data are not point-in-time feeds.','',
           '## What accuracy means','',
           f"Predict whether volatility over the next 20 sessions exceeds the training-period 75th percentile ({summary['high_vol_threshold_annualized']:.2%} annualized). It does not mean price-direction accuracy, crash accuracy, or true-regime accuracy.",'',
           f"Scored {summary['scored_sessions']} dates; {summary['positive_sessions']} high-volatility outcomes. The final 20 sessions lack full future labels and are excluded from classification scoring. All sessions remain in portfolio evaluation.",'',
           '| Model | Accuracy | Balanced accuracy | ROC AUC | Precision | Recall | Brier ↓ |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in metrics.iterrows():lines.append(f'| {name} | {r.accuracy:.2%} | {r.balanced_accuracy:.2%} | {r.roc_auc:.3f} | {r.precision:.2%} | {r.recall:.2%} | {r.brier:.4f} |')
    lines+=['','95% uncertainty intervals use 1,000 paired moving-block resamples of 60 sessions, accounting approximately for overlapping labels and serial dependence. They are conditional on this frozen model and dataset; they do not cover model-selection uncertainty or future structural changes. See `blind_confidence_intervals.csv`, including HMM-minus-baseline intervals.','',
            '## Known-state synthetic tests','',synthetic.to_markdown(index=False),'','State mapping was learned using training labels only. These synthetic percentages do not establish real-market regime accuracy.','',
            '## Portfolio outcomes','',pm[['annualized_return','annualized_volatility','max_drawdown','sharpe_vs_tbill','annual_turnover','trades']].to_markdown(),'','All six portfolios use the same execution/cost conventions. Exposure-matched weight comes from validation, not test. Vol-target is a simple causal baseline. The guarded variant caps equity at 10% when feature magnitudes exceed the frozen out-of-distribution threshold.','',
            'Costs: 5 bps commission + 5 bps slippage on each traded leg; rebalance every five sessions with a 5-percentage-point band. Signals execute at the following close. Added 2-session-delay and 0–25 bps slippage stresses are diagnostic only.','',
            f"Drift warning: {summary['ood_sessions']} sessions ({summary['ood_fraction']:.2%}) exceeded the training-standardized feature guard.",'',
            '## Trust and limitations','',
            '- The historical market/T-bill series are research proxies, not tradeable ETF quotes. Publication delay and historical revisions prevent claims of actual executable returns.',
            '- The forecasting head is supervised logistic calibration on unsupervised HMM state probabilities. The named target is observable; latent regimes have no definitive labels.',
            '- Feature scaling and HMM fitting use training only; forward label windows are purged at the training boundary; holdout labels never fit the model.',
            '- Synthetic seed tests, prefix invariance, serialization, execution and malformed-input checks test implementation. They do not prove profitability.',
            '- Compare with the simple trailing-vol forecast and passive/exposure-matched portfolios. A higher raw accuracy can come from class imbalance. Lower drawdown can come from lower equity exposure.',
            '- No threshold, state count, feature, cost or allocation was changed after revealing these results. Reusing this holdout for future tuning would consume its independence.',
            '- Paper monitoring on a new point-in-time ETF dataset and prospectively timestamped signals is still required before live use.','',
            '## Figures','', '![Forecasts](blind_forecasts.svg)','![Portfolios](blind_portfolios.svg)','![Calibration](blind_calibration.svg)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
