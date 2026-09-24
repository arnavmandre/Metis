"""Metis v2: fit/freeze on consumed history, then one fresh 2026 evaluation."""
import argparse,json,pickle,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,precision_score,recall_score,fbeta_score,confusion_matrix
from sklearn.inspection import permutation_importance
from src.historical import french_prices
from src.features_v2 import make_features_v2
from src.reliability import future_volatility,label_end_dates,purged_mask,classifier_metrics,uncertainty,dump_json,sha,restore_hmm,logistic_predict,equity_features
from src.model_v2 import candidate,inputs,predict_raw,calibrator,calibrated,choose_alert_threshold
from src.backtest import run_backtest
from src.blind_research import portfolio_metrics
ROOT=Path(__file__).resolve().parent
SOURCES=['upgrade.py','v2_protocol.json','src/features_v2.py','src/model_v2.py','src/historical.py','src/reliability.py','src/backtest.py','src/metrics.py','src/hmm_model.py','src/features.py','research/frozen/model.json']
def sources():return {n:sha(ROOT/n) for n in SOURCES}
def read(path):return pd.read_csv(path,index_col='date',parse_dates=['date'])
def prepare(archive,work,c):
    work.mkdir(parents=True,exist_ok=True)
    p,provenance=french_prices(archive,'1999-01-01',c['test_end'])
    p.loc[:c['calibration_end']].to_csv(work/'development.csv')
    p.loc[p.index>c['calibration_end']].to_csv(work/'holdout.csv')
    dump_json(work/'seal.json',{'development':sha(work/'development.csv'),'holdout':sha(work/'holdout.csv'),'provenance':provenance})
    print('2026 partition sealed without displaying outcomes.')
def fit(work,out,c):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists():raise ValueError('Run already frozen; use a new directory')
    seal=json.loads((work/'seal.json').read_text());p=read(work/'development.csv')
    if sha(work/'development.csv')!=seal['development']:raise ValueError('Development checksum mismatch')
    if p.index.max()>pd.Timestamp(c['calibration_end']):raise ValueError('Holdout in development')
    f=make_features_v2(p).loc[c['train_start']:];fv=future_volatility(p,c['horizon']).reindex(f.index)
    ends=label_end_dates(p.index,c['horizon']);y=(fv>c['event_threshold']).astype(int)
    masks={key:purged_mask(f.index,ends,start,end)&fv.notna().to_numpy() for key,start,end in [('train',c['train_start'],c['train_end']),('select','2023-01-01',c['selection_end']),('cal','2025-01-01',c['calibration_end'])]}
    trained={};rows=[]
    for name in c['candidates']:
        m=candidate(name,c['seed']);x=inputs(name,f)
        if name=='har_volatility':
            m.fit(x[masks['train']],np.log(fv.loc[masks['train']]))
            scale=float(np.std(np.log(fv.loc[masks['train']])-m.predict(x[masks['train']]),ddof=1))
        else:m.fit(x[masks['train']],y.loc[masks['train']]);scale=None
        pr=predict_raw(name,m,f.loc[masks['select']],c['event_threshold'],scale)
        rows.append({'candidate':name,**classifier_metrics(y.loc[masks['select']],pr)})
        cal_raw=predict_raw(name,m,f.loc[masks['cal']],c['event_threshold'],scale)
        cal=calibrator(cal_raw,y.loc[masks['cal']]);cp=calibrated(cal_raw,cal)
        trained[name]={'model':m,'calibrator':cal,'residual_scale':scale,'alert_threshold':choose_alert_threshold(y.loc[masks['cal']],cp)}
    selection=pd.DataFrame(rows).sort_values('brier');selection.to_csv(out/'selection.csv',index=False)
    champion=selection.iloc[0].candidate
    # Frozen model selection uses only 2023-24; calibration year is separate.
    validation_importance=[]
    chosen=trained[champion]
    # Honest sensitivity explanation: shuffled feature effects on selection Brier, not causality.
    baseline_p=predict_raw(champion,chosen['model'],f.loc[masks['select']],c['event_threshold'],chosen['residual_scale'])
    baseline_loss=brier_score_loss(y.loc[masks['select']],baseline_p);rng=np.random.default_rng(c['seed'])
    for col in f.columns:
        changed=f.loc[masks['select']].copy();changed[col]=rng.permutation(changed[col].to_numpy())
        pr=predict_raw(champion,chosen['model'],changed,c['event_threshold'],chosen['residual_scale'])
        validation_importance.append({'feature':col,'permuted_brier_increase':brier_score_loss(y.loc[masks['select']],pr)-baseline_loss})
    pd.DataFrame(validation_importance).sort_values('permuted_brier_increase',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    bundle={'models':trained,'champion':champion,'features':list(f),'train_mean':f.loc[masks['train']].mean().to_dict(),'train_std':f.loc[masks['train']].std().replace(0,1).to_dict(),'protocol':c}
    with open(out/'models.pkl','wb') as h:pickle.dump(bundle,h)
    dump_json(out/'freeze.json',{'protocol':c,'source_hashes':sources(),'model_sha256':sha(out/'models.pkl'),'data_seal':seal,'champion':champion,'feature_count':len(f.columns),'instrument':seal['provenance'].get('instrument','US_CRSP_MARKET'),'status':'Frozen before 2026 reveal','warning':'Only load the locally generated, checksummed pickle; never an untrusted pickle'})
    print('Frozen champion:',champion,'Features:',len(f.columns))
def evaluate(work,out):
    frozen=json.loads((out/'freeze.json').read_text());c=frozen['protocol']
    if frozen['source_hashes']!=sources() or sha(out/'models.pkl')!=frozen['model_sha256']:raise ValueError('Source/model changed after freeze')
    for name in ['development','holdout']:
        if sha(work/(name+'.csv'))!=frozen['data_seal'][name]:raise ValueError('Data changed after freeze')
    if (out/'test_metrics.csv').exists():raise ValueError('2026 already revealed; no silent rerun')
    with open(out/'models.pkl','rb') as h:b=pickle.load(h)
    p=pd.concat([read(work/'development.csv'),read(work/'holdout.csv')]);f=make_features_v2(p);f=f.loc[f.index>c['calibration_end']]
    fv=future_volatility(p,c['horizon']).reindex(f.index);valid=fv.notna();y=(fv.loc[valid]>c['event_threshold']).astype(int)
    probabilities={};all_p={};alerts=[]
    for name,label in [(b['champion'],'v2_champion'),('trailing_vol','trailing_vol_retrained')]:
        entry=b['models'][name];pr=calibrated(predict_raw(name,entry['model'],f,c['event_threshold'],entry['residual_scale']),entry['calibrator'])
        all_p[label]=pr;probabilities[label]=pr[valid]
        t=entry['alert_threshold'];pred=pr[valid]>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        alerts.append({'model':label,'threshold':t,'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f2':fbeta_score(y,pred,beta=2,zero_division=0),'false_positive_rate':fp/max(tn+fp,1),'alerts':int(pred.sum())})
    # Original model uses its original fit and calibration; only its forward state is continued.
    old=json.loads((ROOT/'research/frozen/model.json').read_text());ef=equity_features(p);ef=ef.loc[ef.index>old['last_development_date']]
    ox=(ef.to_numpy()-np.array(old['scaler_mean']))/np.array(old['scaler_scale'])
    op=restore_hmm(old['hmm']).filter(ox,np.array(old['last_development_probability']))
    oldpr=pd.Series(logistic_predict(old['risk_head'],op),index=ef.index).reindex(f.index).to_numpy()
    all_p['v1_frozen_hmm']=oldpr;probabilities['v1_frozen_hmm']=oldpr[valid]
    # Existing paired-bootstrap helper expects the reference key 'hmm'; rename only for computation.
    bp={('hmm' if k=='v2_champion' else k):v for k,v in probabilities.items()}
    ci=uncertainty(y.to_numpy(),bp,c['bootstrap_replicates'],c['bootstrap_block_sessions'],c['seed']);ci.index=[('v2_champion' if n=='hmm' else 'v2_champion_minus_'+n[len('hmm_minus_'):] if n.startswith('hmm_minus_') else n) for n in ci.index];ci.to_csv(out/'test_intervals.csv')
    metrics=pd.DataFrame({name:classifier_metrics(y,pr) for name,pr in probabilities.items()}).T;metrics.to_csv(out/'test_metrics.csv')
    pd.DataFrame(alerts).to_csv(out/'alert_metrics.csv',index=False)
    pred=pd.DataFrame(probabilities,index=f.index[valid]);pred['actual_high_vol']=y;pred['realized_forward_vol']=fv.loc[valid];pred.to_csv(out/'predictions.csv')
    thresholds=b['models'][b['champion']]['alert_threshold']
    z=(f-pd.Series(b['train_mean']))/pd.Series(b['train_std']);ood=z.abs().max(axis=1)>c['ood_z_limit']
    target=pd.Series(.8-.7*all_p['v2_champion'],index=f.index).clip(.1,.8)
    targets={'v2':target,'v1':pd.Series(.8-.7*oldpr,index=f.index).clip(.1,.8),'fixed_60_40':pd.Series(.6,index=f.index),'vol_target_10pct':(.1/f.vol_20).clip(.1,.8)}
    portfolios={name:run_backtest(p.loc[f.index],t,cost=c['cost_bps_per_leg']/10000,rebalance_every=c['rebalance_every'],min_trade=c['min_trade']) for name,t in targets.items()}
    rf=p.defensive.pct_change(fill_method=None).reindex(f.index)
    pm=pd.DataFrame({name:portfolio_metrics(bt,rf) for name,bt in portfolios.items()}).T;pm.to_csv(out/'portfolios.csv')
    curves=pd.DataFrame({name:(1+bt.net).cumprod() for name,bt in portfolios.items()});curves.to_csv(out/'equity_curves.csv')
    for name,bt in portfolios.items():bt.to_csv(out/f'ledger_{name}.csv')
    calibration=pd.DataFrame({'p':probabilities['v2_champion'],'y':y.to_numpy()});calibration['bin']=pd.cut(calibration.p,np.linspace(0,1,6),include_lowest=True)
    calibration.groupby('bin',observed=False).agg(n=('y','size'),predicted=('p','mean'),observed=('y','mean')).to_csv(out/'calibration.csv')
    summary={'champion':b['champion'],'feature_count':len(f.columns),'test_start':str(f.index[0].date()),'test_end':str(f.index[-1].date()),'scored_sessions':int(valid.sum()),'positive_sessions':int(y.sum()),'ood_sessions':int(ood.sum()),'event_threshold':c['event_threshold'],'alert_threshold':thresholds,'live_proven':False,'test_scope':'Small 2026 retrospective holdout; 2020-25 reused openly for development','freeze_sha256':sha(out/'freeze.json')}
    dump_json(out/'summary.json',summary)
    print(metrics.to_string());print('2026 evaluation complete. Do not retune against this result.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit','evaluate']);ap.add_argument('--archive',default='../french.download');ap.add_argument('--work',default='data/v2');ap.add_argument('--out',default='outputs/v2');args=ap.parse_args();c=json.loads((ROOT/'v2_protocol.json').read_text());work=Path(args.work);out=Path(args.out)
    if args.stage=='prepare':prepare(args.archive,work,c)
    elif args.stage=='fit':fit(work,out,c)
    else:evaluate(work,out)
