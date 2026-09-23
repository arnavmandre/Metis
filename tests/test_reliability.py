import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.backtest import run_backtest,validate_prices
from src.reliability import future_volatility,label_end_dates,purged_mask,logistic_payload,logistic_predict,hmm_payload,restore_hmm,equity_features
from src.hmm_model import GaussianHMM
from src.blind_research import synthetic_truth

class Reliability(unittest.TestCase):
    def setUp(self):self.p,_=synthetic_truth(5,400)
    def test_forward_label_exact_window(self):
        r=self.p.equity.pct_change();h=20;expected=r.iloc[101:121].std()*np.sqrt(252)
        self.assertAlmostEqual(future_volatility(self.p,h).iloc[100],expected)
        self.assertTrue(future_volatility(self.p,h).iloc[-h:].isna().all())
    def test_purge_drops_cross_boundary_labels(self):
        end=self.p.index[199];last=label_end_dates(self.p.index,20)
        mask=purged_mask(self.p.index,last,self.p.index[0],end)
        self.assertEqual(np.flatnonzero(mask)[-1],179)
        self.assertTrue((last[mask]<=end).all())
    def test_future_mutation_never_changes_features(self):
        altered=self.p.copy();altered.iloc[300:,0]*=10
        pd.testing.assert_frame_equal(equity_features(self.p).iloc[:200],equity_features(altered).iloc[:200])
    def test_serialized_model_matches(self):
        x=np.random.default_rng(4).normal(size=(90,3));m=GaussianHMM(2,n_iter=10).fit(x)
        restored=restore_hmm(json.loads(json.dumps(hmm_payload(m))))
        np.testing.assert_allclose(m.filter(x),restored.filter(x))
        l=LogisticRegression().fit(x,x[:,0]>0)
        np.testing.assert_allclose(l.predict_proba(x)[:,1],logistic_predict(logistic_payload(l),x))
    def test_weekly_cadence_and_buy_hold(self):
        t=pd.Series(.5,index=self.p.index)
        b=run_backtest(self.p,t,rebalance_every=5)
        self.assertTrue(all((np.flatnonzero(b.traded)-1)%5==0))
        self.assertEqual(run_backtest(self.p,t,buy_and_hold=True).traded.sum(),1)
    def test_costs_decrease_nav_on_flat_prices(self):
        p=self.p.copy();p.iloc[:]=100;t=pd.Series(np.arange(len(p))%2,index=p.index)
        free=run_backtest(p,t,cost=0);costly=run_backtest(p,t,cost=.001,slippage_bps=5)
        self.assertLess((1+costly.net).prod(),(1+free.net).prod())
        np.testing.assert_allclose(costly.cost_fraction,costly.commission_fraction+costly.slippage_fraction,atol=1e-12)
    def test_extra_execution_delay(self):
        ix=pd.date_range('2020-01-01',periods=5);p=pd.DataFrame({'equity':[1,2,4,8,16],'defensive':[1]*5},index=ix)
        b=run_backtest(p,pd.Series(1.,index=ix),cost=0,execution_delay=2)
        np.testing.assert_allclose(b.net,[0,0,0,1,1])
    def test_invalid_prices_fail_closed(self):
        for kind in ['missing','negative','duplicate','reversed']:
            p=self.p.copy()
            if kind=='missing':p.iloc[20,0]=np.nan
            if kind=='negative':p.iloc[20,0]=-1
            if kind=='duplicate':p.index=list(p.index[:-1])+[p.index[-2]]
            if kind=='reversed':p=p.iloc[::-1]
            with self.assertRaises(ValueError):validate_prices(p)
    def test_trade_band_preserves_drift(self):
        ix=pd.date_range('2020-01-01',periods=3);p=pd.DataFrame({'equity':[1,1,1.1],'defensive':[1,1,1]},index=ix)
        b=run_backtest(p,pd.Series(.5,index=ix),cost=0,min_trade=.1)
        self.assertFalse(b.traded.iloc[-1]);self.assertAlmostEqual(b.equity_weight.iloc[-1],.55/1.05)
if __name__=='__main__':unittest.main()
