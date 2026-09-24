import unittest
import numpy as np
import pandas as pd
from src.data_loader import synthetic
from src.features import make_features
from src.hmm_model import GaussianHMM
from src.backtest import run_backtest

class Integrity(unittest.TestCase):
    def test_features_prefix_invariant(self):
        p=synthetic(400)
        pd.testing.assert_frame_equal(make_features(p).loc[:p.index[250]],make_features(p.iloc[:251]))
    def test_filter_prefix_and_continuation(self):
        x=np.random.default_rng(2).normal(size=(150,3)); m=GaussianHMM(2,n_iter=15).fit(x[:70])
        full=m.filter(x)
        np.testing.assert_allclose(full[:100],m.filter(x[:100]))
        np.testing.assert_allclose(full[100:],m.filter(x[100:],full[99]))
        np.testing.assert_allclose(full.sum(1),1)
        np.testing.assert_allclose(m.transition.sum(1),1)
    def test_execution_lag(self):
        ix=pd.date_range('2020-01-01',periods=4)
        p=pd.DataFrame({'equity':[100,200,400,800],'bonds':[100]*4},index=ix)
        b=run_backtest(p,pd.Series([0,1,1,1],index=ix),0)
        np.testing.assert_allclose(b.net,[0,0,0,1])
    def test_initial_cost_and_drift(self):
        ix=pd.date_range('2020-01-01',periods=3)
        p=pd.DataFrame({'equity':[100,100,200],'bonds':[100]*3},index=ix)
        b=run_backtest(p,pd.Series(.5,index=ix),.001)
        self.assertAlmostEqual(b.net.iloc[1],1/1.001-1)
        self.assertGreater(b.turnover.iloc[2],.3)
    def test_zero_cost_buy_hold(self):
        p=synthetic(100)[['equity','bonds']]
        b=run_backtest(p,pd.Series(1.,index=p.index),0)
        self.assertAlmostEqual((1+b.net).prod(),p.equity.iloc[-1]/p.equity.iloc[1])

if __name__=='__main__': unittest.main()
