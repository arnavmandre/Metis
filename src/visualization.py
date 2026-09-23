from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plots(prices, features, probs, model, profiles, pca, portfolios, out, title):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'figure.figsize':(12,5),'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    labels = probs.argmax(1); dates=features.index
    colors = plt.get_cmap('tab10')
    cmap=ListedColormap([colors(i) for i in range(model.k)])
    norm=BoundaryNorm(np.arange(model.k+1)-.5,model.k)
    def save(name):
        plt.suptitle(title, fontsize=9, color='gray'); plt.tight_layout(); plt.savefig(out/f'{name}.png',dpi=140); plt.close()
    def shade(ax):
        starts = np.r_[0,np.flatnonzero(np.diff(labels))+1]; ends=np.r_[starts[1:],len(labels)-1]
        for a,b in zip(starts,ends): ax.axvspan(dates[a],dates[b],color=colors(labels[a]),alpha=.17)
    fig,ax=plt.subplots(); ax.plot(dates,prices.loc[dates,'equity'],color='#172d47'); shade(ax); ax.legend(handles=[Patch(facecolor=colors(i),alpha=.4,label=f'State {i}') for i in range(model.k)],ncol=model.k,loc='upper left'); ax.set_title('Equity price · causal filtered states'); save('01_market_regimes')
    plt.figure(); plt.scatter(dates,labels,c=labels,cmap=cmap,norm=norm,s=3); plt.ylabel('State'); save('02_timeline')
    plt.figure(); z=(profiles-profiles.mean())/profiles.std().replace(0,1); plt.imshow(z,aspect='auto',cmap='RdBu_r'); plt.xticks(range(len(z.columns)),z.columns,rotation=90); plt.yticks(range(len(z)),z.index); plt.colorbar(); save('03_profiles')
    plt.figure(); plt.imshow(model.transition,vmin=0,vmax=1,cmap='Blues'); plt.xlabel('To'); plt.ylabel('From'); plt.colorbar(); save('04_transition')
    plt.figure(); plt.stackplot(dates,probs.T,labels=[f'State {i}' for i in range(probs.shape[1])]); plt.legend(loc='upper left'); save('05_probabilities')
    plt.figure(); plt.scatter(pca[:,0],pca[:,1],c=labels,cmap=cmap,norm=norm,s=4,alpha=.5); plt.xlabel('PC1'); plt.ylabel('PC2'); save('06_pca')
    for number,kind in [(7,'drawdown'),(8,'equity'),(9,'sharpe'),(10,'volatility')]:
        plt.figure()
        for name,bt in portfolios.items():
            r=bt.net; nav=(1+r).cumprod()
            y={'drawdown':nav/nav.cummax().clip(lower=1)-1,'equity':nav,'sharpe':r.rolling(126).mean()/r.rolling(126).std()*np.sqrt(252),'volatility':r.rolling(126).std()*np.sqrt(252)}[kind]
            plt.plot(y,label=name)
        plt.title('Held-out test · '+kind); plt.legend(); save(f'{number:02d}_{kind}')
    starts=np.r_[0,np.flatnonzero(np.diff(labels))+1]; durations=np.diff(np.r_[starts,len(labels)])
    plt.figure(); plt.hist(durations,bins=30); plt.xlabel('Observed run length (sessions; boundary runs censored)'); save('11_durations')
    plt.figure(); profiles['return_1'].plot.bar(); plt.ylabel('Contemporaneous mean daily return (descriptive)'); save('12_returns_by_state')
    plt.figure(); plt.imshow(features.corr(),vmin=-1,vmax=1,cmap='RdBu_r'); plt.colorbar(); plt.xticks(range(len(features.columns)),features.columns,rotation=90); save('13_feature_correlation')
    plt.figure(); features.return_1.hist(bins=70); plt.xlabel('Equity daily return'); save('14_return_distribution')
    plt.figure(); features.filter(like='vol_').drop(columns=['vol_ratio'],errors='ignore').plot(ax=plt.gca()); save('15_realized_volatility')
    plt.figure(); features.filter(like='corr_').plot(ax=plt.gca()); save('16_correlations')
