import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from .hmm_model import GaussianHMM

def compare(train, validation, ks, seed):
    rows, models = [], {}
    for k in ks:
        for kind in ('kmeans','gmm','hmm'):
            if kind == 'kmeans':
                m = KMeans(k, n_init=10, random_state=seed).fit(train)
                label = m.predict(validation)
                other = KMeans(k, n_init=10, random_state=seed+1).fit(train)
                score = m.score(validation)/len(validation)
                stability = adjusted_rand_score(label, other.predict(validation))
            elif kind == 'gmm':
                m = GaussianMixture(k, covariance_type='diag', n_init=3, reg_covar=.01, random_state=seed).fit(train)
                label = m.predict(validation); score = m.score(validation); stability = np.nan
            else:
                m = GaussianHMM(k, seed).fit(train)
                previous = m.filter(train)[-1]
                label = m.filter(validation, previous).argmax(1)
                score = m.score(validation, previous)/len(validation); stability = np.nan
            silhouette = silhouette_score(validation, label, sample_size=min(1000,len(label)), random_state=seed) if 1<len(set(label))<len(label) else np.nan
            rows.append({'model':kind,'k':k,'validation_score':score,'silhouette':silhouette,'seed_stability_ari':stability})
            models[kind,k] = m
    table = pd.DataFrame(rows)
    # Select HMM on validation likelihood only; never on test portfolio return.
    winner = table[table.model=='hmm'].sort_values('validation_score', ascending=False).iloc[0]
    return table, models['hmm',int(winner.k)], models
