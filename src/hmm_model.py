"""Diagonal Gaussian HMM: EM training and strictly causal forward filtering."""
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class GaussianHMM:
    def __init__(self, n_components=4, seed=42, n_iter=100, tol=1e-4):
        self.k, self.seed, self.n_iter, self.tol = n_components, seed, n_iter, tol

    def emissions(self, x):
        return -.5 * (np.log(2*np.pi*self.var).sum(1)[None, :] +
                      ((x[:, None, :]-self.means)**2/self.var).sum(2))

    def forward(self, e, initial=None):
        a = np.empty_like(e)
        a[0] = np.log(self.start if initial is None else initial) + e[0]
        for t in range(1, len(e)):
            a[t] = e[t] + logsumexp(a[t-1, :, None] + np.log(self.transition), axis=0)
        return a

    def fit(self, x):
        x = np.asarray(x, float)
        labels = KMeans(self.k, random_state=self.seed, n_init=10).fit_predict(x)
        self.means = np.array([x[labels == j].mean(0) for j in range(self.k)])
        self.var = np.array([x[labels == j].var(0)+.05 for j in range(self.k)])
        self.start = np.full(self.k, 1/self.k)
        self.transition = .90*np.eye(self.k)+.10/self.k
        self.history = []
        for _ in range(self.n_iter):
            e = self.emissions(x)
            a = self.forward(e)
            ll = logsumexp(a[-1])
            b = np.zeros_like(e)
            lt = np.log(self.transition)
            for t in range(len(x)-2, -1, -1):
                b[t] = logsumexp(lt + e[t+1][None, :] + b[t+1][None, :], axis=1)
            gamma = np.exp(a+b-ll)
            counts = np.zeros((self.k, self.k))
            for t in range(len(x)-1):
                counts += np.exp(a[t, :, None]+lt+e[t+1][None, :]+b[t+1][None, :]-ll)
            self.start = np.maximum(gamma[0], 1e-8); self.start /= self.start.sum()
            self.transition = counts + 1e-6
            self.transition /= self.transition.sum(1, keepdims=True)
            mass = gamma.sum(0)[:, None] + 1e-12
            self.means = gamma.T @ x / mass
            self.var = np.maximum(gamma.T @ (x*x)/mass-self.means**2, .01)
            self.history.append(float(ll))
            if len(self.history)>1 and abs(ll-self.history[-2]) < self.tol*(1+abs(self.history[-2])):
                break
        return self

    def filter(self, x, previous=None):
        """P(S_t | X_1,...,X_t); never backward-smoothed or Viterbi labels."""
        initial = None if previous is None else previous @ self.transition
        a = self.forward(self.emissions(np.asarray(x)), initial)
        return np.exp(a-logsumexp(a, axis=1, keepdims=True))

    def score(self, x, previous=None):
        initial = None if previous is None else previous @ self.transition
        return float(logsumexp(self.forward(self.emissions(np.asarray(x)), initial)[-1]))
