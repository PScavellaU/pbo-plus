"""Minimal TuRBO-1 (Eriksson et al., NeurIPS 2019) on [-1,1]^d, batch Thompson sampling.

Reference baseline for the airfoil benchmarks. Uses the Matern-5/2 ARD GP of
pboplus.gp; for speed the GP is fitted on (at most) the n_fit points closest
to the trust-region centre. Restarts the trust region when it collapses.
"""
import numpy as np
from scipy.stats import qmc

from pboplus.gp import GPCritic


class TuRBO:
    def __init__(self, dim, batch=12, n_init=None, seed=0, n_fit=300, n_cand=None):
        self.d, self.q = dim, batch
        self.rng = np.random.default_rng(seed)
        self.n_init = n_init or 2 * dim
        self.n_fit = n_fit
        self.n_cand = n_cand or min(5000, 100 * dim)
        self.gp = GPCritic(dim, n_points=n_fit)
        self.X, self.y = [], []
        self.history, self.best_f, self.best_x = [], np.inf, None
        self._reset_tr()

    def _reset_tr(self):
        self.L, self.L_min, self.L_max = 0.8, 0.5 ** 7, 1.6
        self.succ, self.fail = 0, 0
        self.fail_tol = int(np.ceil(max(4.0 / self.q, self.d / self.q)))
        self.succ_tol = 3
        self.tr_X, self.tr_y = [], []

    def _eval(self, f, x):
        v = float(f(x))
        self.X.append(x), self.y.append(v), self.tr_X.append(x), self.tr_y.append(v)
        if v < self.best_f:
            self.best_f, self.best_x = v, x.copy()
        self.history.append((len(self.y), self.best_f))
        return v

    def optimize(self, f, budget):
        while len(self.y) < budget:
            if len(self.tr_y) < self.n_init:
                # Latin hypercube initial design of this trust region
                n = min(self.n_init, budget - len(self.y))
                S = qmc.LatinHypercube(d=self.d, seed=self.rng).random(n) * 2 - 1
                for x in S:
                    self._eval(f, x)
                continue
            X, y = np.asarray(self.tr_X), np.asarray(self.tr_y)
            ib = np.argmin(y)
            xc = X[ib]
            idx = np.argsort(np.sum((X - xc) ** 2, 1))[:self.n_fit]
            gp = self.gp.fit(X[idx], y[idx], optimize_hyper=True)
            w = gp.ell / np.exp(np.mean(np.log(gp.ell)))
            lb = np.clip(xc - w * self.L / 2, -1, 1)
            ub = np.clip(xc + w * self.L / 2, -1, 1)
            # Candidates: Sobol points in the TR, perturbing a random subset of coordinates
            sob = qmc.Sobol(d=self.d, scramble=True, seed=self.rng).random(self.n_cand)
            pert = lb + (ub - lb) * sob
            prob = min(20.0 / self.d, 1.0)
            mask = self.rng.random((self.n_cand, self.d)) <= prob
            mask[np.arange(self.n_cand), self.rng.integers(0, self.d, self.n_cand)] = True
            C = np.where(mask, pert, xc)
            # Thompson sampling on a subset (joint posterior samples)
            sub = C[self.rng.choice(len(C), min(len(C), 800), replace=False)]
            mu, cov = self._posterior(gp, sub)
            ev, E = np.linalg.eigh(0.5 * (cov + cov.T))
            Lc = E * np.sqrt(np.maximum(ev, 0.0))
            samples = mu[:, None] + Lc @ self.rng.standard_normal((len(sub), self.q))
            chosen = []
            for j in range(self.q):
                order = np.argsort(samples[:, j])
                k = next(i for i in order if i not in chosen)
                chosen.append(k)
            prev_best = np.min(self.tr_y)
            vals = [self._eval(f, sub[k]) for k in chosen[:budget - len(self.y)]]
            if min(vals) < prev_best - 1e-3 * abs(prev_best):
                self.succ, self.fail = self.succ + 1, 0
            else:
                self.succ, self.fail = 0, self.fail + 1
            if self.succ == self.succ_tol:
                self.L, self.succ = min(2 * self.L, self.L_max), 0
            elif self.fail == self.fail_tol:
                self.L, self.fail = self.L / 2, 0
            if self.L < self.L_min:
                self._reset_tr()
        return self.best_x, self.best_f

    @staticmethod
    def _posterior(gp, Zt):
        from pboplus.gp import _matern52
        from scipy.linalg import cho_solve
        Ks = gp.s2 * _matern52(Zt, gp.Z, gp.ell)
        mu = Ks @ gp.alpha
        v = cho_solve(gp.c, Ks.T)
        cov = gp.s2 * _matern52(Zt, Zt, gp.ell) - Ks @ v
        return mu * gp.ys + gp.ym, cov * gp.ys ** 2
