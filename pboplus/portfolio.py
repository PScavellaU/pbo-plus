"""Portfolio of PBO+ policies with a shared archive and successive halving.

For multimodal design spaces the outcome of one policy depends strongly on
the basin it falls into. The portfolio starts K policies from different
points of the box, lets them share every true evaluation (each critic is
fitted on the whole archive, in its own policy coordinates) and, at fixed
fractions of the budget, keeps only the best half of the policies (successive
halving, Jamieson & Talwalkar 2016), so that the end of the budget is spent
refining the most promising basin.
"""
import numpy as np

from pboplus.core import PBOPlus


class PortfolioPBO:
    def __init__(self, dim, lam=None, sigma0=0.25, n_policies=4, halving_at=(0.3, 0.6),
                 spread=0.5, seed=None, **kw):
        self.d = dim
        rng = np.random.default_rng(seed)
        # first policy at the baseline (a = 0), the others spread around it
        means = [np.zeros(dim)] + [np.clip(rng.uniform(-spread, spread, dim), -1, 1) for _ in range(n_policies - 1)]
        self.pols = [PBOPlus(dim, lam=lam, sigma0=sigma0, mean0=m, seed=None if seed is None else seed * 100 + i, **kw)
                     for i, m in enumerate(means)]
        self.alive = list(range(n_policies))
        self.own_best = [np.inf] * n_policies
        self.halving_at = halving_at
        self.history, self.best_f, self.best_x, self.n_eval = [], np.inf, None, 0

    def optimize(self, f, budget, batch=False):
        stage = 0
        while self.n_eval < budget:
            for i in list(self.alive):
                if self.n_eval >= budget:
                    break
                p = self.pols[i]
                n0 = len(p.y)
                p.batch = batch
                p.step(f, budget - self.n_eval)
                Xn, yn = p.X[n0:], p.y[n0:]
                self.own_best[i] = min([self.own_best[i]] + list(yn))
                for x, v in zip(Xn, yn):
                    self.n_eval += 1
                    if v < self.best_f:
                        self.best_f, self.best_x = v, x.copy()
                    self.history.append((self.n_eval, self.best_f))
                    for j in self.alive:            # share the evaluation with the others
                        if j != i:
                            self.pols[j].X.append(x.copy())
                            self.pols[j].y.append(v)
            if stage < len(self.halving_at) and self.n_eval >= self.halving_at[stage] * budget and len(self.alive) > 1:
                keep = max(1, len(self.alive) // 2)
                self.alive = sorted(self.alive, key=lambda i: self.own_best[i])[:keep]
                stage += 1
        return self.best_x, self.best_f
