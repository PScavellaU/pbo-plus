"""Gaussian-process critic for PBO+ (non-quadratic, multimodal landscapes).

Matern-5/2 kernel with ARD length scales, fitted in the policy's own
coordinates z = A^-1 (x - m), so that an isotropic prior is already a good
guess. Hyper-parameters maximize the log marginal likelihood (L-BFGS-B,
warm-started from the previous fit). Same interface as QuadraticCritic.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

SQ5 = np.sqrt(5.0)


def _matern52(A, B, ell):
    D = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) / ell) ** 2, 0.0).sum(-1))
    return (1.0 + SQ5 * D + 5.0 / 3.0 * D ** 2) * np.exp(-SQ5 * D)


class GPCritic:
    kind = 'gp'

    def __init__(self, dim, n_points=None, beta=0.0):
        self.d = dim
        self.n_lin, self.n_diag = dim + 1, 2 * dim + 1
        self.n_full = (dim + 1) * (dim + 2) // 2
        self.n_points = n_points or int(min(250, max(40, 10 * dim)))
        self.beta = beta                     # LCB exploration weight for the actor step
        self.theta = np.concatenate([np.zeros(dim) + np.log(2.0), [0.0, np.log(1e-4)]])
        self._fitted_hyper = 0

    def local_size(self, n, model_size, min_pop):
        return max(self.n_points, min_pop * n)

    def _nll(self, th, Z, y):
        """Negative log marginal likelihood and its analytic gradient."""
        d, n = self.d, len(y)
        ell, s2, nz = np.exp(th[:d]), np.exp(th[d]), np.exp(th[d + 1])
        Dk = (Z[:, None, :] - Z[None, :, :]) ** 2 / ell ** 2          # n x n x d
        r = np.sqrt(np.maximum(Dk.sum(-1), 0.0))
        e = np.exp(-SQ5 * r)
        K0 = (1.0 + SQ5 * r + 5.0 / 3.0 * r ** 2) * e
        K = s2 * K0 + (nz + 1e-8) * np.eye(n)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        a = cho_solve(c, y)
        nll = 0.5 * y @ a + np.sum(np.log(np.diag(c[0]))) + 0.5 * n * np.log(2 * np.pi)
        W = cho_solve(c, np.eye(n)) - np.outer(a, a)                  # K^-1 - a a^T
        g = np.empty_like(th)
        dK_r = s2 * 5.0 / 3.0 * (1.0 + SQ5 * r) * e                   # times Dk_k -> dK/dlog(ell_k)
        g[:d] = 0.5 * np.einsum('ij,ijk->k', W * dK_r, Dk)
        g[d] = 0.5 * np.sum(W * (s2 * K0))
        g[d + 1] = 0.5 * np.trace(W) * nz
        return nll, g

    def fit(self, Z, y, optimize_hyper=True):
        self.ym, self.ys = float(np.mean(y)), float(np.std(y) + 1e-12)
        yn = (y - self.ym) / self.ys
        if optimize_hyper:
            lb = np.concatenate([np.full(self.d, np.log(0.05)), [np.log(0.05), np.log(1e-6)]])
            ub = np.concatenate([np.full(self.d, np.log(50.0)), [np.log(20.0), np.log(0.5)]])
            th0 = np.clip(self.theta, lb, ub)
            r = minimize(self._nll, th0, args=(Z, yn), jac=True, method='L-BFGS-B', bounds=list(zip(lb, ub)),
                         options=dict(maxiter=50))
            if np.isfinite(r.fun):
                self.theta = r.x
        ell, s2, nz = np.exp(self.theta[:self.d]), np.exp(self.theta[self.d]), np.exp(self.theta[self.d + 1])
        K = s2 * _matern52(Z, Z, ell) + (nz + 1e-8) * np.eye(len(y))
        jitter = 1e-8
        while True:
            try:
                self.c = cho_factor(K + jitter * np.eye(len(y)), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        self.Z, self.alpha = Z, cho_solve(self.c, yn)
        self.ell, self.s2 = ell, s2
        return self

    def predict(self, Zt, return_std=False):
        Ks = self.s2 * _matern52(Zt, self.Z, self.ell)
        mu = Ks @ self.alpha
        if not return_std:
            return mu * self.ys + self.ym
        v = cho_solve(self.c, Ks.T)
        var = np.maximum(self.s2 - np.sum(Ks * v.T, axis=1), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys

    def argmin_trust(self, r, rng, z_best=None, n_cand=2000):
        """Approximate minimizer of the GP lower confidence bound in the ball |z| <= r."""
        Zc = rng.standard_normal((n_cand, self.d))
        if z_best is not None:
            Zc[: n_cand // 4] = z_best + 0.3 * rng.standard_normal((n_cand // 4, self.d))
        nrm = np.linalg.norm(Zc, axis=1, keepdims=True)
        Zc = np.where(nrm > r, Zc * (r / nrm), Zc)
        mu, sd = self.predict(Zc, return_std=True)
        return Zc[np.argmin(mu - self.beta * sd)]
