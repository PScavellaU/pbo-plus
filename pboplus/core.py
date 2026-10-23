"""PBO+ : an improved policy-based optimization.

PBO (Viquerat et al., 2023) is a single-step policy gradient on a Gaussian
policy N(mu, Sigma) whose parameters come out of three small networks fed with
a constant state. PBO+ keeps the same view (a Gaussian policy improved from
batches of single-step rewards) but changes how the policy is parametrized,
how the advantage is built and how the gradient is taken:

1. Direct parametrization  Sigma = sigma^2 B B^T  (no constant-input networks:
   they only add an ill-conditioned reparametrization of the same 2d+d(d-1)/2
   numbers).
2. Natural policy gradient in local coordinates (exponential map update of
   xNES, Glasmachers et al. 2010): invariant to linear transformations of the
   search space, no learning-rate tuning of Adam on network weights.
3. Rank-based utilities instead of z-score advantages with clipping/decay
   (invariant to monotone transformations of the reward; robust to outliers
   such as diverged CFD runs).
4. Mirrored (antithetic) sampling: variance reduction of the gradient.
5. Optional importance-weighted reuse of the previous generation (the role of
   PBO's history buffers, done without bias).
6. A critic Q(a): a local quadratic surrogate of the cost (lq-CMA-ES style,
   Hansen 2019) that ranks every sampled action; only the actions the critic
   cannot be trusted on are evaluated with the true (expensive) function. The
   number of true evaluations per generation adapts to the critic's Kendall-tau
   rank correlation on held-out data.
7. IPOP-style restarts on stagnation.

Costs are MINIMIZED. Actions live in the box [-1, 1]^d like in PBO.
"""
import numpy as np
from scipy.linalg import expm
from scipy.stats import kendalltau

from pboplus.gp import GPCritic
GPCriticBase = GPCritic


def xnes_utilities(lam):
    r = np.arange(1, lam + 1)
    u = np.maximum(0.0, np.log(lam / 2.0 + 1.0) - np.log(r))
    return u / u.sum() - 1.0 / lam


class QuadraticCritic:
    """Local polynomial surrogate Q(a) ~ cost, fitted in the policy's own coordinates.

    The model degree grows with the data available: linear, then diagonal
    quadratic, then full quadratic (the latter is exact on a convex quadratic,
    and therefore recovers the full covariance of the problem like CMA-ES).
    """

    def __init__(self, dim, rank_weights=False, max_weight=20.0, truncation=0.75):
        self.d = dim
        self.n_lin, self.n_diag = dim + 1, 2 * dim + 1
        self.n_full = (dim + 1) * (dim + 2) // 2
        # lq-CMA-ES style: drop the worst points, weight the others linearly by rank
        self.rank_weights, self.max_weight, self.truncation = rank_weights, max_weight, truncation

    def _features(self, z, kind):
        cols = [np.ones(len(z)), *z.T]
        if kind in ('diag', 'full'):
            cols += list((z ** 2).T)
        if kind == 'full':
            iu = np.triu_indices(self.d, 1)
            cols += list((z[:, iu[0]] * z[:, iu[1]]).T)
        return np.stack(cols, axis=1)

    def fit(self, Z, y):
        n = len(y)
        if n >= int(1.1 * self.n_full) + 1:
            kind = 'full'
        elif n >= self.n_diag + 1:
            kind = 'diag'
        else:
            kind = 'lin'
        if self.rank_weights:
            n_dof = {'lin': self.n_lin, 'diag': self.n_diag, 'full': self.n_full}[kind]
            keep = np.argsort(y)[:max(n_dof + 2, int(self.truncation * (n + 1)))]
            Z, y = Z[keep], y[keep]
            w = np.linspace(self.max_weight, 1.0, len(y))
        else:
            # Weighted least squares: points near the current mean dominate.
            w = 1.0 / (1.0 + 0.1 * np.sum(Z ** 2, axis=1))
        F = self._features(Z, kind) * w[:, None]
        coef, *_ = np.linalg.lstsq(F, y * w, rcond=None)
        self.kind, self.coef = kind, coef
        return self

    def predict(self, Z):
        return self._features(Z, self.kind) @ self.coef

    def local_size(self, n, model_size, min_pop):
        return max(int(model_size * self.n_full), min_pop * n)


class PBOPlus:
    """Gaussian policy improved by natural policy gradient steps.

    update='cma' : natural gradient (rank-mu) plus cumulation: momentum of the
                   mean step (evolution path, rank-one update) and path-length
                   step-size control, as in CMA-ES. This is the PBO update
                   brought to its fixed point: PBO tries to mimic these
                   heuristics with Adam on networks.
    update='xnes': plain natural gradient with exponential map (xNES).
    """

    def __init__(self, dim, lam=None, sigma0=0.3, mean0=None, update='cma',
                 mirrored=None, critic=True, reuse=False, restarts=True,
                 seed=None, tau_trust=0.85, tau_bad=0.5, inject=True,
                 incremental=True, pop_factor=1, model_size=1.5, model_min_pop=1,
                 restart_mode='ipop', stall_gens=None, critic_type='quadratic', gp_beta=0.0,
                 tau_mode=None, rank_weights=False, local_select='nearest', min_batch=1):
        self.d = dim
        self.rng = np.random.default_rng(seed)
        self.lam0 = lam or 4 + int(3 * np.log(dim))
        self.sigma0 = sigma0
        self.mean0 = np.zeros(dim) if mean0 is None else np.asarray(mean0, float)
        self.update = update
        # Mirroring biases path-length control: default on for xNES only
        self.mirrored = (update == 'xnes') if mirrored is None else mirrored
        self.use_critic, self.restarts = critic, restarts
        # 'ipop': double lambda, restart from the best point
        # 'bipop': BIPOP regimes (Hansen 2009), random restart points
        # 'nbc': basin-aware restarts seeded by nearest-better clustering of the archive
        self.restart_mode, self.stall_gens = restart_mode, stall_gens
        self.n_restarts, self.used = 0, {'large': 0, 'small': 0}
        self.regime, self.lam_large = 'large', None
        self.basins = []                 # end points of previous runs
        self.sigma_init = sigma0
        self.inject = inject and critic
        self.incremental = incremental
        self.model_size, self.model_min_pop = model_size, model_min_pop
        # With a critic most samples are ranked for free: sample more of them
        self.pop_factor = pop_factor if critic else 1
        self.reuse = reuse and update == 'xnes'
        self.tau_trust, self.tau_bad = tau_trust, tau_bad
        # A GP interpolates its data: only out-of-sample rank correlation is informative
        self.tau_mode = tau_mode or {'quadratic': 'insample', 'gp': 'oos', 'auto': 'dynamic'}[critic_type]
        self.pred_log = []
        self.local_select = local_select
        self.min_batch, self.batch = max(1, int(min_batch)), False
        if critic_type == 'gp':
            self.critic = GPCritic(dim, beta=gp_beta)
        elif critic_type == 'auto':
            from pboplus.gp import AutoCritic
            self.critic = AutoCritic(dim, QuadraticCritic(dim, rank_weights), GPCritic(dim, beta=gp_beta))
        else:
            self.critic = QuadraticCritic(dim, rank_weights)
        self.X, self.y = [], []          # archive of true evaluations
        self.n_eval = 0
        self.best_x, self.best_f = None, np.inf
        self.history = []                # (n_eval, best_f) after each true evaluation
        self._init_policy(self.lam0, self.mean0)

    # ------------------------------------------------------------------ policy
    def _init_policy(self, lam, mean):
        d = self.d
        self.lam = lam
        self.npop = lam * self.pop_factor      # sampled actions per generation
        self.m = mean.copy()
        self.sigma = self.sigma_init
        self.B = np.eye(d)               # Sigma = sigma^2 B B^T
        self.prev = None
        self.n_true = lam                # true evaluations per generation (critic-adapted)
        self.gen, self.best_gen, self.restart_best = 0, 0, np.inf
        if self.update == 'xnes':
            self.eta_s = self.eta_B = (9.0 + 3.0 * np.log(d)) / (5.0 * d * np.sqrt(d))
        else:
            mu = self.npop // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            self.w = w / w.sum()
            self.mueff = 1.0 / np.sum(self.w ** 2)
            me = self.mueff
            self.cc = (4 + me / d) / (d + 4 + 2 * me / d)
            self.cs = (me + 2) / (d + me + 5)
            self.c1 = 2 / ((d + 1.3) ** 2 + me)
            self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((d + 2) ** 2 + me))
            self.damps = 1 + 2 * max(0, np.sqrt((me - 1) / (d + 1)) - 1) + self.cs
            self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
            self.C, self.ps, self.pc = np.eye(d), np.zeros(d), np.zeros(d)

    def _A(self):
        return self.sigma * self.B

    def _to_z(self, X):
        return np.linalg.solve(self._A(), (X - self.m).T).T

    @staticmethod
    def _reflect(x):
        # Mirror reflection into [-1, 1] (period 4)
        y = np.mod(x + 1.0, 4.0)
        return np.where(y > 2.0, 4.0 - y, y) - 1.0

    # ---------------------------------------------------------------- critic
    def _fit_local(self, X, y, n):
        """Fit the critic on the archive points closest to the policy (Mahalanobis)."""
        Z = self._to_z(X)
        k = min(len(y), self.critic.local_size(n, self.model_size, self.model_min_pop))
        if self.local_select == 'recent':
            idx = np.arange(len(y))[::-1][:k]
        else:
            idx = np.argsort(np.sum(Z ** 2, axis=1))[:k]
        if isinstance(self.critic, GPCriticBase) or hasattr(self.critic, 'record'):
            # Hyper-parameters once per generation; later refits only update the data
            fresh = getattr(self, '_hyper_gen', None) != (self.n_restarts, self.gen)
            self._hyper_gen = (self.n_restarts, self.gen)
            return self.critic.fit(Z[idx], y[idx], optimize_hyper=fresh)
        return self.critic.fit(Z[idx], y[idx])

    def _critic_ranks(self, Xc, f, budget_left, force=()):
        """Rank candidates with the critic, evaluate with f only what is needed."""
        n = len(Xc)
        vals = np.full(n, np.nan)
        if self.incremental and len(self.y) >= self.critic.n_lin + 2:
            return self._critic_ranks_incremental(Xc, f, budget_left, force)
        if len(self.y) < self.critic.n_lin + 2:
            order, n_true = np.arange(n), n
        else:
            q = self._fit_local(np.asarray(self.X), np.asarray(self.y), n).predict(self._to_z(Xc))
            order, n_true = np.argsort(q), min(n, max(1, self.n_true))
        todo = list(force) + [i for i in order[:n_true] if i not in force]
        idx = todo[:max(n_true, len(force))][:budget_left]
        vals[idx] = self._eval_many(Xc[idx], f)
        ev = ~np.isnan(vals)
        if not ev.all():
            q = self._fit_local(np.asarray(self.X), np.asarray(self.y), n).predict(self._to_z(Xc))
            # Pessimistic offset: a predicted point never beats a verified one by model error
            vals[~ev] = q[~ev] + max(0.0, np.max(vals[ev] - q[ev]))
        self._adapt_n_true(n)
        return vals

    def _critic_ranks_incremental(self, Xc, f, budget_left, force):
        """Evaluate the critic's best candidates in growing chunks until the
        critic ranks the recent true evaluations well (Kendall tau), as in lq-CMA-ES."""
        n = len(Xc)
        vals = np.full(n, np.nan)
        Zc = self._to_z(Xc)
        X, y = np.asarray(self.X), np.asarray(self.y)
        q = self._fit_local(X, y, n).predict(Zc)
        todo = list(force)
        chunk = max(self.min_batch, len(todo))
        while True:
            rest = [i for i in np.argsort(q) if np.isnan(vals[i]) and i not in todo]
            todo += rest[:chunk - len(todo)]
            todo = todo[:budget_left]
            vals[todo] = self._eval_many(Xc[todo], f)
            for i in todo:
                self.pred_log.append((q[i], vals[i]))   # prediction made before the evaluation
                if hasattr(self.critic, 'record'):
                    self.critic.record(i, vals[i])
            budget_left -= len(todo)
            todo = []
            if np.all(~np.isnan(vals)) or budget_left <= 0:
                break
            X, y = np.asarray(self.X), np.asarray(self.y)
            c = self._fit_local(X, y, n)
            q = c.predict(Zc)
            oos = self.tau_mode == 'oos' or (self.tau_mode == 'dynamic' and c.kind == 'gp')
            if oos:
                # Out-of-sample: predictions made before each point was evaluated
                pl = np.array(self.pred_log[-max(10, n):])
                tau = kendalltau(pl[:, 0], pl[:, 1])[0] if len(pl) >= 5 else np.nan
            else:
                n_tau = min(len(y), max(15, n))
                tau = kendalltau(c.predict(self._to_z(X[-n_tau:])), y[-n_tau:])[0]
            if np.isfinite(tau) and tau >= self.tau_trust:
                break
            chunk = int(np.ceil(1.5 * chunk)) if chunk > 1 else 2
        ev = ~np.isnan(vals)
        if not ev.all():
            vals[~ev] = q[~ev] + max(0.0, np.max(vals[ev] - q[ev]))
        return vals

    def _adapt_n_true(self, n):
        """Held-out Kendall tau of the critic on the latest true evaluations."""
        n_val = max(4, self.n_true)
        if len(self.y) < self.critic.n_lin + 2 + n_val:
            return
        X, y = np.asarray(self.X), np.asarray(self.y)
        q = self._fit_local(X[:-n_val], y[:-n_val], n).predict(self._to_z(X[-n_val:]))
        tau = kendalltau(q, y[-n_val:])[0]
        if np.isnan(tau):
            tau = 0.0
        if tau >= self.tau_trust:
            self.n_true = max(1, self.n_true // 2)
        elif tau < self.tau_bad:
            self.n_true = min(n, 2 * self.n_true)

    def _critic_argmin(self):
        """Actor step from the critic: minimizer of the local quadratic model (z coords).

        A deterministic-policy-gradient style action: when the fitted critic
        is a convex quadratic, its minimizer (restricted to a trust region of
        radius sqrt(d)+2 in sigma units) is injected into the population.
        """
        d = self.d
        if len(self.y) < self.critic.n_diag + 2:
            return None
        c = self._fit_local(np.asarray(self.X), np.asarray(self.y), self.lam)
        if c.kind == 'gp':
            zb = self._to_z(self.best_x[None])[0] if self.best_x is not None else None
            return c.argmin_trust(np.sqrt(d) + 2.0, self.rng, zb)
        if c.kind == 'lin':
            return None
        g = c.coef[1:d + 1]
        H = np.diag(2.0 * c.coef[d + 1:2 * d + 1])
        if c.kind == 'full':
            iu = np.triu_indices(d, 1)
            H[iu] = c.coef[2 * d + 1:]
            H[(iu[1], iu[0])] = c.coef[2 * d + 1:]
        try:
            if np.linalg.eigvalsh(H).min() <= 0:
                return None
            z = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        r = np.sqrt(d) + 2.0
        nz = np.linalg.norm(z)
        return z if nz <= r else z * (r / nz)

    # ------------------------------------------------------------- main loop
    def _eval_many(self, Xs, f):
        """True evaluations of a block of actions: one call with the whole block when
        the objective is a batch function (parallel CFD runs), else one call each."""
        if len(Xs) == 0:
            return np.zeros(0)
        if self.batch:
            vs = np.asarray(f(np.asarray(Xs)), float).ravel()
            return np.array([self._true_eval(x, None, v) for x, v in zip(Xs, vs)])
        return np.array([self._true_eval(x, f) for x in Xs])

    def _true_eval(self, x, f, v=None):
        v = float(f(x)) if v is None else float(v)
        self.X.append(x.copy()), self.y.append(v)
        self.n_eval += 1
        if v < self.best_f:
            self.best_f, self.best_x = v, x.copy()
        self.history.append((self.n_eval, self.best_f))
        return v

    def step(self, f, budget_left):
        d, lam = self.d, self.npop
        if self.mirrored:
            h = self.rng.standard_normal(((lam + 1) // 2, d))
            Z = np.vstack([h, -h])[:lam]
        else:
            Z = self.rng.standard_normal((lam, d))
        inj = self._critic_argmin() if self.inject else None
        if inj is not None:
            Z[-1] = inj
        X_raw = self.m + Z @ self._A().T
        # The policy optimizes f o reflect: the gradient uses the sampled z,
        # the function is evaluated at the reflection of x into the box.
        X = self._reflect(X_raw)

        if self.use_critic and len(self.y) > 0:
            vals = self._critic_ranks(X, f, budget_left, force=(lam - 1,) if inj is not None else ())
        else:
            vals = self._eval_many(X[:budget_left], f)
            Z, X_raw = Z[:len(vals)], X_raw[:len(vals)]
        if len(vals) < 2:
            return

        if self.update == 'xnes':
            self._xnes_step(Z, X_raw, vals)
        else:
            if inj is not None:
                # Injected solutions: clip the Mahalanobis length (Hansen 2011)
                cy = np.sqrt(d) + 2.0 * d / (d + 2.0)
                Z[-1] *= min(1.0, cy / max(np.linalg.norm(Z[-1]), 1e-300))
            self._cma_step(Z, vals)
        # The box has width 2: a larger spread only samples its reflections
        self.sigma = min(self.sigma, 1.0 / np.linalg.norm(self.B, 2))

        # Stagnation -> IPOP restart (double population, restart from best)
        self.gen += 1
        # Progress is measured on true evaluations only (critic values can be optimistic)
        if self.best_f < self.restart_best:
            self.restart_best, self.best_gen = self.best_f, self.gen
        tiny = self.sigma * np.linalg.norm(self.B, 2) < 1e-15
        stall = self.stall_gens if self.stall_gens is not None else 120 + 30 * d / lam
        stuck = self.gen - self.best_gen > stall
        flat = np.ptp(vals) == 0.0
        self.used[self.regime] += lam
        if self.restarts and (tiny or stuck or flat):
            self._restart()

    def _restart(self):
        self.n_restarts += 1
        self.basins.append(self.m.copy())
        if self.restart_mode == 'ipop':
            self.sigma_init = self.sigma0
            self._init_policy(2 * self.lam, self.best_x if self.best_x is not None else self.mean0)
            return
        d = self.d
        if self.restart_mode == 'nbc':
            seed = self._nbc_seed()
            if seed is not None:
                x0, r = seed
                self.sigma_init = float(np.clip(r / np.sqrt(d), 0.05, self.sigma0))
                self._init_policy(self.lam0, x0)
                return
        # BIPOP: alternate a doubling large population and cheap small runs
        self.lam_large = 2 * (self.lam_large or self.lam0)
        if self.n_restarts == 1 or self.used['small'] >= self.used['large']:
            self.regime, lam, self.sigma_init = 'large', self.lam_large, self.sigma0
        else:
            u = self.rng.uniform()
            self.regime = 'small'
            lam = int(self.lam0 * (0.5 * self.lam_large / self.lam0) ** (u ** 2))
            self.sigma_init = self.sigma0 * 10 ** (-2 * u)
        self._init_policy(max(lam, self.lam0), self.rng.uniform(-0.8, 0.8, d))

    def _nbc_seed(self):
        """Nearest-better clustering (Preuss 2010) on the archive: a point far from
        every better point heads its own basin. Return the best unexplored head."""
        X, y = np.asarray(self.X), np.asarray(self.y)
        if len(y) < 4 * self.d:
            return None
        o = np.argsort(y)[:300]
        X, y = X[o], y[o]
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        nb = np.array([D[i, :i].min() if i else np.inf for i in range(len(y))])
        thr = 2.0 * np.mean(nb[1:])
        for i in np.nonzero(nb > thr)[0]:
            if all(np.linalg.norm(X[i] - b) > 0.5 * min(nb[i], 1.0) for b in self.basins):
                return X[i].copy(), min(nb[i], 1.0) / 2.0
        return None

    def _logdet_A(self):
        return np.sum(np.log(np.abs(np.diag(np.linalg.qr(self._A())[1]))))

    def _xnes_step(self, Z, X_raw, vals):
        d = self.d
        w = np.ones(len(vals))
        n_cur = len(vals)
        # Importance-weighted reuse of the previous generation (PBO's history buffer)
        if self.reuse and self.prev is not None:
            Xp, vp, logp_old = self.prev
            Zp = self._to_z(Xp)
            logp_new = -0.5 * np.sum(Zp ** 2, axis=1) - self._logdet_A()
            Z, vals = np.vstack([Z, Zp]), np.concatenate([vals, vp])
            w = np.concatenate([w, np.minimum(1.0, np.exp(logp_new - logp_old))])
        logp = -0.5 * np.sum(Z[:n_cur] ** 2, axis=1) - self._logdet_A()
        self.prev = (X_raw.copy(), vals[:n_cur].copy(), logp)

        U = np.empty(len(vals))
        U[np.argsort(vals)] = xnes_utilities(len(vals))
        U = U * w
        G_d = U @ Z
        G_M = (Z.T * U) @ Z - U.sum() * np.eye(d)
        G_s = np.trace(G_M) / d
        G_B = G_M - G_s * np.eye(d)
        self.m = self.m + self.sigma * self.B @ G_d
        self.sigma *= np.exp(0.5 * self.eta_s * G_s)
        self.B = self.B @ expm(0.5 * self.eta_B * G_B)

    def _cma_step(self, Z, vals):
        d, mu = self.d, min(len(self.w), len(vals) // 2 or 1)
        w = self.w[:mu] / self.w[:mu].sum()
        sel = np.argsort(vals)[:mu]
        Y = Z[sel] @ self.B.T                    # steps in units of sigma
        y_w = w @ Y
        self.m = self.m + self.sigma * y_w
        # Path-length (CSA) step-size control: a momentum test on the mean steps
        invsqrtC = np.linalg.inv(self.B)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC @ y_w
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (d + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        # Rank-one (momentum) + rank-mu (natural gradient) covariance update
        dh = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu + self.c1 * dh) * self.C
                  + self.c1 * np.outer(self.pc, self.pc) + self.cmu * (Y.T * w) @ Y)
        self.sigma *= np.exp(min(1.0, (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)))
        self.C = 0.5 * (self.C + self.C.T)
        ev, E = np.linalg.eigh(self.C)
        self.B = E @ np.diag(np.sqrt(np.maximum(ev, 1e-300))) @ E.T

    def optimize(self, f, budget, batch=False):
        """Minimize f over [-1,1]^d with `budget` true evaluations.

        batch=True: f receives an (n, d) array and returns n costs, so that each
        block (whole generation, or critic chunk of at least `min_batch` actions)
        can be evaluated in parallel, e.g. as simultaneous CFD jobs.
        """
        self.batch = batch
        while self.n_eval < budget:
            self.step(f, budget - self.n_eval)
        return self.best_x, self.best_f
