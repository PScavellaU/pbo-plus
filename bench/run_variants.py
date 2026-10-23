"""Run PBO+ variants and CMA-ES references on the PBO benchmarks.

Saves bench/variants/<problem>/<variant>.npy : array (n_seeds, budget) with the
best cost found after each true function evaluation.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per process: avoids oversubscription
import sys
import argparse
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pboplus.core import PBOPlus          # noqa: E402
from pboplus.benchmarks import get_problems as _paper, get_holdout_problems  # noqa: E402


def get_problems():
    from pboplus.airfoil import get_airfoil_problems
    from pboplus.benchmarks import get_dev_multimodal
    return {**_paper(), **get_holdout_problems(), **get_dev_multimodal(), **get_airfoil_problems()}

SIGMA0 = 0.4   # ~ initial std of the original PBO (0.85 * sigmoid(.) at init)
SIGMA0_AIRFOIL = 0.25

VARIANTS = {
    'pbo+':               dict(update='cma', critic=True),
    'pbo+_noinject':      dict(update='cma', critic=True, inject=False),
    'pbo+_noincremental': dict(update='cma', critic=True, incremental=False),
    'pbo+_nocritic':      dict(update='cma', critic=False),
    'xnes':               dict(update='xnes', critic=False),
    'xnes_reuse':         dict(update='xnes', critic=False, reuse=True),
    'pbo+_gp':            dict(update='cma', critic=True, critic_type='gp'),
    'pbo+_auto':          dict(update='cma', critic=True, critic_type='auto'),
    'pbo+_gp_cheap':      dict(update='cma', critic=True, critic_type='gp', use_cheap=True),
    'pbo+_cheap':         dict(update='cma', critic=True, use_cheap=True),
    'pbo+_portfolio':     dict(portfolio=True, update='cma', critic=True),
    'pbo+_gp_portfolio':  dict(portfolio=True, update='cma', critic=True, critic_type='gp'),
    'turbo':           'turbo',
    'cma-es':          'cma',
    'cma-es_cheap':    'cma_cheap',
    'lq-cma-es':       'lq',
}


def curve(history, budget):
    out = np.full(budget, np.nan)
    for n, b in history:
        if n <= budget:
            out[n - 1] = b
    # forward fill
    for i in range(1, budget):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def run_one(args):
    pname, vname, seed = args
    p = get_problems()[pname]
    v = VARIANTS[vname]
    # Airfoil boxes are wide w.r.t. good designs: smaller initial spread, same for every method
    sigma0 = SIGMA0_AIRFOIL if pname.startswith('af_') else SIGMA0
    if isinstance(v, dict) and v.get('portfolio'):
        from pboplus.portfolio import PortfolioPBO
        kw = {k: x for k, x in v.items() if k != 'portfolio'}
        opt = PortfolioPBO(p.dim, lam=p.n_ind, sigma0=kw.pop('sigma0', sigma0), seed=seed, **kw)
        opt.optimize(p, p.budget)
        return curve(opt.history, p.budget)
    if isinstance(v, dict):
        kw = dict(v)
        if kw.pop('use_cheap', False):
            kw['cheap'] = p.cheap
        lam = p.n_ind * kw.pop('lam_factor', 1)
        opt = PBOPlus(p.dim, lam=lam, sigma0=kw.pop('sigma0', sigma0), seed=seed, **kw)
        opt.optimize(p, p.budget)
        return curve(opt.history, p.budget)
    if v == 'turbo':
        sys.path.insert(0, os.path.join(ROOT, 'baselines'))
        from turbo import TuRBO
        opt = TuRBO(p.dim, batch=p.n_ind, seed=seed)
        opt.optimize(p, p.budget)
        return curve(opt.history, p.budget)
    import cma
    hist, best = [], [np.inf]

    def f(x):
        v_ = p(np.asarray(x))
        best[0] = min(best[0], v_)
        hist.append((len(hist) + 1, best[0]))
        return v_
    opts = {'popsize': p.n_ind, 'bounds': [-1, 1], 'seed': seed + 1, 'verbose': -9,
            'maxfevals': p.budget, 'tolfun': 0, 'tolx': 0, 'tolfunhist': 0, 'tolflatfitness': 1e9}
    try:
        if v == 'cma':
            es = cma.CMAEvolutionStrategy(np.zeros(p.dim), sigma0, opts)
            while len(hist) < p.budget and not es.stop():
                X = es.ask()
                es.tell(X, [f(x) for x in X])
        elif v == 'cma_cheap':
            # Same knowledge as PBO+ 'cheap': geometrically infeasible candidates are
            # not evaluated (no budget) and ranked behind every evaluated one.
            es = cma.CMAEvolutionStrategy(np.zeros(p.dim), sigma0, opts)
            while len(hist) < p.budget and not es.stop():
                X = es.ask()
                pc = [p.cheap(np.clip(x, -1, 1)) for x in X]
                vals = [f(x) if c <= 0 else None for x, c in zip(X, pc)]
                worst = max([v_ for v_ in vals if v_ is not None], default=0.0)
                es.tell(X, [v_ if v_ is not None else worst + 1.0 + c for v_, c in zip(vals, pc)])
        else:
            o = {k: opts[k] for k in ('bounds', 'seed', 'verbose', 'maxfevals', 'popsize')}
            cma.fmin_lq_surr2(f, np.zeros(p.dim), sigma0, options=o)
    except Exception as e:  # budget exhausted inside the library, or stop condition
        if len(hist) < 2:
            raise e
    return curve(hist, p.budget)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', default=','.join(_paper()))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--cpu', type=int, default=2)
    a = ap.parse_args()
    jobs = [(p, v, s) for p in a.problems.split(',') for v in a.variants.split(',') for s in range(a.seeds)]
    # Each run is cached in bench/variants/<problem>/runs/<variant>_<seed>.npy as soon as it ends
    def cache(p, v, s):
        return os.path.join(ROOT, 'bench', 'variants', p, 'runs', f'{v}_{s}.npy')
    todo = [j for j in jobs if not os.path.exists(cache(*j))]
    with Pool(a.cpu) as pool:
        for j, r in zip(todo, pool.imap(run_one, todo, chunksize=1)):
            os.makedirs(os.path.dirname(cache(*j)), exist_ok=True)
            np.save(cache(*j), r)
            print('finished', *j, f'{r[-1]:.4g}', flush=True)
    out = {}
    for j in jobs:
        out.setdefault(j[:2], []).append(np.load(cache(*j)))
    for (p, v), rs in out.items():
        d = os.path.join(ROOT, 'bench', 'variants', p)
        np.save(os.path.join(d, v + '.npy'), np.array(rs))
        fin = np.array(rs)[:, -1]
        print(f'{p:18s} {v:16s} median {np.median(fin):.2e}  q25 {np.quantile(fin, .25):.2e}  q75 {np.quantile(fin, .75):.2e}', flush=True)
