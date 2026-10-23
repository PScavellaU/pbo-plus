"""Run PBO+ variants and CMA-ES references on the PBO benchmarks.

Saves bench/variants/<problem>/<variant>.npy : array (n_seeds, budget) with the
best cost found after each true function evaluation.
"""
import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pboplus.core import PBOPlus          # noqa: E402
from pboplus.benchmarks import get_problems as _paper, get_holdout_problems  # noqa: E402


def get_problems():
    return {**_paper(), **get_holdout_problems()}

SIGMA0 = 0.4   # ~ initial std of the original PBO (0.85 * sigmoid(.) at init)

VARIANTS = {
    'pbo+':               dict(update='cma', critic=True),
    'pbo+_noinject':      dict(update='cma', critic=True, inject=False),
    'pbo+_noincremental': dict(update='cma', critic=True, incremental=False),
    'pbo+_nocritic':      dict(update='cma', critic=False),
    'xnes':               dict(update='xnes', critic=False),
    'xnes_reuse':         dict(update='xnes', critic=False, reuse=True),
    'cma-es':          'cma',
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
    if isinstance(v, dict):
        kw = dict(v)
        lam = p.n_ind * kw.pop('lam_factor', 1)
        opt = PBOPlus(p.dim, lam=lam, sigma0=SIGMA0, seed=seed, **kw)
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
            es = cma.CMAEvolutionStrategy(np.zeros(p.dim), SIGMA0, opts)
            while len(hist) < p.budget and not es.stop():
                X = es.ask()
                es.tell(X, [f(x) for x in X])
        else:
            o = {k: opts[k] for k in ('bounds', 'seed', 'verbose', 'maxfevals', 'popsize')}
            cma.fmin_lq_surr2(f, np.zeros(p.dim), SIGMA0, options=o)
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
    with Pool(a.cpu) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    out = {}
    for (p, v, s), r in zip(jobs, res):
        out.setdefault((p, v), []).append(r)
    for (p, v), rs in out.items():
        d = os.path.join(ROOT, 'bench', 'variants', p)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, v + '.npy'), np.array(rs))
        fin = np.array(rs)[:, -1]
        print(f'{p:18s} {v:16s} median {np.median(fin):.2e}  q25 {np.quantile(fin, .25):.2e}  q75 {np.quantile(fin, .75):.2e}', flush=True)
