"""Best STRICTLY feasible performance per run (the penalized cost can trade tiny
constraint violations for performance). Re-runs the main airfoil methods with an
objective wrapper that records every evaluated design's metrics."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'bench'))
import run_variants as rv  # noqa: E402
from pboplus.airfoil import get_airfoil_problems  # noqa: E402


class Tracked:
    def __init__(self, p):
        self.p, self.best = p, {0.0: -np.inf, 0.01: -np.inf}

    def __getattr__(self, k):
        return getattr(self.p, k)

    def __call__(self, a):
        i = self.p.info(np.asarray(a))
        for tol in self.best:
            if i['violation'] <= tol + 1e-12 and np.isfinite(i['perf']):
                self.best[tol] = max(self.best[tol], i['perf'])
        return float(i['cost']) if np.isfinite(i['cost']) else 1e4


def job(a):
    pname, vname, seed = a
    P = get_airfoil_problems()
    t = Tracked(P[pname])
    orig = rv.get_problems
    rv.get_problems = lambda: {pname: t}
    try:
        rv.run_one((pname, vname, seed))
    finally:
        rv.get_problems = orig
    return pname, vname, seed, t.best[0.0], t.best[0.01]


if __name__ == '__main__':
    probs = sys.argv[1].split(',')
    variants = sys.argv[2].split(',')
    seeds = int(sys.argv[3])
    out = os.path.join(ROOT, 'results', 'feasible_best.json')
    res = json.load(open(out)) if os.path.exists(out) else []
    done = {(r[0], r[1], r[2]) for r in res}
    jobs = [(p, v, s) for p in probs for v in variants for s in range(seeds) if (p, v, s) not in done]
    with Pool(int(os.environ.get('CPU', '4'))) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(list(r)); print(*r, flush=True)
            json.dump(res, open(out, 'w'))
