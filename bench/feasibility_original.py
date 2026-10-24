"""Best strictly feasible performance of the original PBO runs, from their logs."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import glob
import json
import numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pboplus.airfoil import get_airfoil_problems  # noqa: E402


def job(path):
    pname = path.split('/original/')[1].split('/')[0]
    p = get_airfoil_problems()[pname]
    L = np.loadtxt(path)
    best = {0.0: -np.inf, 0.01: -np.inf}
    for x in L[:, 3:3 + p.dim]:
        a = (x - p.x_0) / p.scale
        i = p.info(a)
        for t in best:
            if i['violation'] <= t + 1e-12 and np.isfinite(i['perf']):
                best[t] = max(best[t], i['perf'])
    run = int(os.path.basename(path).split('_')[1])
    return [pname, 'pbo', run, best[0.0], best[0.01]]


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(ROOT, 'bench', 'original', 'af_*', 'results', '*', 'pbo_[0-9]*')))
    with Pool(int(os.environ.get('CPU', '1'))) as pool:
        res = pool.map(job, files)
    json.dump(res, open(os.path.join(ROOT, 'results', 'feasible_best_original.json'), 'w'))
    for r in res:
        print(*r)
