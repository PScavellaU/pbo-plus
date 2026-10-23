"""How multimodal is the airfoil L/D landscape? Run PBO+ (cheap split) from several
seeds, collect the best designs and measure how far apart they are (in design space
and in shape), together with their performance."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pboplus.core import PBOPlus  # noqa: E402
from pboplus.airfoil import get_airfoil_problems, make_airfoil  # noqa: E402

PNAME = sys.argv[1] if len(sys.argv) > 1 else 'af_ld_m02_re1e7'


def job(seed):
    p = get_airfoil_problems()[PNAME]
    o = PBOPlus(p.dim, lam=p.n_ind, sigma0=0.25, seed=seed, cheap=p.cheap)
    a, f = o.optimize(p, p.budget)
    return seed, a.tolist(), float(f)


if __name__ == '__main__':
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    with Pool(int(os.environ.get('CPU', '1'))) as pool:
        res = pool.map(job, seeds)
    p = get_airfoil_problems()[PNAME]
    A = np.array([r[1] for r in res]); F = np.array([r[2] for r in res])
    X = np.array([p.convert(a) for a in A])
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    shapes = []
    for x in X:
        c = np.asarray(make_airfoil(x[:18]).upper_coordinates()), np.asarray(make_airfoil(x[:18]).lower_coordinates())
        shapes.append(c)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    order = np.argsort(F)
    cmap = plt.get_cmap('viridis')
    for k, i in enumerate(order):
        af = make_airfoil(X[i][:18]); co = np.asarray(af.coordinates)
        ax[0].plot(co[:, 0], co[:, 1], color=cmap(k / max(1, len(F) - 1)), lw=1.2, label=f'seed {res[i][0]}: L/D={-F[i]:.0f}')
    ax[0].set_aspect('equal'); ax[0].legend(fontsize=7, ncol=2); ax[0].set_title(f'{PNAME}: best design of each run'); ax[0].grid(alpha=.3)
    im = ax[1].imshow(D[np.ix_(order, order)], cmap='magma')
    ax[1].set_title('distance between best designs (design space)'); plt.colorbar(im, ax=ax[1])
    ax[1].set_xticks(range(len(F))); ax[1].set_xticklabels([f'{-F[i]:.0f}' for i in order], rotation=90, fontsize=7)
    ax[1].set_yticks(range(len(F))); ax[1].set_yticklabels([f'{-F[i]:.0f}' for i in order], fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, 'results', f'multimodality_{PNAME}.png'), dpi=110)
    json.dump({'seeds': [r[0] for r in res], 'cost': F.tolist(), 'designs': X.tolist(),
               'mean_pairwise_distance': float(D[np.triu_indices(len(F), 1)].mean())},
              open(os.path.join(ROOT, 'results', f'multimodality_{PNAME}.json'), 'w'), indent=1)
    print('L/D:', np.round(-F, 1)); print('mean pairwise design distance', D[np.triu_indices(len(F), 1)].mean().round(3))
    print('max |x| range per var', np.round(X.max(0) - X.min(0), 3))
