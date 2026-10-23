"""Shapes and polars of the best airfoils found by each optimizer (one seed).

Usage: python bench/plot_airfoils.py [seed]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'baselines'))
from pboplus.core import PBOPlus                    # noqa: E402
from pboplus.airfoil import get_airfoil_problems, make_airfoil  # noqa: E402

ALGOS = {'PBO+ (GP critic)': ('pbo+', dict(critic_type='gp')), 'PBO+ (quadratic critic)': ('pbo+', {}),
         'CMA-ES': ('cma', None), 'TuRBO-1': ('turbo', None)}
COL = {'PBO+ (GP critic)': 'darkred', 'PBO+ (quadratic critic)': 'tab:red', 'CMA-ES': 'tab:blue',
       'TuRBO-1': 'tab:gray', 'baseline (≈NACA 4412)': 'k'}


def best_design(p, algo, kw, seed):
    if algo == 'pbo+':
        o = PBOPlus(p.dim, lam=p.n_ind, sigma0=0.25, seed=seed, **kw)
        return o.optimize(p, p.budget)[0]
    if algo == 'turbo':
        from turbo import TuRBO
        o = TuRBO(p.dim, batch=p.n_ind, seed=seed)
        return o.optimize(p, p.budget)[0]
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(p.dim), 0.25, {'popsize': p.n_ind, 'bounds': [-1, 1], 'seed': seed + 1,
                                                          'verbose': -9, 'maxfevals': p.budget})
    es.optimize(p)
    return es.result.xbest


def main(seed=0):
    P = get_airfoil_problems()
    fig, axs = plt.subplots(2, len(P), figsize=(6 * len(P), 7.5))
    report = {}
    for j, (pname, p) in enumerate(P.items()):
        designs = {'baseline (≈NACA 4412)': np.zeros(p.dim)}
        for lab, (algo, kw) in ALGOS.items():
            designs[lab] = best_design(p, algo, kw, seed)
        report[pname] = {}
        for lab, a in designs.items():
            info = p.info(a)
            x = p.convert(a)
            af = make_airfoil(x[:18])
            c = np.asarray(af.coordinates)
            tag = f"{lab}: {info['perf']:.1f}{'' if info['feasible'] else ' (infeasible)'}"
            axs[0, j].plot(c[:, 0], c[:, 1], color=COL[lab], lw=1.6 if lab != 'baseline (≈NACA 4412)' else 1.0,
                           ls='--' if lab.startswith('baseline') else '-', label=tag)
            al = np.linspace(-4, 14, 37)
            re = {'af_ld_m02_re1e7': (1e7, 0.2), 'af_endurance_re3e5': (3e5, 0.05), 'af_mincd_cl09': (2e6, 0.3)}[pname]
            aero = af.get_aero_from_neuralfoil(alpha=al, Re=re[0], mach=re[1], model_size='large')
            axs[1, j].plot(np.asarray(aero['CD']) * 1e4, aero['CL'], color=COL[lab],
                           ls='--' if lab.startswith('baseline') else '-', lw=1.4)
            report[pname][lab] = {k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else None) for k, v in info.items()}
        axs[0, j].set_aspect('equal'); axs[0, j].set_title(pname); axs[0, j].legend(fontsize=7.5, loc='upper right')
        axs[0, j].set_ylim(-0.15, 0.25); axs[0, j].grid(alpha=.3)
        axs[1, j].set_xlabel('CD [counts]'); axs[1, j].set_ylabel('CL'); axs[1, j].set_xlim(0, 300)
        axs[1, j].grid(alpha=.3); axs[1, j].set_title('drag polar (NeuralFoil)')
    fig.suptitle(f'Best airfoils found with the same budget (seed {seed}); legend: performance', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    fig.savefig(os.path.join(ROOT, 'results', 'airfoil_shapes.png'), dpi=110)
    json.dump(report, open(os.path.join(ROOT, 'results', 'airfoil_best_designs.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
