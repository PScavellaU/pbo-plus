"""Compare original PBO with PBO+ variants: convergence plots and a summary table."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per process: avoids oversubscription
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pboplus.benchmarks import get_problems, get_holdout_problems  # noqa: E402
from pboplus.airfoil import get_airfoil_problems  # noqa: E402

FLOOR = 1e-20
TARGETS = (1e-3, 1e-8)
SHOW = ['pbo', 'pbo+', 'pbo+_gp', 'pbo+_noinject', 'pbo+_nocritic', 'xnes', 'cma-es', 'lq-cma-es', 'turbo']
COLORS = {'pbo': 'k', 'pbo+': 'tab:red', 'pbo+_noinject': 'tab:pink', 'pbo+_noincremental': 'tab:brown',
          'pbo+_nocritic': 'tab:orange', 'xnes': 'tab:green', 'xnes_reuse': 'tab:olive',
          'cma-es': 'tab:blue', 'lq-cma-es': 'tab:purple', 'pbo+_gp': 'darkred', 'turbo': 'tab:gray'}
LABELS = {'pbo': 'PBO (original code)', 'pbo+': 'PBO+ (full)', 'pbo+_noinject': 'PBO+ w/o actor step',
          'pbo+_noincremental': 'PBO+ w/o incremental critic', 'pbo+_nocritic': 'PBO+ w/o critic',
          'xnes': 'natural-gradient PBO (xNES)', 'xnes_reuse': 'xNES + IS reuse',
          'cma-es': 'CMA-ES (pycma)', 'lq-cma-es': 'lq-CMA-ES (pycma)',
          'pbo+_gp': 'PBO+ (GP critic)', 'turbo': 'TuRBO-1 (BO)'}
ORDER = ['pbo', 'pbo+', 'pbo+_gp', 'pbo+_noinject', 'pbo+_noincremental', 'pbo+_nocritic', 'xnes', 'xnes_reuse', 'cma-es', 'lq-cma-es', 'turbo']


def load_original(pname, budget):
    """Best-so-far curves from the original code's per-evaluation logs (pbo_<run>)."""
    runs = sorted(glob.glob(os.path.join(ROOT, 'bench', 'original', pname, 'results', '*', 'pbo_[0-9]*')))
    curves = []
    for r in runs:
        c = np.loadtxt(r)[:, 2]                     # cost = -reward, in evaluation order
        if len(c) < budget or np.all(c[-1:] == 0):
            continue
        if pname == 'branin_2d':                    # original subtracts 0.397887 (rounded)
            c = c + 0.397887 - 5.0 / (4.0 * np.pi)
        curves.append(np.minimum.accumulate(c[:budget]))
    return np.array(curves) if curves else None


def evals_to_target(curves, t):
    hit = curves <= t
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, -1)


def load_suite(P):
    data = {}
    for pname, p in P.items():
        d = {}
        o = load_original(pname, p.budget)
        if o is not None:
            d['pbo'] = o
        for f in glob.glob(os.path.join(ROOT, 'bench', 'variants', pname, '*.npy')):
            v = os.path.basename(f)[:-4]
            if v in COLORS:
                d[v] = np.load(f)
        data[pname] = d
    return data


AF_YLABEL = {'af_ld_m02_re1e7': 'best L/D (penalized)', 'af_endurance_re3e5': 'best mean CL^1.5/CD (penalized)',
             'af_mincd_cl09': '−best CD [counts] (penalized)'}


def plot_curves(P, data, fname, title, airfoil=False):
    names = list(P)
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.8 * nr), squeeze=False)
    for ax in axs.flat[len(names):]:
        ax.axis('off')
    for ax, pname in zip(axs.flat, names):
        for v in SHOW:
            if v not in data[pname]:
                continue
            c = -data[pname][v] if airfoil else np.maximum(data[pname][v], FLOOR)
            x = np.arange(1, c.shape[1] + 1)
            med, q1, q3 = np.median(c, 0), np.quantile(c, .25, 0), np.quantile(c, .75, 0)
            ax.plot(x, med, color=COLORS[v], lw=2.4 if v in ('pbo', 'pbo+') else 1.2,
                    ls='--' if v in ('cma-es', 'lq-cma-es') else '-', label=f'{LABELS[v]} (n={len(c)})')
            ax.fill_between(x, q1, q3, color=COLORS[v], alpha=0.12)
        if airfoil:
            lo = np.nanmin([np.median(-data[pname][v][:, c.shape[1] // 4]) for v in data[pname]])
            hi = np.nanmax([np.quantile(-data[pname][v][:, -1], .75) for v in data[pname]])
            ax.set_ylim(lo, hi + 0.05 * abs(hi - lo))
        else:
            ax.set_yscale('log')
        ax.set_title(f'{pname} (d={P[pname].dim}, λ={P[pname].n_ind})', fontsize=10)
        ax.set_xlabel('function evaluations', fontsize=9)
        ax.set_ylabel(AF_YLABEL.get(pname, 'best f − f*') + '  (median, IQR)', fontsize=9)
        ax.grid(alpha=.3)
    hl = {}
    for ax in axs.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l.split(' (n=')[0], h)
    if hl:
        fig.legend(list(hl.values()), list(hl), loc='lower center', ncol=min(len(hl), 5), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(ROOT, 'results', fname), dpi=110)
    plt.close(fig)


def plot_ecdf(P, data, fname, title, targets=np.logspace(2, -8, 21)):
    """COCO-style runtime ECDF: fraction of (problem, run, target) pairs solved vs evals/budget."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(0, 1, 201)
    for v in SHOW:
        frac = np.zeros_like(grid)
        n = 0
        for pname in P:
            if v not in data[pname]:
                continue
            c = data[pname][v]
            B = c.shape[1]
            for run in c:
                for t in targets:
                    hit = np.nonzero(run <= t)[0]
                    if len(hit):
                        frac += grid >= (hit[0] + 1) / B
                    n += 1
        if n:
            ax.plot(grid, frac / n, color=COLORS[v], lw=2.4 if v in ('pbo', 'pbo+') else 1.3,
                    ls='--' if v in ('cma-es', 'lq-cma-es') else '-', label=LABELS[v])
    ax.set_xlabel('fraction of the evaluation budget of the original PBO setup')
    ax.set_ylabel('fraction of (problem, run, target) solved')
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper left'); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'results', fname), dpi=120)
    plt.close(fig)


def table(P, data):
    lines = ['| problem | algorithm | runs | median final | q25 | q75 | evals to 1e-3 (median, solved) | evals to 1e-8 (median, solved) |',
             '|---|---|---|---|---|---|---|---|']
    for pname in P:
        for v in ORDER:
            if v not in data[pname]:
                continue
            c = data[pname][v]
            fin = c[:, -1]
            cells = []
            for t in TARGETS:
                e = evals_to_target(c, t)
                ok = e[e > 0]
                cells.append(f'{int(np.median(ok)) if len(ok) else "–"} ({len(ok)}/{len(e)})')
            name = f'**{LABELS[v]}**' if v in ('pbo', 'pbo+') else LABELS[v]
            lines.append(f'| {pname} | {name} | {len(c)} | {np.median(fin):.1e} | {np.quantile(fin, .25):.1e} | {np.quantile(fin, .75):.1e} | {cells[0]} | {cells[1]} |')
    return lines


def main():
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out = []
    for suite, P, title in (('paper', get_problems(), 'Benchmarks of the PBO paper (same budgets and λ as the original json files)'),
                            ('holdout', get_holdout_problems(), 'Hold-out functions (never used to design or tune PBO+)')):
        data = load_suite(P)
        if not any(data.values()):
            continue
        plot_curves(P, data, f'convergence_{suite}.png', title)
        plot_ecdf(P, data, f'ecdf_{suite}.png', 'Runtime ECDF, targets 1e+2 … 1e-8 — ' + suite)
        out += [f'### {title}', ''] + table(P, data) + ['']
    P = get_airfoil_problems()
    data = load_suite(P)
    if any(data.values()):
        plot_curves(P, data, 'convergence_airfoil.png', 'Airfoil design (CST + NeuralFoil), 18–19 variables', airfoil=True)
        out += ['### Airfoil design (performance = −penalized cost; higher is better)', '',
                '| problem | algorithm | runs | median final | q25 | q75 | best run | worst run | median at 50% budget |',
                '|---|---|---|---|---|---|---|---|---|']
        for pname in P:
            for v in ORDER:
                if v not in data[pname]:
                    continue
                c = -data[pname][v]
                f, h = c[:, -1], c[:, c.shape[1] // 2]
                name = f'**{LABELS[v]}**' if v in ('pbo', 'pbo+') else LABELS[v]
                out.append(f'| {pname} | {name} | {len(c)} | {np.median(f):.2f} | {np.quantile(f, .25):.2f} | {np.quantile(f, .75):.2f} | {f.max():.2f} | {f.min():.2f} | {np.median(h):.2f} |')
        out.append('')
    open(os.path.join(ROOT, 'results', 'summary.md'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
