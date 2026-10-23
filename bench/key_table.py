"""Compact table: evaluations to reach 1e-8 (median over successful runs) and success rate."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import load_suite, evals_to_target, ROOT  # noqa: E402
from pboplus.benchmarks import get_problems, get_holdout_problems  # noqa: E402

COLS = [('pbo', 'PBO orig.'), ('pbo+', '**PBO+**'), ('cma-es', 'CMA-ES'), ('lq-cma-es', 'lq-CMA-ES')]


def cell(c, t):
    if c is None:
        return 'n/a'
    e = evals_to_target(c, t)
    ok = e[e > 0]
    if not len(ok):
        return f'– (0/{len(e)}), med f={np.median(c[:, -1]):.1g}'
    return f'{int(np.median(ok))} ({len(ok)}/{len(e)})'


out = []
for title, P in (('Funzioni del paper', get_problems()), ('Funzioni di controllo (mai usate per la taratura)', get_holdout_problems())):
    data = load_suite(P)
    out += [f'**{title}**: valutazioni per raggiungere f − f* ≤ 1e-8 (mediana sui run riusciti; tra parentesi run riusciti/totali)', '',
            '| problema | d | budget | ' + ' | '.join(c[1] for c in COLS) + ' |', '|---|---|---|' + '---|' * len(COLS)]
    for pname, p in P.items():
        out.append(f'| {pname} | {p.dim} | {p.budget} | ' + ' | '.join(cell(data[pname].get(v), 1e-8) for v, _ in COLS) + ' |')
    out.append('')
open(os.path.join(ROOT, 'results', 'key_table.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
