"""Template: PBO+ driving SU2 airfoil simulations in parallel batches.

Adapt the three commands marked ADAPT (mesh generation, solver launch,
output parsing) to your own chain (e.g. c2dx/Construct2D O-grid + SU2 on a
SLURM cluster). Each block of designs proposed by PBO+ is meshed and run
simultaneously; the critic decides how many designs a block contains
(at least `min_batch`, e.g. the number of jobs you can run at once).
"""
import os
import subprocess
import numpy as np

from pboplus.core import PBOPlus
from pboplus.airfoil import XMIN, XMAX, X0, make_airfoil, geometry_violations

WORK = os.path.abspath('runs')
N_PARALLEL = 12                    # simultaneous CFD jobs
PENALTY = 200.0

# ADAPT: commands (placeholders)
MESH_CMD = 'python run_construct2d.py --coords {coords} --out {case}/mesh.su2'
SOLVE_CMD = 'sbatch --wait --chdir {case} su2_job.sh'          # runs SU2_CFD config.cfg in {case}
HISTORY = 'history.csv'                                        # SU2 output with CL, CD columns

scale = np.maximum(XMAX - X0, X0 - XMIN)


def to_design(a):
    return np.clip(X0 + scale * np.clip(a, -1, 1), XMIN, XMAX)


def read_coefficients(case):
    """ADAPT: return (CL, CD) averaged over the last iterations of SU2 history."""
    import csv
    with open(os.path.join(case, HISTORY)) as fh:
        rows = list(csv.DictReader(fh, skipinitialspace=True))
    key = {k.strip().strip('"'): k for k in rows[0]}
    last = rows[-200:]
    cl = np.mean([float(r[key['CL']]) for r in last])
    cd = np.mean([float(r[key['CD']]) for r in last])
    return cl, cd


counter = [0]


def batch_cost(A):
    """Cost of a block of actions: geometry checks locally, CFD jobs in parallel."""
    cases, viols, procs = [], [], []
    for a in A:
        x = to_design(a)
        af = make_airfoil(x)
        v = sum(geometry_violations(af, x).values())
        case = os.path.join(WORK, f'case_{counter[0]:05d}')
        counter[0] += 1
        os.makedirs(case, exist_ok=True)
        np.savetxt(os.path.join(case, 'design.txt'), x)
        coords = os.path.join(case, 'airfoil.dat')
        np.savetxt(coords, np.asarray(af.coordinates))
        cases.append(case), viols.append(v)
        if v > 1.0:               # clearly infeasible geometry: do not spend a CFD run
            procs.append(None)
            continue
        cmd = MESH_CMD.format(coords=coords, case=case) + ' && ' + SOLVE_CMD.format(case=case)
        procs.append(subprocess.Popen(cmd, shell=True))
    costs = []
    for case, v, p in zip(cases, viols, procs):
        if p is None:
            costs.append(1e3 + PENALTY * v)
            continue
        ok = p.wait() == 0
        try:
            cl, cd = read_coefficients(case) if ok else (np.nan, np.nan)
        except (OSError, KeyError, ValueError, IndexError):
            cl, cd = np.nan, np.nan
        if not np.isfinite(cd) or cd <= 0:
            costs.append(1e3)     # diverged run: ranks last, cannot corrupt the rank-based update
        else:
            costs.append(-cl / cd + PENALTY * v)
    return costs


if __name__ == '__main__':
    opt = PBOPlus(dim=18, lam=12, sigma0=0.25, critic_type='gp', min_batch=N_PARALLEL, seed=0)
    a_best, f_best = opt.optimize(batch_cost, budget=600, batch=True)
    print('best L/D', -f_best, 'design', to_design(a_best))
