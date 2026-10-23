"""Airfoil design problems: Kulfan/CST parametrization + NeuralFoil aerodynamics.

Design vector (18 variables, like ShapeBench's NeuralFoil environment, Sharpe 2025):
8 upper CST weights, 8 lower CST weights, leading-edge weight, TE thickness.
Some tasks append the angle of attack as a 19th variable.

Costs are MINIMIZED (negative performance + penalties). Constraints are
handled with normalized violations times a large weight, so that every
feasible design beats every clearly infeasible one. `info(x)` returns the
unpenalized metrics and the feasibility flag for reporting.
"""
import numpy as np
import aerosandbox as asb

from pboplus.benchmarks import Problem

UPPER = (-0.30, 0.60)
LOWER = (-0.30, 0.30)
LE = (-0.50, 0.50)
TE = (0.000, 0.010)
# Baseline (approximate NACA 4412, as ShapeBench's default design)
X0 = np.array([0.17, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04,
               -0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005,
               0.0, 0.0025])
XMIN = np.array([UPPER[0]] * 8 + [LOWER[0]] * 8 + [LE[0], TE[0]])
XMAX = np.array([UPPER[1]] * 8 + [LOWER[1]] * 8 + [LE[1], TE[1]])


def _wiggliness(af):
    return sum(np.sum(np.diff(np.diff(a)) ** 2) for a in (af.lower_weights, af.upper_weights))


W_NACA0012 = _wiggliness(asb.KulfanAirfoil('naca0012'))


def make_airfoil(x):
    return asb.KulfanAirfoil(upper_weights=x[:8], lower_weights=x[8:16],
                             leading_edge_weight=x[16], TE_thickness=x[17])


def geometry_violations(af, x, t033_min=0.128, t090_min=0.014, te_min=6.03):
    """ShapeBench-style geometric constraints (normalized violations)."""
    v = {}
    th = np.asarray(af.local_thickness())
    v['thickness>0'] = max(0.0, -float(np.min(th))) / 0.01
    v['t033'] = max(0.0, (t033_min - float(af.local_thickness(0.33))) / t033_min)
    v['t090'] = max(0.0, (t090_min - float(af.local_thickness(0.90))) / t090_min)
    v['te_angle'] = max(0.0, (te_min - float(af.TE_angle())) / te_min)
    v['uw0'] = max(0.0, (0.05 - x[0]) / 0.05)
    v['lw0'] = max(0.0, (x[8] + 0.05) / 0.05)
    w_lim = 2.0 * W_NACA0012
    v['wiggliness'] = max(0.0, (_wiggliness(af) - w_lim) / w_lim)
    return v


class AirfoilProblem(Problem):
    PENALTY = 200.0

    def __init__(self, name, task, extra_bounds=None, x0_extra=None, n_ind=12, n_gen=84):
        x_min, x_max, x_0 = XMIN, XMAX, X0
        if extra_bounds is not None:
            x_min = np.concatenate([x_min, [b[0] for b in extra_bounds]])
            x_max = np.concatenate([x_max, [b[1] for b in extra_bounds]])
            x_0 = np.concatenate([x_0, x0_extra])
        super().__init__(name, x_min, x_max, x_0, None, n_ind, n_gen, f_opt=0.0)
        self.task = task
        self.f_opt = None   # unknown optimum
        # Kink-free map: one slope per variable, a = 0 is the baseline design.
        # (The piecewise map of the original PBO has a slope discontinuity at a = 0,
        #  which stalls every covariance-adapting method; see README.)
        self.scale = np.maximum(self.x_max - self.x_0, self.x_0 - self.x_min)

    def convert(self, a):
        return np.clip(self.x_0 + self.scale * np.clip(a, -1.0, 1.0), self.x_min, self.x_max)

    def info(self, a):
        x = self.convert(np.asarray(a, float))
        try:
            perf, viol, extra = self.task(x)
        except Exception:      # geometry/solver failure
            return dict(perf=np.nan, violation=np.inf, feasible=False, cost=1e4)
        tv = float(sum(viol.values()))
        return dict(perf=perf, violation=tv, feasible=tv <= 1e-9, cost=-perf + self.PENALTY * tv, **extra)

    def __call__(self, a):
        c = self.info(a)['cost']
        return float(c) if np.isfinite(c) else 1e4


def _aero(af, alpha, Re, mach):
    a = af.get_aero_from_neuralfoil(alpha=np.atleast_1d(np.asarray(alpha, float)), Re=Re, mach=mach,
                                    model_size='large')
    return {k: np.atleast_1d(np.asarray(a[k], float)) for k in ('CL', 'CD', 'CM', 'analysis_confidence')}


def task_ld_m02_re1e7(x):
    """ShapeBench 'ld_ratio_constrained_m02_re1e7': max L/D at alpha=5, Re=1e7, M=0.2."""
    af = make_airfoil(x)
    v = geometry_violations(af, x)
    a = _aero(af, 5.0, 1e7, 0.2)
    v['CM'] = max(0.0, (-0.133 - a['CM'][0]) / 0.133)
    v['confidence'] = max(0.0, (0.90 - a['analysis_confidence'][0]) / 0.90)
    cd = max(a['CD'][0], 1e-5)
    return a['CL'][0] / cd, v, dict(CL=a['CL'][0], CD=cd)


def task_endurance_lowre(x):
    """Low-Re glider: max mean CL^1.5/CD over alpha in {0,3,6,9}, Re=3e5, M=0.05.
    Laminar separation bubbles make this landscape strongly non-linear."""
    af = make_airfoil(x)
    v = geometry_violations(af, x, t033_min=0.10, t090_min=0.010)
    al = np.array([0.0, 3.0, 6.0, 9.0])
    a = _aero(af, al, 3e5, 0.05)
    v['confidence'] = float(np.sum(np.maximum(0.0, (0.90 - a['analysis_confidence']) / 0.90)))
    v['CM'] = float(np.sum(np.maximum(0.0, (-0.15 - a['CM']) / 0.15)))
    cd = np.maximum(a['CD'], 1e-5)
    pe = np.sign(a['CL']) * np.abs(a['CL']) ** 1.5 / cd
    return float(np.mean(pe)), v, dict(CL=float(np.mean(a['CL'])), CD=float(np.mean(cd)))


def task_mincd_cl(x):
    """Min drag at prescribed lift CL=0.9 (alpha is a design variable), Re=2e6, M=0.3.
    The equality constraint makes the feasible set a thin curved manifold."""
    af = make_airfoil(x[:18])
    v = geometry_violations(af, x[:18], t033_min=0.12)
    a = _aero(af, x[18], 2e6, 0.3)
    v['CL=0.9'] = max(0.0, abs(a['CL'][0] - 0.9) - 0.01) / 0.9 * 10.0
    v['CM'] = max(0.0, (-0.10 - a['CM'][0]) / 0.10)
    v['confidence'] = max(0.0, (0.90 - a['analysis_confidence'][0]) / 0.90)
    cd = max(a['CD'][0], 1e-5)
    return -1e4 * cd, v, dict(CL=a['CL'][0], CD=cd)   # perf = -drag counts


def get_airfoil_problems():
    return {
        'af_ld_m02_re1e7': AirfoilProblem('af_ld_m02_re1e7', task_ld_m02_re1e7),
        'af_endurance_re3e5': AirfoilProblem('af_endurance_re3e5', task_endurance_lowre),
        'af_mincd_cl09': AirfoilProblem('af_mincd_cl09', task_mincd_cl, extra_bounds=[(-2.0, 10.0)], x0_extra=[2.0]),
    }
