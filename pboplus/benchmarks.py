"""Benchmark functions of the original PBO paper/repository (jviquerat/pbo, envs/).

Each problem works in the original *action* space a in [-1,1]^d: the action is
mapped to the physical variable x with the same piecewise-linear map around the
starting point x_0 used by the original `base_env.convert_actions`, so a=0 is
the original starting point. All problems return a cost to MINIMIZE (the
original returns reward = -cost).
"""
import numpy as np
from math import cos, sqrt, pi


class Problem:
    def __init__(self, name, x_min, x_max, x_0, f, n_ind, n_gen, f_opt=0.0):
        self.name, self.f, self.f_opt = name, f, f_opt
        self.x_min, self.x_max, self.x_0 = map(np.asarray, (x_min, x_max, x_0))
        self.dim = len(self.x_0)
        # Population size and generations of the original json files
        self.n_ind, self.n_gen = n_ind, n_gen
        self.budget = n_ind * n_gen

    def convert(self, a):
        a = np.clip(a, -1.0, 1.0)
        return np.where(a >= 0.0, self.x_0 + (self.x_max - self.x_0) * a,
                        self.x_0 + (self.x_0 - self.x_min) * a)

    def __call__(self, a):
        return self.f(self.convert(a)) - self.f_opt


def rosen(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))


def branin(x):
    b, c, t = 5.1 / (4.0 * pi ** 2), 5.0 / pi, 1.0 / (8.0 * pi)
    return (x[1] - b * x[0] ** 2 + c * x[0] - 6.0) ** 2 + 10.0 * (1.0 - t) * cos(x[0]) + 10.0


class ParabolaTriangle(Problem):
    """Parabola on a triangle: the second component's range depends on the first."""

    def __init__(self):
        super().__init__('parabola_triangle', [0, 0], [1, 1], [0.2, 0.2],
                         lambda x: (x[0] - 0.1) ** 2 + (x[1] - 0.8) ** 2, 6, 50)

    def convert(self, a):
        a = np.clip(a, -1.0, 1.0)
        x0 = self.x_0
        c0 = x0[0] + ((1.0 - x0[0]) if a[0] >= 0 else x0[0]) * a[0]
        c1 = x0[1] + ((1.0 - c0 - x0[1]) if a[1] >= 0 else x0[1]) * a[1]
        return np.array([c0, c1])


def get_problems():
    r5_0 = [0.0] + [-1.0] * 4
    r10_0 = [0.0] + [-1.0] * 9
    return {
        'sphere_2d': Problem('sphere_2d', [-5, -5], [5, 5], [2.5, 2.5], lambda x: float(np.sum(x ** 2)), 6, 50),
        'branin_2d': Problem('branin_2d', [0, 0], [15, 15], [7.5, 7.5], branin, 6, 50, f_opt=5.0 / (4.0 * pi)),
        'griewank_2d': Problem('griewank_2d', [-10, -10], [10, 10], [5, 5],
                               lambda x: 1.0 + (x[0] ** 2 + x[1] ** 2) / 4000.0 - cos(x[0]) * cos(x[1] / sqrt(2.0)), 6, 50),
        'himmelblau': Problem('himmelblau', [-4, -4], [4, 4], [0, 0],
                              lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2, 10, 21),
        'parabola_triangle': ParabolaTriangle(),
        'rosenbrock_2d': Problem('rosenbrock_2d', [-2, -2], [2, 2], [0, -1], rosen, 6, 150),
        'rosenbrock_5d': Problem('rosenbrock_5d', [-2] * 5, [2] * 5, r5_0, rosen, 8, 300),
        'rosenbrock_10d': Problem('rosenbrock_10d', [-2] * 10, [2] * 10, r10_0, rosen, 10, 600),
    }


def get_holdout_problems():
    """Functions NOT used while designing/tuning PBO+ (generalization check).

    Same conventions: actions in [-1,1]^d, start at the box centre offset,
    budget = n_ind * n_gen, minimum value 0.
    """
    rng = np.random.default_rng(12345)

    def rot(d):
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return q
    R5, R10 = rot(5), rot(10)

    def ellipsoid(R):
        d = R.shape[0]
        c = 1e6 ** (np.arange(d) / (d - 1))
        return lambda x: float(np.sum(c * (R @ (x - 0.3)) ** 2))

    def rastrigin(x):
        return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * pi * x)))

    def ackley(x):
        d = len(x)
        return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d)) - np.exp(np.sum(np.cos(2 * pi * x)) / d) + 20 + np.e)

    def levy(x):
        w = 1 + (x - 1) / 4
        return float(np.sin(pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(pi * w[:-1] + 1) ** 2))
                     + (w[-1] - 1) ** 2 * (1 + np.sin(2 * pi * w[-1]) ** 2))

    def styblinski(x):
        return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x) + 39.16616570377142 * len(x))

    def diffpow(x):
        return float(np.sum(np.abs(x) ** (2 + 4 * np.arange(len(x)) / (len(x) - 1))))

    return {
        'rot_ellipsoid_5d': Problem('rot_ellipsoid_5d', [-2] * 5, [2] * 5, [1.0] * 5, ellipsoid(R5), 8, 150),
        'rot_ellipsoid_10d': Problem('rot_ellipsoid_10d', [-2] * 10, [2] * 10, [1.0] * 10, ellipsoid(R10), 10, 300),
        'diffpow_8d': Problem('diffpow_8d', [-2] * 8, [2] * 8, [1.0] * 8, diffpow, 10, 150),
        'levy_5d': Problem('levy_5d', [-10] * 5, [10] * 5, [5.0] * 5, levy, 8, 150),
        'styblinski_4d': Problem('styblinski_4d', [-5] * 4, [5] * 4, [0.5] * 4, styblinski, 8, 100),
        'ackley_5d': Problem('ackley_5d', [-5] * 5, [5] * 5, [2.0] * 5, ackley, 8, 150),
        'rastrigin_3d': Problem('rastrigin_3d', [-5.12] * 3, [5.12] * 3, [2.0] * 3, rastrigin, 6, 150),
        'rosenbrock_20d': Problem('rosenbrock_20d', [-2] * 20, [2] * 20, [0.0] + [-1.0] * 19, rosen, 12, 1000),
    }


def get_dev_multimodal():
    """Multimodal DEVELOPMENT set, used to design the restart strategies
    (kept disjoint from the hold-out set)."""
    def schwefel(x):
        return float(418.9828872724338 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def eggholder(x):
        a, b = x
        return float(-(b + 47) * np.sin(np.sqrt(abs(a / 2 + b + 47))) - a * np.sin(np.sqrt(abs(a - b - 47)))) + 959.6406627208506

    def michalewicz(x):
        i = np.arange(1, len(x) + 1)
        return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / pi) ** 20)) + 4.687658179

    return {
        'schwefel_3d': Problem('schwefel_3d', [-500] * 3, [500] * 3, [100.0] * 3, schwefel, 8, 150),
        'eggholder_2d': Problem('eggholder_2d', [-512] * 2, [512] * 2, [0.0] * 2, eggholder, 6, 150),
        'michalewicz_5d': Problem('michalewicz_5d', [0] * 5, [pi] * 5, [1.5] * 5, michalewicz, 8, 150),
    }
