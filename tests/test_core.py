import numpy as np
from pboplus.core import PBOPlus, QuadraticCritic, xnes_utilities
from pboplus.benchmarks import get_problems


def test_utilities_sum_to_zero():
    for lam in (4, 6, 10, 17):
        assert abs(xnes_utilities(lam).sum()) < 1e-12


def test_critic_exact_on_quadratic():
    rng = np.random.default_rng(0)
    d = 4
    H = rng.standard_normal((d, d)); H = H @ H.T
    f = lambda z: 0.5 * np.einsum('ij,jk,ik->i', z, H, z) + z.sum(1) + 3.0
    Z = rng.standard_normal((40, d))
    c = QuadraticCritic(d).fit(Z, f(Z))
    assert c.kind == 'full'
    Zt = rng.standard_normal((10, d))
    assert np.allclose(c.predict(Zt), f(Zt), atol=1e-8)


def test_reflection_stays_in_box():
    x = np.linspace(-9, 9, 1001)
    y = PBOPlus._reflect(x)
    assert y.min() >= -1 and y.max() <= 1


def test_budget_respected_and_solves_rosenbrock():
    p = get_problems()['rosenbrock_5d']
    for critic, tol in ((False, 1e-2), (True, 1e-8)):
        o = PBOPlus(p.dim, lam=p.n_ind, sigma0=0.4, critic=critic, seed=1)
        _, fb = o.optimize(p, p.budget)
        assert o.n_eval == p.budget
        assert fb < tol
