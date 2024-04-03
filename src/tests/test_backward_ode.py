import jax.numpy as jnp

from ..stochastic_processes.cell_diffusion_process import CellDiffusionProcess, CellDiffusionProcessAux
from ..backward_filtering.backward_ode import BackwardODE

def test_backward_ode():
    dim = 2
    alpha = 1.0 / 16.0
    sigma = 0.1
    L0 = jnp.array([[1.0, 0.0],
                    [0.0, 1.0]])
    v = jnp.array([2.0, -0.1])
    m, d = L0.shape
    Sigma = 1e-10 * jnp.eye(m, dtype=jnp.float32)
    ts = jnp.linspace(0.0, 4.0, 500)

    p = CellDiffusionProcess(dim, jnp.float32, alpha, sigma)
    pt = CellDiffusionProcessAux(dim, jnp.float32, alpha, sigma)
    ode = BackwardODE(p, pt, L0, v, Sigma)

    Ls, Ms, mus = ode.solve(ts[::-1])
    assert Ls.shape == (len(ts), m, d)
    assert Ms.shape == (len(ts), m, m)
    assert mus.shape == (len(ts), m)