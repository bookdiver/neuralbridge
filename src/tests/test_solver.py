import jax
import jax.numpy as jnp

from ..stochastic_processes.cell_model import CellDiffusionProcess
from ..solvers.wiener_process import WienerProcess
from ..solvers.euler import Euler

def test_wiener_process():
    dim = 1
    dtype = jnp.float32
    rng_key = jax.random.PRNGKey(0)
    process = WienerProcess(dim, dtype, rng_key)
    ts = jnp.linspace(0.0, 1.0, 500)
    path = process.sample_path(ts)
    print(path[:10])
    assert path.shape == (len(ts)-1, dim)

def test_euler():
    dim = 2
    alpha = 1.0 / 16.0
    sigma = .1
    dtype = jnp.float32
    rng_key = jax.random.PRNGKey(0)
    x0 = jnp.array([1.0, 1.0])

    wiener_process = WienerProcess(dim, dtype, rng_key)
    sde = CellDiffusionProcess(alpha, sigma)
    solver = Euler(sde, wiener_process)
    path = solver.solve(x0)
    print(path[:10])
    assert path.shape == (len(sde.ts)-1, dim)

    solver = Euler(sde)
    dWs = wiener_process.sample_path(sde.ts)
    path = solver.solve(x0, dWs)
    print(path[:10])
    assert path.shape == (len(sde.ts)-1, dim)