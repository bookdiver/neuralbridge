import jax
import jax.numpy as jnp

from ..stochastic_processes.cell_model import CellDiffusionProcess
from ..utils.sample_path import SamplePath
from ..stochastic_processes.abstract_processes.wiener_process import WienerProcess
from ..solvers.euler import Euler

def test_sample_path():
    ts = jnp.linspace(0.0, 1.0, 500)
    xs = jnp.zeros((len(ts), 2))
    path = SamplePath(ts=ts, xs=xs)
    print(path)
    ys = jnp.ones((len(ts), 3))
    path.add("ys", ys)
    print(path)
    assert path.ts.shape == (len(ts),)
    assert path[0]["ys"].shape == (3,)

def test_wiener_process():
    rng_key = jax.random.PRNGKey(0)
    process = WienerProcess(dim=1, rng_key=rng_key, dt=0.01)
    evaluate_ts = jnp.linspace(0.0, 1.0, 500)
    path = process.sample_path()
    print(path.xs[:10])
    assert path.xs.shape == (100, 1)
    path = process.sample_path(evaluate_ts)
    print(path.xs[:10])
    assert path.xs.shape == (499, 1)

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