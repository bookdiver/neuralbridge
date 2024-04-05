import jax.numpy as jnp

from ..stochastic_processes.cell_model import CellDiffusionProcess, CellDiffusionProcessAux
from ..stochastic_processes.abstract_processes.guided_bridge_process import GuidedBridgeProcess


def test_diffusion_process():
    alpha = 1.0 / 16.0
    sigma = .1
    x = jnp.array([1.0, 1.0])
    t = 0.5
    process = CellDiffusionProcess(alpha, sigma)
    assert process.f(t, x).shape == x.shape
    assert process.g(t, x).shape == (x.shape[0], x.shape[0])
    assert process.Sigma(t, x).shape == (x.shape[0], x.shape[0])

def test_auxiliary_process():
    alpha = 1.0 / 16.0
    sigma = .1
    x = jnp.array([1.0, 1.0])
    t = 0.5
    process = CellDiffusionProcessAux(alpha, sigma)
    assert process.beta(t).shape == x.shape
    assert process.B(t).shape == (x.shape[0], x.shape[0])
    assert process.g(t, x).shape == (x.shape[0], x.shape[0])
    assert process.Sigma(t, x).shape == (x.shape[0], x.shape[0])

def test_guided_bridge_process():
    alpha = 1.0 / 16.0
    sigma = .1
    x = jnp.array([1.0, 1.0])
    t = 0.5
    start = jnp.array([0.0, 0.0])
    target = jnp.array([1.0, 1.0])
    L0 = jnp.eye(2)
    process = GuidedBridgeProcess(CellDiffusionProcess(alpha, sigma), 
                                  CellDiffusionProcessAux(alpha, sigma), 
                                  L0, 
                                  T=1.0, 
                                  dt = 0.01,
                                  start=start, 
                                  target=target)
    assert process.f(t, x).shape == x.shape
    assert process.g(t, x).shape == (x.shape[0], x.shape[0])
    assert process.Sigma(t, x).shape == (x.shape[0], x.shape[0])
