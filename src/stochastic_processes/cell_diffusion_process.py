import jax.numpy as jnp

from .continuous_time_process import ContinuousTimeProcess
from .auxiliary_process import AuxiliaryProcess

class CellDiffusionProcess(ContinuousTimeProcess):

    def __init__(self, dim: int, dtype: jnp.ndarray, alpha: float, sigma: float):
        self.dim = dim
        self.dtype = dtype
        self.alpha = alpha
        self.sigma = sigma

    def f(self, t: float | jnp.ndarray, x: jnp.ndarray):
        u = lambda x, alpha: x**4 / (alpha + x**4)
        return jnp.ndarray([u(x[0], self.alpha) + 1.0 - u(x[1], self.alpha) - x[0],
                            u(x[1], self.alpha) + 1.0 - u(x[0], self.alpha) - x[1]],
                            dtype=self.dtype)
    
    def g(self, t: float | jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    

class CellDiffusionProcessAux(AuxiliaryProcess):

    def __init__(self, dim: int, dtype: jnp.ndarray, alpha: float, sigma: float):
        self.dim = dim
        self.dtype = dtype
        self.alpha = alpha
        self.sigma = sigma

    def beta(self, t: float | jnp.ndarray):
        return jnp.ones(self.dim, dtype=self.dtype)
    
    def B(self, t: float | jnp.ndarray):
        return -1.0 * jnp.eye(self.dim, dtype=self.dtype)
    
    def g(self, t: float | jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)

