import jax.numpy as jnp

from .bases import (
    ContinuousTimeProcess,
    AuxiliaryProcess
)

class CellDiffusionProcess(ContinuousTimeProcess):

    def __init__(self, 
                 alpha: float, 
                 sigma: float,
                 *,
                 T: float=1.0,
                 dt: float=1e-3, 
                 dim: int=2, 
                 dtype: jnp.dtype=jnp.float32):
        self.T = T
        self.dt = dt
        self.dim = dim
        self.dtype = dtype

        self.alpha = alpha
        self.sigma = sigma

    def f(self, t: float, x: jnp.ndarray):
        def u(x, alpha):
            return x**4 / (alpha + x**4)
        
        return jnp.array([u(x[0], self.alpha) + 1.0 - u(x[1], self.alpha) - x[0],
                            u(x[1], self.alpha) + 1.0 - u(x[0], self.alpha) - x[1]],
                            dtype=self.dtype)
    
    def g(self, t: float, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: float, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    

class CellDiffusionProcessAux(AuxiliaryProcess):

    def __init__(self, 
                 alpha: float, 
                 sigma: float, 
                 *, 
                 T: float=1.0,
                 dt: float=1e-3,
                 dim: int=2, 
                 dtype: jnp.dtype=jnp.float32):
        self.T = T
        self.dt = dt
        self.dim = dim
        self.dtype = dtype

        self.alpha = alpha
        self.sigma = sigma

    def beta(self, t: float):
        return jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: float):
        return -1.0 * jnp.eye(self.dim, dtype=self.dtype)

    def g(self, t: float, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)

    def Sigma(self, t: float, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)

