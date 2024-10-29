from typing import Callable
import jax.numpy as jnp

from .bases import (
    ContinuousTimeProcess,
    AuxiliaryProcess
)
class BrownianProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float = 0.0,
                 sigma: float = 1.0,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma

    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.gamma * jnp.ones(self.dim, dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class BrownianAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 gamma: float = 0.0,
                 sigma: float = 1.0,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        
    def beta(self, t: jnp.ndarray):
        return self.gamma * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)

class BrownianBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 score_fn: Callable[[float, jnp.ndarray], jnp.ndarray] | None = None,
                 v: jnp.ndarray | None = None,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        self.v = v
        self.score_fn = score_fn
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        if self.score_fn is not None:
            return self.gamma * jnp.ones(self.dim, dtype=self.dtype) + self.Sigma(t, x) @ self.score_fn(t, x)
        else:
            assert self.v is not None, "v must be provided"
            return self.gamma * jnp.ones(self.dim, dtype=self.dtype) + (self.v - x - self.gamma * (self.T - t)) / (self.T - t + 1e-10)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OUProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        return - self.gamma * x
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OUAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 sigma: float,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.sigma = sigma
        
    def beta(self, t: jnp.ndarray):
        return jnp.zeros(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    
class OUBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 xT: jnp.ndarray,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        self.xT = xT
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        return - self.gamma * x + \
            4.0 * self.gamma**2 * jnp.exp(- self.gamma * (self.T - t)) * \
                (x * jnp.exp(- self.gamma * (self.T - t)) - self.xT) / (jnp.exp(-2.0 * self.gamma * (self.T - t)) - 1.0)
                
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    

class GeometricBrownianMotion(ContinuousTimeProcess):

    def __init__(self,
                 mu: float,
                 sigma: float,
                 *,
                 T: float = 1.0,
                 dim: int = 1,
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.mu = mu
        self.sigma = sigma

    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.mu * x

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * x
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * x**2
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * x**-2
    

class CellDiffusionProcess(ContinuousTimeProcess):

    def __init__(self, 
                 alpha: float, 
                 sigma: float,
                 *,
                 T: float = 1.0,
                 dim: int = 2, 
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.alpha = alpha
        self.sigma = sigma

    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        def u(x, alpha):
            return x**4 / (alpha + x**4)
        
        return jnp.array([u(x[0], self.alpha) + 1.0 - u(x[1], self.alpha) - x[0],
                            u(x[1], self.alpha) + 1.0 - u(x[0], self.alpha) - x[1]],
                            dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class CellDiffusionAuxProcess(AuxiliaryProcess):

    def __init__(self, 
                 alpha: float, 
                 sigma: float, 
                 *, 
                 T: float = 1.0,
                 dim: int = 2, 
                 dtype: jnp.dtype = jnp.float32):
        super().__init__(T, dim, dtype)
        self.alpha = alpha
        self.sigma = sigma

    def beta(self, t: jnp.ndarray):
        return jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray):
        return - 1.0 * jnp.eye(self.dim, dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)

    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)

