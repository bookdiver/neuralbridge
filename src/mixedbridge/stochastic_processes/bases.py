import abc
from typing import Callable

import jax
import jax.numpy as jnp

class ContinuousTimeProcess(abc.ABC):

    dim: int
    dtype: jnp.dtype
    T: float
    dt: float
    
    def __init__(self, T: float, dt: float, dim: int, dtype: jnp.dtype):
        self.T = T
        self.dt = dt
        self.dim = dim
        self.dtype = dtype

    @property
    def ts(self):
        assert self.dt is not None, "Time step size must be provided!"
        return jnp.arange(0, self.T + self.dt, self.dt, dtype=self.dtype)
    
    @property
    def reverse_ts(self):
        return jnp.flip(self.ts) 
    
    @property
    def dts(self):
        return jnp.diff(self.ts) 
    
    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass

    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.g(t, x, *args, **kwargs) @ self.g(t, x, *args, **kwargs).T
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return jnp.linalg.inv(self.Sigma(t, x, *args, **kwargs))
    
    def div_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        _jacobian = jnp.stack(
            [jax.jacfwd(self.Sigma, argnums=1)(t, x[i], *args, **kwargs)
             for i in range(self.dim)],
            axis=0
        )
        return jnp.trace(_jacobian)
    
class AuxiliaryProcess(ContinuousTimeProcess):

    dim: int
    dtype: jnp.dtype
    T: float
    dt: float
    
    def __init__(self, T: float, dt: float, dim: int, dtype: jnp.dtype):
        super().__init__(T, dt, dim, dtype)

    @abc.abstractmethod
    def beta(self, t: float, *args, **kwargs):
        pass

    @abc.abstractmethod
    def B(self, t: float, *args, **kwargs):
        pass

    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.beta(t, *args, **kwargs) + self.B(t, *args, **kwargs) @ x

    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass 

class BridgeProcess(ContinuousTimeProcess):
    dim: int
    dtype: jnp.dtype
    T: float
    dt: float
    _f: Callable[[float, jnp.ndarray], jnp.ndarray]
    _g: Callable[[float, jnp.ndarray], jnp.ndarray]
    score_fn: Callable[[float, jnp.ndarray], jnp.ndarray]
    
    def __init__(self, 
                 T: float, 
                 dt: float, 
                 dim: int, 
                 dtype: jnp.dtype, 
                 f: Callable[[float, jnp.ndarray], jnp.ndarray],
                 g: Callable[[float, jnp.ndarray], jnp.ndarray],
                 score_fn: Callable[[float, jnp.ndarray], jnp.ndarray] | None = None):
        super().__init__(T, dt, dim, dtype)
        self._f = f
        self._g = g
        self._score_fn = score_fn
        
    def score(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self._score_fn(t, x)
    
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self._f(t, x, *args, **kwargs) + self.Sigma(t, x, *args, **kwargs) @ self.score(t, x, *args, **kwargs)
    
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self._g(t, x, *args, **kwargs)
    
    
    
    