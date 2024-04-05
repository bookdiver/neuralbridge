import abc

import jax.numpy as jnp

class ContinuousTimeProcess(abc.ABC):
    dim: int
    dtype: jnp.dtype
    T: float
    dt: float

    @property
    def ts(self):
        return jnp.arange(0, self.T + self.dt, self.dt)
    
    @property
    def reverse_ts(self):
        return self.ts[::-1]
    
    @property
    def dts(self):
        return jnp.diff(self.ts)
    
    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass

    @abc.abstractmethod
    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass