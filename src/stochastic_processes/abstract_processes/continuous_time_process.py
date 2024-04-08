import abc
from functools import partial

import jax
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
    @partial(jax.jit, static_argnums=(0,))
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass