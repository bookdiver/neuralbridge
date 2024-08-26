from __future__ import annotations
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
        assert self.dt is not None, "Time step size must be provided!"
        return jnp.arange(0, self.T + self.dt, self.dt, dtype=self.dtype)
    
    @property
    def reverse_ts(self):
        return jnp.flip(self.ts) 
    
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
    
class AuxiliaryProcess(ContinuousTimeProcess):
    dim: int
    dtype: jnp.dtype
    T: float
    dt: float

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

    @abc.abstractmethod
    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        pass
    

    

