import abc

import jax.numpy as jnp

from .continuous_time_process import ContinuousTimeProcess

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
