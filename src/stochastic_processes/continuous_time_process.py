import abc

import jax.numpy as jnp

class ContinuousTimeProcess(abc.ABC):
    dim: int
    dtype: jnp.dtype

    @abc.abstractmethod
    def f(self, t: float | jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def g(self, t: float | jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        pass