import abc
from typing import Union

import jax.numpy as jnp

Ttype = Union[jnp.ndarray, float]

class ContinuousTimeProcess(abc.ABC):
    dim: int
    dtype: jnp.dtype

    @abc.abstractmethod
    def f(self, t: Ttype, x: jnp.ndarray, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def g(self, t: Ttype, x: jnp.ndarray, *args, **kwargs):
        pass