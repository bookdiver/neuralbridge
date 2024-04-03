import abc

import jax.numpy as jnp

class AuxiliaryProcess(abc.ABC):
    dim: int
    dtype: jnp.dtype

    @abc.abstractmethod
    def beta(self, t, *args, **kwargs):
        pass

    @abc.abstractmethod
    def B(self, t, *args, **kwargs):
        pass

    def f(self, t, x, *args, **kwargs):
        return self.beta(t, *args, **kwargs) + self.B(t, *args, **kwargs) @ x

    @abc.abstractmethod
    def g(self, t, x, *args, **kwargs):
        pass 

