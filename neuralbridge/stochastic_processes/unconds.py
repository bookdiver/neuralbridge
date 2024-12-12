from neuralbridge.setups import *

class ContinuousTimeProcess(abc.ABC):
    T: float
    dim: int
    dtype: jnp.dtype
    
    def __init__(self, T: float, dim: int, dtype: Optional[jnp.dtype] = jnp.float32):
        self.T = T
        self.dim = dim
        self.dtype = dtype 
    
    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray, **kwargs):
        pass
    
    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, **kwargs):
        pass

    def Sigma(self, t: float, x: jnp.ndarray, **kwargs):
        g = self.g(t, x, **kwargs)
        return jnp.einsum('i j, k j -> i k', g, g)
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, **kwargs):
        return jnp.linalg.inv(self.Sigma(t, x, **kwargs))
    
    def div_Sigma(self, t: float, x: jnp.ndarray, **kwargs):
        _jacobian = jnp.stack(
            [jax.jacfwd(self.Sigma, argnums=1)(t, x[i], **kwargs)
             for i in range(self.dim)],
            axis=0
        )
        return jnp.trace(_jacobian)
    
class AuxiliaryProcess(ContinuousTimeProcess):
    T: float
    dim: int
    dtype: jnp.dtype
    
    def __init__(self, T: float, dim: int, dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)

    @abc.abstractmethod
    def beta(self, t: float, **kwargs):
        pass

    @abc.abstractmethod
    def B(self, t: float, **kwargs):
        pass

    def f(self, t: float, x: jnp.ndarray, **kwargs):
        return self.beta(t, **kwargs) + jnp.einsum('i j, j -> i', self.B(t, **kwargs), x)

    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, **kwargs):
        pass 
    
    
    
    