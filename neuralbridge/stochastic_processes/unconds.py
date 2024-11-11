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
    T: float
    dim: int
    dtype: jnp.dtype
    
    def __init__(self, T: float, dim: int, dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)

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
    
    
    
    