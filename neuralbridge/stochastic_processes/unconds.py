from neuralbridge.setups import *

class ContinuousTimeProcess(abc.ABC):
    dim: int
    dtype: Optional[jnp.dtype] = DEFAULT_DTYPE
    
    def __init__(self, dim: int):
        self.dim = dim
    
    @abc.abstractmethod
    def f(self, t: tTYPE, x: xTYPE, **kwargs) -> Tuple[xTYPE, xTYPE]:
        pass
    
    @abc.abstractmethod
    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        pass

    def Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
        g = self.g(t, x)
        return jnp.einsum('i j, k j -> i k', g, g)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return jnp.linalg.solve(self.Sigma(t, x), jnp.eye(self.dim))
    
    def div_Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
        _jacobian = jnp.stack(
            [jax.jacfwd(self.Sigma, argnums=1)(t, x[i])
             for i in range(self.dim)],
            axis=0
        )
        return jnp.trace(_jacobian)
    
class AuxiliaryProcess(ContinuousTimeProcess):
    dim: int
    dtype: Optional[jnp.dtype] = DEFAULT_DTYPE
    
    def __init__(self, dim: int):
        super().__init__(dim)

    @abc.abstractmethod
    def beta(self, t: tTYPE) -> xTYPE:
        pass

    @abc.abstractmethod
    def B(self, t: tTYPE) -> xTYPE:
        pass

    def f(self, t: tTYPE, x: xTYPE) -> xTYPE:
        drift = self.beta(t) + jnp.einsum('i j, j -> i', self.B(t), x)
        return drift

    @abc.abstractmethod
    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        pass 
    
    
    
    