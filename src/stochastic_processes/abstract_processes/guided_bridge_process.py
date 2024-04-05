import jax.numpy as jnp

from .continuous_time_process import ContinuousTimeProcess
from .auxiliary_process import AuxiliaryProcess
from ...backward_filtering.backward_ode import BackwardODE

class GuidedBridgeProcess(ContinuousTimeProcess):

    def __init__(self, 
                 original_process: ContinuousTimeProcess, 
                 auxiliary_process: AuxiliaryProcess,
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 *,
                 start: jnp.ndarray,
                 target: jnp.ndarray,
                 ):
        super().__init__()
        assert start.shape == target.shape
        
        self._B = auxiliary_process.B
        self._f = original_process.f
        self._g = original_process.g

        self.T = original_process.T
        self.dt = original_process.dt
        self.dim = original_process.dim
        self.dtype = original_process.dtype

        self.start = start
        self.target = target

        self.backward_ode = BackwardODE(
            original_process, 
            auxiliary_process, 
            L0, 
            Sigma0=Sigma0)

        self.Ls, self.Ms, self.mus = self.solve_backward_ode()
    
    def find_t(self, t: float):
        return (t / self.dt).astype(int)

    def r(self, 
          t: float, 
          x: jnp.ndarray, 
          L: jnp.ndarray, 
          M: jnp.ndarray,
          mu: jnp.ndarray) -> jnp.ndarray:
        return -L @ self._B(t) @ M @ (self.target - mu - L @ x)
        
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        t_idx = self.find_t(t)
        L, M, mu = self.Ls[t_idx], self.Ms[t_idx], self.mus[t_idx]
        return self._f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, L, M, mu, *args, **kwargs)
    
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self._g(t, x, *args, **kwargs)

    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.g(t, x, *args, **kwargs) @ self.g(t, x, *args, **kwargs).T
    
    def solve_backward_ode(self):
        Ls, Ms, mus = self.backward_ode.solve(self.reverse_ts)
        return Ls, Ms, mus