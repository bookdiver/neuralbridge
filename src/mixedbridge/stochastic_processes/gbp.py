from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp

from .bases import ContinuousTimeProcess, AuxiliaryProcess
from ..solvers.ode import BackwardODE

class GuidedBridgeProcess(ContinuousTimeProcess):

    def __init__(self, 
                 ori_proc: ContinuousTimeProcess, 
                 aux_proc: AuxiliaryProcess,
                 x0: jnp.ndarray,
                 v: jnp.ndarray,
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 ts: jnp.ndarray,
                 ode_solver_kernel: str = "dopri5"
                 ):
        super().__init__(ori_proc.T, ori_proc.dim, ori_proc.dtype)
        assert x0.shape == v.shape
        
        self.ori_proc = ori_proc
        self.aux_proc = aux_proc

        self.x0 = x0
        self.v = v

        self.backward_ode = BackwardODE(
            aux_proc=aux_proc, 
            L0=L0, 
            Sigma0=Sigma0,
            kernel=ode_solver_kernel
        )
        
        self.ts = ts
        self.Ls, self.Ms, self.mus = self._solve_backward_ode(ts)
        self.Hs = jax.vmap(lambda L, M: L.T @ M @ L)(self.Ls, self.Ms)
    
    def _find_t(self, t: float):
        return jnp.searchsorted(self.ts, t)

    @partial(jax.jit, static_argnums=(0,))
    def r(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        t_idx = self._find_t(t)
        L, M, mu = self.Ls[t_idx], self.Ms[t_idx], self.mus[t_idx]
        return L.T @ M @ (self.v - mu - L @ x)
    
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, *args, **kwargs)
    
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.g(t, x, *args, **kwargs)

    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.Sigma(t, x, *args, **kwargs)
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.inv_Sigma(t, x, *args, **kwargs)
    
    def _solve_backward_ode(self, ts: jnp.ndarray):
        reverse_ts = jnp.flip(ts)
        Ls, Ms, mus = self.backward_ode.solve(reverse_ts)
        return Ls, Ms, mus
    
    def G(self, t: float, x: jnp.ndarray) -> float:
        r = self.r(t, x)
        term1 = jnp.inner(self.ori_proc.f(t, x) - self.aux_proc.f(t, x), r)
        A = self.ori_proc.Sigma(t, x) - self.aux_proc.Sigma(t, x)
        term2 = -0.5 * jnp.trace(A @ (self.Hs[self._find_t(t)] - r @ r.T))
        return term1 + term2

class MixedGuidedBridgeProcess(GuidedBridgeProcess):
    
    def __init__(self, 
                 ori_proc: ContinuousTimeProcess, 
                 aux_proc: AuxiliaryProcess,
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 x0: jnp.ndarray,
                 v: jnp.ndarray,
                 ts: jnp.ndarray,
                 nu: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 ):
        super().__init__(ori_proc=ori_proc, aux_proc=aux_proc, L0=L0, Sigma0=Sigma0, x0=x0, v=v, ts=ts)
        self.nu = nu

    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, *args, **kwargs) \
               + self.g(t, x, *args, **kwargs) @ self.nu(t, x, *args, **kwargs)
