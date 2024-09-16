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
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 *,
                 x0: jnp.ndarray,
                 xT: jnp.ndarray,
                 ):
        super().__init__(ori_proc.T, ori_proc.dt, ori_proc.dim, ori_proc.dtype)
        assert x0.shape == xT.shape
        
        self.ori_proc = ori_proc
        self.aux_proc = aux_proc

        self.x0 = x0
        self.xT = xT

        self.backward_ode = BackwardODE(
            ori_proc, 
            aux_proc, 
            L0, 
            Sigma0=Sigma0)

        self.Ls, self.Ms, self.mus = self.solve_backward_ode()
        self.Hs = jax.vmap(lambda L, M: L.T @ M @ L)(self.Ls, self.Ms)
    
    def find_t(self, t: float):
        return (t / self.dt).astype(int)

    @partial(jax.jit, static_argnums=(0,))
    def r(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        t_idx = self.find_t(t)
        L, M, mu = self.Ls[t_idx], self.Ms[t_idx], self.mus[t_idx]
        return L.T @ M @ (self.xT - mu - L @ x)
    
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, *args, **kwargs)
    
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.g(t, x, *args, **kwargs)

    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.Sigma(t, x, *args, **kwargs)
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.inv_Sigma(t, x, *args, **kwargs)
    
    def solve_backward_ode(self):
        Ls, Ms, mus = self.backward_ode.solve(self.reverse_ts)
        return Ls, Ms, mus
    
    def G(self, t: float, x: jnp.ndarray) -> float:
        r = self.r(t, x)
        term1 = jnp.inner(self.ori_proc.f(t, x) - self.aux_proc.f(t, x), r)
        A = self.ori_proc.Sigma(t, x) - self.aux_proc.Sigma(t, x)
        term2 = -0.5 * jnp.trace(A @ (self.Hs[self.find_t(t)] - r @ r.T))
        return term1 + term2

class MixedGuidedBridgeProcess(GuidedBridgeProcess):
    
    def __init__(self, 
                 ori_proc: ContinuousTimeProcess, 
                 aux_proc: AuxiliaryProcess,
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 *,
                 x0: jnp.ndarray,
                 xT: jnp.ndarray,
                 nu: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 ):
        super().__init__(ori_proc=ori_proc, aux_proc=aux_proc, L0=L0, Sigma0=Sigma0, x0=x0, xT=xT)
        self.nu = nu

    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, *args, **kwargs) \
               + self.g(t, x, *args, **kwargs) @ self.nu(t, x, *args, **kwargs)
