from functools import partial
import jax
import jax.numpy as jnp

from .bases import ContinuousTimeProcess, AuxiliaryProcess
from ..solvers.ode import BackwardODE
from ..utils.sample_path import SamplePath

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
        
        self.ori_process = original_process
        self.aux_process = auxiliary_process

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
        self.Hs = jax.vmap(lambda L, M: L.T @ M @ L)(self.Ls, self.Ms)
    
    def find_t(self, t: float):
        return (t / self.dt).astype(int)

    @partial(jax.jit, static_argnums=(0,))
    def r(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        t_idx = self.find_t(t)
        L, M, mu = self.Ls[t_idx], self.Ms[t_idx], self.mus[t_idx]
        return -L.T @ self.aux_process.B(t) @ M @ (self.target - mu - L @ x)
    
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_process.f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, *args, **kwargs)
    
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_process.g(t, x, *args, **kwargs)

    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.g(t, x, *args, **kwargs) @ self.g(t, x, *args, **kwargs).T
    
    def solve_backward_ode(self):
        Ls, Ms, mus = self.backward_ode.solve(self.reverse_ts)
        return Ls, Ms, mus
    
    def G(self, t: float, x: jnp.ndarray) -> float:
        r = self.r(t, x)
        term1 = (self.ori_process.f(t, x) - self.aux_process.f(t, x)) @ r
        term2 = 0.5 * jnp.trace(
            (self.ori_process.Sigma(t, x) - self.aux_process.Sigma(t, x))
            @ (self.Hs[self.find_t(t)] - r @ r.T)
        )
        return term1 + term2
    
    def log_likelihood(self, sample_path: SamplePath, skip: int = 25) -> float:
        ts, xs = sample_path.ts, sample_path.xs
        res = 0.0

        def scan_fn(carry: tuple, val: tuple) -> tuple:
            log_psi = carry
            t, t_next, x = val
            G = self.G(t, x)
            log_psi += G * (t_next - t)

            return log_psi, (t, t_next, x)

        slice1 = slice(None, -(skip+1))
        slice2 = slice(1, -skip) if skip > 0 else slice(1, None) 
        res, _ = jax.lax.scan(scan_fn, 0.0, (ts[slice1], ts[slice2], xs[slice1])) 
        
        return res