from neuralbridge.setups import *
from neuralbridge.stochastic_processes.bases import ContinuousTimeProcess, AuxiliaryProcess
from neuralbridge.solvers.ode import BackwardODE

class GuidedBridgeProcess(ContinuousTimeProcess):

    def __init__(self, 
                 ori_proc: ContinuousTimeProcess, 
                 aux_proc: AuxiliaryProcess,
                 u: jnp.ndarray,
                 v: jnp.ndarray,
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 ts: jnp.ndarray,
                 ode_solver_kernel: str = "dopri5"
                 ):
        super().__init__(ori_proc.T, ori_proc.dim, ori_proc.dtype)
        assert u.shape == v.shape
        
        self.ori_proc = ori_proc
        self.aux_proc = aux_proc

        self.u = u
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
    
    def G(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        r = self.r(t, x)
        term1 = jnp.inner(self.ori_proc.f(t, x) - self.aux_proc.f(t, x), r)
        A = self.ori_proc.Sigma(t, x) - self.aux_proc.Sigma(t, x)
        term2 = -0.5 * jnp.trace(A @ (self.Hs[self._find_t(t)] - r @ r.T))
        return term1 + term2
    
    def G_max(self, t: float, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        term1 = jnp.linalg.norm(self.ori_proc.f(t, x) - self.aux_proc.f(t, x))
        term2 = 1.0 + jnp.linalg.norm(x - v) / (self.T - t + 1e-10)
        return term1 * term2

class NeuralGuidedBridgeProcess(GuidedBridgeProcess):
    
    def __init__(self, 
                 guided_process: GuidedBridgeProcess | None = None,
                 *,  # Force keyword arguments after guided_process
                 ori_proc: ContinuousTimeProcess | None = None, 
                 aux_proc: AuxiliaryProcess | None = None,
                 L0: jnp.ndarray | None = None,
                 Sigma0: jnp.ndarray | None = None,
                 u: jnp.ndarray | None = None,
                 v: jnp.ndarray | None = None,
                 ts: jnp.ndarray | None = None,
                 nu: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = lambda: None,
                 ):
        if guided_process is not None:
            # Initialize from existing GuidedBridgeProcess
            self.__dict__.update(guided_process.__dict__)
        else:
            # Initialize from individual components
            super().__init__(
                ori_proc=ori_proc,
                aux_proc=aux_proc,
                L0=L0,
                Sigma0=Sigma0,
                u=u,
                v=v,
                ts=ts
            )
        self.nu = nu

    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        return self.ori_proc.f(t, x, *args, *kwargs)  \
               + self.Sigma(t, x, *args, **kwargs) @ self.r(t, x, *args, **kwargs) \
               + self.g(t, x, *args, **kwargs) @ self.nu(t, x, *args, **kwargs)
