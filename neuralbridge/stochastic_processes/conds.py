import flax.linen as nn

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import (
    ContinuousTimeProcess, AuxiliaryProcess
)
from neuralbridge.solvers.ode import BackwardODE

class ForwardBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 ori_proc: ContinuousTimeProcess,
                 score_fn: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(ori_proc.T, ori_proc.dim, dtype)
        
        self.ori_proc = ori_proc
        self.score_fn = score_fn
        
    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        assert self.score_fn is not None, f"Score function is not provided"
        return self.ori_proc.f(t, x) + self.Sigma(t, x) @ self.score_fn(t, x)
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.ori_proc.g(t, x)
    
    def Sigma(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return self.ori_proc.Sigma(t, x)
    
class ReversedBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self, 
                 ori_proc: ContinuousTimeProcess,
                 score_fn: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(ori_proc.T, ori_proc.dim, dtype)
        
        self.ori_proc = ori_proc
        self.score_fn = score_fn
        
    def f(self, t: float, z: jnp.ndarray) -> jnp.ndarray:
        assert self.score_fn is not None, f"Score function is not provided"
        return self.ori_proc.f(self.T - t, z) + self.Sigma(self.T - t, z) @ self.score_fn(self.T - t, z)    # !!! NOTE: right now the Jacobian is not considered yet.
    
    def g(self, t: float, z: jnp.ndarray) -> jnp.ndarray:
        return self.ori_proc.g(self.T - t, z)
    
    def Sigma(self, t: float, z: jnp.ndarray) -> jnp.ndarray:
        return self.ori_proc.Sigma(self.T - t, z)

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
        t = t.squeeze() if t.ndim > 0 else t
        return jnp.searchsorted(self.ts, t)

    @partial(jax.jit, static_argnums=(0,))
    def r(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        t_idx = self._find_t(t)
        L, M, mu = self.Ls[t_idx], self.Ms[t_idx], self.mus[t_idx]
        return L.T @ M @ (self.v - mu - L @ x)
    
    def f(self, t: float, x: jnp.ndarray):
        return self.ori_proc.f(t, x)  \
               + self.Sigma(t, x) @ self.r(t, x)
    
    def g(self, t: float, x: jnp.ndarray):
        return self.ori_proc.g(t, x)

    def Sigma(self, t: float, x: jnp.ndarray):
        return self.ori_proc.Sigma(t, x)
    
    def inv_Sigma(self, t: float, x: jnp.ndarray):
        return self.ori_proc.inv_Sigma(t, x)
    
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

class NeuralBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self, 
                 guided_process: GuidedBridgeProcess,
                 neural_net: nn.Module
        ):
        super().__init__(guided_process.T, guided_process.dim, guided_process.dtype)
        
        self.guided_process = guided_process
        self.neural_net = neural_net
        
    def nu(
        self, t: float, x: jnp.ndarray, variables: Any,
        *, training: bool = False, mutable: Optional[Union[str, List[str], bool]] = False
    ) -> jnp.ndarray:
        t = rearrange(jnp.array([t]), "t -> 1 t")
        x = rearrange(x, "d -> 1 d")
        output, *_ = self.neural_net.apply(
            variables=variables,
            t=t, 
            x=x, 
            training=training,
            mutable=mutable
        )
        if output.ndim == 1:
            return output
        else:
            return rearrange(output, "1 d -> d")

    def f(
        self, t: float, x: jnp.ndarray, variables: Any,
        training: bool = False, mutable: Optional[Union[str, List[str], bool]] = False
    ) -> jnp.ndarray:
        return self.guided_process.f(t, x)  \
               + self.g(t, x) @ self.nu(t, x, variables, training=training, mutable=mutable)
               
    def g(self, t: float, x: jnp.ndarray):
        return self.guided_process.g(t, x)
    
    def G(self, t: float, x: jnp.ndarray):
        return self.guided_process.G(t, x)
