import flax.linen as nn

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import (
    ContinuousTimeProcess, AuxiliaryProcess
)
from neuralbridge.solvers.ode import BackwardODE

class ReversedBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self, 
                 X: ContinuousTimeProcess,
                 score_fn: Callable[[float, jnp.ndarray], jnp.ndarray]):
        super().__init__(X.T, X.dim, X.dtype)
        
        self.X = X
        
        self.score_fn = score_fn
        
    def f(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        assert self.score_fn is not None, f"Score function is not provided"
        return - self.X.f(self.T - t, y) + jnp.einsum("i j, j -> i", self.Sigma(self.T - t, y), self.score_fn(self.T - t, y))    # !!! NOTE: right now the Jacobian is not considered yet.
    
    def g(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        return self.X.g(self.T - t, y)
    
    def Sigma(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        return self.X.Sigma(self.T - t, y)

class GuidedBridgeProcess(ContinuousTimeProcess):

    def __init__(self, 
                 X: ContinuousTimeProcess, 
                 X_tilde: AuxiliaryProcess,
                 u: jnp.ndarray,
                 v: jnp.ndarray,
                 L0: jnp.ndarray,
                 Sigma0: jnp.ndarray,
                 ts: jnp.ndarray,
                 ode_solver_kernel: str = "dopri5"
                 ):
        super().__init__(X.T, X.dim, X.dtype)
        
        self.X = X
        self.X_tilde = X_tilde

        self.u = u
        
        self.v_ = v # could include nans to identify unobserved points
        self.v = self.v_[~jnp.isnan(self.v_)] 

        self.backward_ode = BackwardODE(
            X_tilde=X_tilde, 
            L0=L0, 
            Sigma0=Sigma0,
            mu0=jnp.zeros_like(self.v),
            kernel=ode_solver_kernel
        )
        
        self.ts = ts
        self.Ls, self.Ms, self.mus = self._solve_backward_ode(ts)
        self.Hs = jnp.einsum("b j i, b j k, b k l -> b i l", self.Ls, self.Ms, self.Ls)
    
    def _find_t(self, t: float):
        t = t.squeeze() if t.ndim > 0 else t
        return jnp.searchsorted(self.ts, t)

    @partial(jax.jit, static_argnums=(0,))
    def r(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        t_idx = self._find_t(t)
        L, M, mu, H = self.Ls[t_idx], self.Ms[t_idx], self.mus[t_idx], self.Hs[t_idx]
        F = jnp.einsum("j i, j k, k -> i", L, M, self.v - mu)
        return F - jnp.einsum("i j, j -> i", H, x)
    
    def f(self, t: float, x: jnp.ndarray):
        return self.X.f(t, x)  \
               + self.Sigma(t, x) @ self.r(t, x)
    
    def g(self, t: float, x: jnp.ndarray):
        return self.X.g(t, x)

    def Sigma(self, t: float, x: jnp.ndarray):
        return self.X.Sigma(t, x)
    
    def inv_Sigma(self, t: float, x: jnp.ndarray):
        return self.X.inv_Sigma(t, x)
    
    def _solve_backward_ode(self, ts: jnp.ndarray):
        reverse_ts = jnp.flip(ts)
        Ls, Ms, mus = self.backward_ode.solve(reverse_ts)
        return Ls, Ms, mus
    
    def G(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        r = self.r(t, x)
        term1 = jnp.einsum("i, i -> ", self.X.f(t, x) - self.X_tilde.f(t, x), r)
        A = self.X.Sigma(t, x) - self.X_tilde.Sigma(t, x)
        term2 = -0.5 * jnp.trace(A @ (self.Hs[self._find_t(t)] - jnp.einsum("i, j -> i j", r, r)))
        return term1 + term2

class NeuralBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self, 
                 X_circ: GuidedBridgeProcess,
                 neural_net: nn.Module
        ):
        super().__init__(X_circ.T, X_circ.dim, X_circ.dtype)
        
        self.X_circ = X_circ
        self.neural_net = neural_net
        
    def nu(
        self, t: float, x: jnp.ndarray, variables: Any
    ) -> jnp.ndarray:
        t = rearrange(jnp.array([t]), "t -> 1 t")
        x = rearrange(x, "d -> 1 d")
        output = self.neural_net.apply(
            variables=variables,
            t=t,
            x=x
        )
        output = rearrange(output, "1 d -> d")
        return output

    def f(
        self, t: float, x: jnp.ndarray, variables: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nu = self.nu(t, x, variables)
        return self.X_circ.f(t, x) + jnp.einsum("i j, j -> i", self.g(t, x), nu), nu
               
    def g(self, t: float, x: jnp.ndarray):
        return self.X_circ.g(t, x)
    
    def G(self, t: float, x: jnp.ndarray):
        return self.X_circ.G(t, x)
