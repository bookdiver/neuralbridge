import flax.linen as nn

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import (
    ContinuousTimeProcess, AuxiliaryProcess
)
from neuralbridge.solvers.ode import BackwardODE

class ReversedBridgeProcess(ContinuousTimeProcess):
    X_unc: ContinuousTimeProcess
    neural_net: nn.Module
    nn_variables: dict

    def __init__(self, 
                 X_unc: ContinuousTimeProcess,
                 T: float,
                 neural_net: nn.Module,
                 nn_variables: FrozenDict = FrozenDict({})):
        super().__init__(X_unc.dim)
        self.X_unc = X_unc
        self.T = T
        self.neural_net = neural_net
        self.nn_variables = nn_variables
        
    def nn_apply(self, t: tTYPE, y: xTYPE, variables: FrozenDict, training: bool, rngs: jax.Array | dict) -> xTYPE:
        t = rearrange(jnp.array([t]), "t -> 1 t")
        y = rearrange(y, "d -> 1 d")
        output = self.neural_net.apply(
            variables=variables if len(variables) > 0 else self.nn_variables,
            t=t,
            x=y,
            training=training,
            rngs=rngs
        )
        output = rearrange(output, "1 d -> d")
        return output
        
    def f(self, t: tTYPE, y: xTYPE, nn_variables: FrozenDict, training: bool, rngs: jax.Array | dict) -> Tuple[xTYPE, xTYPE]:
        nn_val = self.nn_apply(self.T - t, y, nn_variables, training, rngs)
        f_add = jnp.einsum("i j, j -> i", self.Sigma(t, y), nn_val)
        drift = -self.X_unc.f(self.T - t, y) + f_add
        return drift, nn_val
    
    def g(self, t: tTYPE, y: xTYPE) -> xTYPE:
        return self.X_unc.g(self.T - t, y)
    
    def Sigma(self, t: tTYPE, y: xTYPE) -> xTYPE:
        return self.X_unc.Sigma(self.T - t, y)

class GuidedBridgeProcess(ContinuousTimeProcess):
    X_unc: ContinuousTimeProcess    # original unconditional process
    X_aux: AuxiliaryProcess      # auxiliary process
    u: xTYPE                   # starting points
    v: xTYPE                   # conditioning points, may have fewer components than u, missing nans components
    eps: float                  # observation noise variance
    ts: Sequence[tTYPE]        # time grid
    ode_solver_kernel: str      # backward filtering ODE solver kernel
    nn_variables: FrozenDict = FrozenDict({})   # No trainable network

    def __init__(self, 
                 X_unc: ContinuousTimeProcess, 
                 X_aux: AuxiliaryProcess,
                 u: xTYPE,
                 v: xTYPE,
                 eps: float,
                 ts: Sequence[tTYPE],
                 ode_solver_kernel: str = "dopri5"
                 ):
        super().__init__(X_unc.dim)
        
        self.X_unc = X_unc
        self.X_aux = X_aux

        self.u = u
        self.v = v[~jnp.isnan(v)]
        n_observed = len(self.v) 
        
        if len(self.v) == len(self.u):     # fully observed case
            LT = jnp.eye(X_unc.dim)
        else:                            # partially observed case
            n_unobserved = len(self.u) - n_observed
            LT = jnp.concatenate([jnp.eye(n_observed), jnp.zeros((n_observed, n_unobserved))], axis=1)
        
        self.backward_ode = BackwardODE(
            X_aux=X_aux, 
            LT=LT, 
            SigmaT=eps * jnp.eye(n_observed),
            muT=jnp.zeros_like(self.v),
            kernel=ode_solver_kernel
        )
        
        self.ts = ts
        self.Ls, self.Ms, self.mus = self._solve_backward_ode(ts)
        self.Hs = jnp.einsum("t j i, t j k, t k l -> t i l", self.Ls, self.Ms, self.Ls)  # H(t) = L^T(t) M(t) L(t)
        self.Fs = jnp.einsum("t j i, t j k, t k -> t i", self.Ls, self.Ms, self.v[jnp.newaxis, :] - self.mus)  # F(t) = L^T(t) M(t) (v(t) - mu(t))
    
    def _find_t_idx(self, t: tTYPE) -> int:
        t = t.squeeze() if t.ndim > 0 else t
        return jnp.searchsorted(self.ts, t, side="left")

    @partial(jax.jit, static_argnums=(0,))
    def r(self, t: tTYPE, x: xTYPE) -> xTYPE:
        t_idx = self._find_t_idx(t)
        F, H = self.Fs[t_idx], self.Hs[t_idx]
        return F - jnp.einsum("i j, j -> i", H, x)
    
    def f(self, t: tTYPE, x: xTYPE, **kwargs) -> Tuple[xTYPE, xTYPE]:
        drift = self.X_unc.f(t, x) + jnp.einsum("i j, j -> i", self.Sigma(t, x), self.r(t, x))
        return drift, jnp.zeros_like(x)

    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return self.X_unc.g(t, x)

    def Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return self.X_unc.Sigma(t, x)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return self.X_unc.inv_Sigma(t, x)
    
    def _solve_backward_ode(self, ts: Sequence[tTYPE]):
        reverse_ts = jnp.flip(ts)
        Ls, Ms, mus = self.backward_ode.solve(reverse_ts)
        return Ls, Ms, mus
    
    def G(self, t: tTYPE, x: xTYPE) -> xTYPE:
        r = self.r(t, x)
        term1 = jnp.einsum("i, i -> ", self.X_unc.f(t, x) - self.X_aux.f(t, x), r)
        A = self.X_unc.Sigma(t, x) - self.X_aux.Sigma(t, x)
        term2 = -0.5 * jnp.trace(A @ (self.Hs[self._find_t_idx(t)] - jnp.einsum("i, j -> i j", r, r)))
        return term1 + term2

class NeuralBridgeProcess(ContinuousTimeProcess):
    X_gui: GuidedBridgeProcess
    neural_net: nn.Module
    nn_variables: FrozenDict
    u: xTYPE
    v: xTYPE
    
    def __init__(self, 
                 X_gui: GuidedBridgeProcess,
                 neural_net: nn.Module,
                 nn_variables: FrozenDict = FrozenDict({})
        ):
        super().__init__(X_gui.dim)
        
        self.X_gui = X_gui
        self.neural_net = neural_net
        self.nn_variables = nn_variables
        self.u = X_gui.u
        self.v = X_gui.v
        
    def nn_apply(self, t: tTYPE, x: xTYPE, variables: FrozenDict, training: bool, rngs: jax.Array | dict) -> xTYPE:
        t = rearrange(jnp.array([t]), "t -> 1 t")
        x = rearrange(x, "d -> 1 d")
        output = self.neural_net.apply(
            variables=variables if len(variables) > 0 else self.nn_variables,
            t=t,
            x=x,
            training=training,
            rngs=rngs
        )
        output = rearrange(output, "1 d -> d")
        return output

    def f(self, t: tTYPE, x: xTYPE, nn_variables: FrozenDict, training: bool, rngs: jax.Array | dict) -> Tuple[xTYPE, xTYPE]:
        nn_val = self.nn_apply(t, x, nn_variables, training, rngs)
        f_add = jnp.einsum("i j, j -> i", self.g(t, x), nn_val)
        drift = self.X_gui.f(t, x)[0] + f_add
        return drift, nn_val
               
    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return self.X_gui.g(t, x)
    
    def G(self, t: tTYPE, x: xTYPE) -> jnp.ndarray:
        return self.X_gui.G(t, x)
