import abc
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from .bif import BackwardInformationFilter
from .nn import MLPSmall, MLPMedium, MLPLarge

class BaseSDE(abc.ABC):
    """Abstract base class for all Itô stochastic differential equations.
    
    Defines the standard interface for SDEs in the form:
        dX_t = f(t, X_t) dt + g(t, X_t) dW_t
    """
    def __init__(self, dim_x, dim_w):
        """Initialize SDE with specified dimensions."""
        self.dim_x = dim_x
        self.dim_w = dim_w
    
    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """Drift coefficient f(t, x)."""
        pass
    
    @abc.abstractmethod
    def g(self, *args, **kwargs):
        """Diffusion coefficient g(t, x)."""
        pass
        
    
class PriorSDE(BaseSDE):
    """Base class for prior SDEs with time-dependent coefficients.
    
    Standard SDE: dX_t = f(t, X_t) dt + g(t, X_t) dW_t
    """
    def __init__(self, dim_x, dim_w):
        super().__init__(dim_x, dim_w)
        
    @abc.abstractmethod
    def f(self, t, x):
        pass
    
    @abc.abstractmethod
    def g(self, t, x):
        pass

    def a(self, t, x):
        g = self.g(t, x)
        return g @ g.T

class BridgeSDE(BaseSDE):
    """Base class for bridge SDEs (only applicable for a few cases with closed-form solutions).
    
    Bridge SDE: dX_t = (f(t, X_t) + g(t, X_t) r(t, X_t)) dt + g(t, X_t) dW_t
    
    where r(t, X_t) is the score function of p(v | X_t)
    """
    def __init__(self, dim_x, dim_w, observations):
        super().__init__(dim_x, dim_w)
        self.v = observations["v"]
        self.T = observations["T"]
    
    @abc.abstractmethod
    def f(self, t, x):
        pass
    
    @abc.abstractmethod
    def g(self, t, x):
        pass
    
    @abc.abstractmethod
    def r(self, t, x):
        pass
    
    def a(self, t, x):
        g = self.g(t, x)
        return g @ g.T
    
class AuxiliarySDE(BaseSDE):
    """Auxiliary SDE with linear coefficients.
    
    Linear SDE of the form:
        dX_t = (β(t) + B(t)X_t) dt + g(t) dW_t
    
    Used in guided proposals to approximate the bridge law.
    """
    def __init__(self, dim_x, dim_w):
        super().__init__(dim_x, dim_w)
        
    @abc.abstractmethod
    def B(self, t):
        pass
    
    @abc.abstractmethod
    def beta(self, t):
        pass

    def f(self, t, x):
        return self.beta(t) + self.B(t) @ x
    
    @abc.abstractmethod
    def g(self, t, x=None):
        pass
    
    @abc.abstractmethod
    def a(self, t, x=None):
        pass
    
class GuidedProposalSDE(BaseSDE):
    """Guided proposal SDE obtained by solving backward filtering ODEs
    
    Guided proposal SDE: dX_t = (f(t, X_t) + g(t, X_t) r_tilde(t, X_t)) dt + g(t, X_t) dW_t
    
    where r_tilde(t, X_t) is the score function of p_tilde(v | X_t), i.e., the transition of the auxiliary SDE.
    """
    def __init__(self, prior_sde, auxiliary_sde, observations, ts, eq_type="cFH"):
        super().__init__(prior_sde.dim_x, prior_sde.dim_w)
        
        self.prior_sde = prior_sde
        self.auxiliary_sde = auxiliary_sde
        self.ts = ts
        
        bif = BackwardInformationFilter(
            prior_sde=self.prior_sde, 
            auxiliary_sde=self.auxiliary_sde,
            observations=observations
        )
        
        self.Hs, self.Fs, _ = bif.solve(
            ts=self.ts,
            eq_type=eq_type
        )
    
    def _find_t_idx(self, t):
        t = t.squeeze() if t.ndim > 0 else t
        return jnp.searchsorted(self.ts, t, side="left")

    def r_tilde(self, t, x):
        t_idx = self._find_t_idx(t)
        F, H = self.Fs[t_idx], self.Hs[t_idx]
        return F - jnp.dot(H, x)
    
    def f(self, t, x):
        return self.prior_sde.f(t, x) + self.a(t, x) @ self.r_tilde(t, x)

    def g(self, t, x):
        return self.prior_sde.g(t, x)

    def a(self, t, x):
        return self.prior_sde.a(t, x)
    
    def G(self, t, x):
        t_idx = self._find_t_idx(t)
        H = self.Hs[t_idx]
        r_tilde = self.r_tilde(t, x)
        f_diff = self.prior_sde.f(t, x) - self.auxiliary_sde.f(t, x)
        a_diff = self.prior_sde.a(t, x) - self.auxiliary_sde.a(t, x)
        term1 = f_diff @ r_tilde
        term2 = -0.5 * jnp.trace(a_diff @ (H - jnp.outer(r_tilde, r_tilde)))
        return term1 + term2
    
class NeuralBridgeSDE(nnx.Module):
    """Neural Guided Diffusion Bridge SDE.
    
    Neural Bridge SDE: dX_t = (f(t, X_t) + g(t, X_t) r_tilde(t, X_t) + g(t, X_t) nn(t, X_t)) dt + g(t, X_t) dW_t
    
    where nn(t, X_t) is a parameterised neural network.
    """
    def __init__(self, guided_proposal_sde, nn_config):
        self.guided_proposal_sde = guided_proposal_sde
        self.dim_x = guided_proposal_sde.dim_x
        self.dim_w = guided_proposal_sde.dim_w
        
        self.nn_type = nn_config['nn_type']
        self.nn_kwargs =  {k: v for k, v in nn_config.items() if k != 'nn_type'}
        
        self.nn = self._create_nn()
        
    def _create_nn(self):
        nn_classes = {
            "MLPSmall": MLPSmall,
            "MLPMedium": MLPMedium, 
            "MLPLarge": MLPLarge
        }
        
        if self.nn_type not in nn_classes:
            raise ValueError(f"Neural network type '{self.nn_type}' not recognized. Available types: {nn_classes.keys()}")
        
        return nn_classes[self.nn_type](
            input_dim=self.guided_proposal_sde.dim_x,
            output_dim=self.guided_proposal_sde.dim_w,
            **self.nn_kwargs
        )
    
    def f(self, t, x, verbose=False):
        f_circ = self.guided_proposal_sde.f(t, x)
        nn_output = self.nn(t, x)
        if verbose:
            return f_circ + self.g(t, x) @ nn_output, nn_output
        else:
            return f_circ + self.g(t, x) @ nn_output
    
    def g(self, t, x):
        return self.guided_proposal_sde.g(t, x)
    
    def a(self, t, x):
        return self.guided_proposal_sde.a(t, x)
    
    def solve_with_gradients(self, rng_key, x0, ts):
        dts = jnp.diff(ts)
        n_steps = len(dts)
        dWs = jr.normal(rng_key, (n_steps, self.dim_w)) * jnp.sqrt(dts[:, None])
        
        def scan_body(carry, vals):
            t, x, ll = carry
            dt, dW = vals
            drift, v_new = self.f(t, x, verbose=True)
            x_new = x + drift * dt + self.g(t, x) @ dW
            ll_new = ll + self.guided_proposal_sde.G(t, x) * dt
            t_new = t + dt
            return (t_new, x_new, ll_new), (x_new, v_new)
        
        (_, _, ll), (xs, vs) = jax.lax.scan(
            scan_body,
            init=(0.0, x0, 0.0),
            xs=(dts, dWs),
            length=n_steps
        )
        return xs, (vs, ll)