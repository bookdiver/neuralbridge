import abc
import jax.numpy as jnp

from .bif import BackwardInformationFilter

class BaseSDE(abc.ABC):
    dim: int
    
    def __init__(self, dim):
        self.dim = dim
    
    @abc.abstractmethod
    def b(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def sigma(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def a(self, *args, **kwargs):
        pass
        
    
class UnconditionalSDE(BaseSDE):
    dim: int
    
    def __init__(self, dim):
        super().__init__(dim)
        
    @abc.abstractmethod
    def b(self, t, x):
        pass
    
    @abc.abstractmethod
    def sigma(self, t, x):
        pass

    def a(self, t, x):
        sigma = self.sigma(t, x)
        return sigma @ sigma.T
        
    
class AuxiliarySDE(BaseSDE):
    dim: int
    T: float
    _beta_func: callable
    _B_func: callable
    _a_tilde_T: jnp.ndarray
    _a_tilde_0: jnp.ndarray
    
    def __init__(self, dim, coeffs):
        super().__init__(dim)
        self.T = coeffs["T"]
        self._beta_func = coeffs["beta"]
        self._B_func = coeffs["B"]
        self._a_tilde_0 = coeffs["a_tilde_0"]
        self._a_tilde_T = coeffs["a_tilde_T"]
        fake_t = jnp.array(0.0)
        assert self._beta_func(fake_t).shape == (dim, )
        assert self._B_func(fake_t).shape == (dim, dim)
        assert self._a_tilde_0.shape == (dim, dim) if self._a_tilde_0 is not None else True
        assert self._a_tilde_T.shape == (dim, dim)
        
    def B(self, t):
        return self._B_func(t)
    
    def beta(self, t):
        return self._beta_func(t)

    def b(self, t, x):
        return self.beta(t) + self.B(t) @ x
    
    def sigma(self, t, x):
        a_tilde = self.a(t, x)
        return jnp.linalg.cholesky(a_tilde)
    
    def a(self, t, x):
        if self._a_tilde_0 is None:
            return self._a_tilde_T
        else:
            return self._a_tilde_T * (t / self.T) + self._a_tilde_0 * (1.0 - t / self.T)
    

class GuidedBridgeSDE(BaseSDE):
    sde: UnconditionalSDE
    aux_sde: AuxiliarySDE 
    dim: int
    ts: jnp.ndarray
    
    # backward information filtering variables
    Hs: jnp.ndarray
    Fs: jnp.ndarray
    
    def __init__(self, sde, aux_sde, obs_params, ts, eq_type="cFH"):
        super().__init__(sde.dim)
        
        self.sde = sde
        self.aux_sde = aux_sde
        self.ts = ts
        
        bif = BackwardInformationFilter(
            sde=self.sde, 
            aux_sde=self.aux_sde,
            obs_params=obs_params
        )
        
        self.Hs, self.Fs, _ = bif.solve(
            ts=self.ts,
            eq_type=eq_type
        )
    
    def _find_t_idx(self, t):
        t = t.squeeze() if t.ndim > 0 else t
        return jnp.searchsorted(self.ts, t, side="left")

    def r(self, t, x):
        t_idx = self._find_t_idx(t)
        F, H = self.Fs[t_idx], self.Hs[t_idx]
        return F - H @ x
    
    def b(self, t, x):
        return self.sde.b(t, x) + self.a(t, x) @ self.r(t, x)

    def sigma(self, t, x):
        return self.sde.sigma(t, x)

    def a(self, t, x):
        return self.sde.a(t, x)
    
    def G(self, t, x):
        t_idx = self._find_t_idx(t)
        H = self.Hs[t_idx]
        r = self.r(t, x)
        b_diff = self.sde.b(t, x) - self.aux_sde.b(t, x)
        a_diff = self.sde.a(t, x) - self.aux_sde.a(t, x)
        term1 = jnp.dot(b_diff, r)
        term2 = -0.5 * jnp.trace(a_diff @ (H - jnp.outer(r, r)))
        return term1 + term2