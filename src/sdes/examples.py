import jax.numpy as jnp

from .base import UnconditionalSDE

########################################################
# Brownian Process
########################################################
class BrownianSDE(UnconditionalSDE):
    """ N-D Brownian process:
        dX_t = _mu * dt + _sigma * dW_t
    """
    _mu: jnp.ndarray
    _sigma: jnp.ndarray
    
    def __init__(self, dim, params):
        super().__init__(dim)
        self._mu    = params["mu"]
        self._sigma = params["sigma"]
        assert self._mu.shape    == (dim, )
        assert self._sigma.shape == (dim, dim)

    def b(self, t, x):
        return self._mu

    def sigma(self, t, x):
        return self._sigma
    
class BrownianBridgeSDE(UnconditionalSDE):
    """ N-D Brownian bridge process:
        dX^*_t = (v_T - X^*_t) / (T - t) dt + _sigma * dW_t
    """
    _mu: jnp.ndarray
    _sigma: jnp.ndarray
    _vT: jnp.ndarray
    _T: float
    
    def __init__(self, dim, params):
        super().__init__(dim)
        self._mu = params["mu"]
        self._sigma = params["sigma"]
        self._vT = params["vT"]
        self._T = params["T"]
        assert self._mu.shape    == (dim, )
        assert self._sigma.shape == (dim, dim)
        assert self._vT.shape    == (dim, )
    
    def b(self, t, x):
        return (self._vT - x) / (self._T - t)
    
    def sigma(self, t, x):
        return self._sigma
    
########################################################
# Ornstein-Uhlenbeck Process
########################################################
class OUSDE(UnconditionalSDE):
    """ N-D Ornstein-Uhlenbeck process:
        dX_t = _theta * (_mu - X_t) * dt + _sigma * dW_t
    """
    _theta: jnp.ndarray
    _mu: jnp.ndarray
    _sigma: jnp.ndarray
    
    def __init__(self, dim, params):
        super().__init__(dim)
        self._theta = params["theta"]
        self._mu    = params["mu"]
        self._sigma = params["sigma"]
        assert self._theta.shape == (dim, dim)
        assert self._mu.shape    == (dim, )
        assert self._sigma.shape == (dim, dim)
        
    def b(self, t, x):
        return self._theta @ (self._mu - x)
    
    def sigma(self, t, x):
        return self._sigma
    
########################################################
# Cell Diffusion Process
########################################################
class CellSDE(UnconditionalSDE):
    _sigma: jnp.ndarray
    
    def __init__(self, params):
        super().__init__(dim=2)
        self._sigma = params["sigma"]

    def b(self, t, x):
        u = lambda x: x**4 / (2.0**(-4) + x**4)
        
        return jnp.array([u(x[0]) + 1.0 - u(x[1]) - x[0],
                          u(x[1]) + 1.0 - u(x[0]) - x[1]])
    
    def sigma(self, t, x):
        return self._sigma
    
########################################################
# FitzHugh-Nagumo Process
########################################################
class FHNSDE(UnconditionalSDE):
    _chi: float
    _s: float
    _gamma: float
    _alpha: float
    _sigma: float
    
    def __init__(self, dim, params):
        super().__init__(dim)
        self._chi = params.get("chi", 0.1)
        self._s = params.get("s", 0.0)
        self._gamma = params.get("gamma", 1.5)
        self._alpha = params.get("alpha", 0.8)
        self._sigma = params.get("sigma", 0.3)
        
    def b(self, t, x):
        term1 = jnp.array([[1. / self._chi, -1. / self._chi],
                           [self._gamma,        -1.]])     # (2, 2)
        term2 = jnp.array([(- x[0]**3 + self._s) / self._chi, self._alpha])     # (2,)
        return jnp.einsum("i j, j -> i", term1, x) + term2  # (2,)
    
    def sigma(self, t, x):
        return jnp.array([0., self._sigma]).reshape(2, 1)     # (2, 1)

########################################################
# Stochastic Landmark Process
########################################################
class LandmarkGaussianSDE(UnconditionalSDE):
    _k_alpha: float
    _k_sigma: float
    n: int              # number of landmarks
    m: int              # dimension of each landmark
    
    def __init__(self, dim, params):
        super().__init__(dim)
        self._k_alpha = params["k_alpha"]
        self._k_sigma = params["k_sigma"]
        self.n        = params["n"]
        self.m        = params["m"]
        assert self.n * self.m == self.dim
    
    @classmethod
    def gaussian_kernel(cls, x1, x2, k_alpha, k_sigma):
        return 0.5 * k_alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x1 - x2), axis=0) / k_sigma**2)
    
    @classmethod
    def compute_q_half(cls, x, m, n, k_alpha, k_sigma):
        x_mn = x.reshape(m, n)
        q_half_block = cls.gaussian_kernel(
            x1=x_mn[:, None, :], 
            x2=x_mn[:, :, None],
            k_alpha=k_alpha,
            k_sigma=k_sigma
        )
        return jnp.block([
            [q_half_block, jnp.zeros_like(q_half_block)],
            [jnp.zeros_like(q_half_block), q_half_block]
        ])
        
    
    def b(self, t, x):
        return jnp.zeros((self.dim, ))
    
    def sigma(self, t, x):
        return self.compute_q_half(
            x=x, 
            m=self.m, 
            n=self.n, 
            k_alpha=self._k_alpha, 
            k_sigma=self._k_sigma
        )