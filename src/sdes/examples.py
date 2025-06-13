import jax.numpy as jnp

from .base import UnconditionalSDE, BridgeSDE

########################################################
# Brownian Process
########################################################
class BrownianSDE(UnconditionalSDE):
    """ N-D Brownian process:
        dX_t = \gamma * dt + \sigma * dW_t
    """
    _gamma: jnp.ndarray
    _sigma: jnp.ndarray
    
    def __init__(self, dim_x, dim_w, params):
        super().__init__(dim_x, dim_w)
        self._gamma = params["gamma"]
        self._sigma = params["sigma"]

    def b(self, t, x):
        return self._gamma

    def sigma(self, t, x):
        return self._sigma
    
class BrownianBridgeSDE(BridgeSDE):
    """ N-D Brownian bridge process:
        dX^*_t = (v_T - X^*_t) / (T - t) dt + \sigma * dW_t
    """
    _gamma: jnp.ndarray
    _sigma: jnp.ndarray
    _vT: jnp.ndarray
    _T: float
    
    def __init__(self, dim_x, dim_w, params, vT, T):
        super().__init__(dim_x, dim_w, vT, T)
        self._gamma = params["gamma"]
        self._sigma = params["sigma"]
    
    def b(self, t, x):
        return (self._vT - x) / (self._T - t)
    
    def sigma(self, t, x):
        return self._sigma
    
########################################################
# Ornstein-Uhlenbeck Process
########################################################
class OUSDE(UnconditionalSDE):
    """ 1-D Ornstein-Uhlenbeck process:
        dX_t = \gamma * (\mu - X_t) * dt + \sigma * dW_t
    """
    _gamma: jnp.ndarray
    _mu: jnp.ndarray
    _sigma: jnp.ndarray
    
    def __init__(self, dim_x, dim_w, params):
        super().__init__(dim_x, dim_w)
        self._gamma = params["gamma"]
        self._mu    = params["mu"]
        self._sigma = params["sigma"]
        
    def b(self, t, x):
        return self._gamma @ (self._mu - x)
    
    def sigma(self, t, x):
        return self._sigma
    
class OUBridgeSDE(BridgeSDE):
    """ 1-D Ornstein-Uhlenbeck bridge process:
        dX^*_t = (\gamma * (\mu - X^*_t) + \gamma / sinh(\gamma * (T - t)) * ((v_T - \mu) - exp(-\gamma * (T - t)) * (X^*_t - \mu))) dt + \sigma * dW_t
        NOTE: This is a 1-D process. n-d is much more complicated.
    """
    _gamma: jnp.ndarray
    _mu: jnp.ndarray
    _sigma: jnp.ndarray
    
    def __init__(self, params, vT, T):
        super().__init__(dim_x=1, dim_w=1, vT=vT, T=T)
        self._gamma = params["gamma"]
        self._mu = params["mu"]
        self._sigma = params["sigma"]
    
    def b(self, t, x):
        delta = self._T - t
        exp_delta = jnp.exp(-self._gamma * delta)
        drift1 = self._gamma * (self._mu - x)
        drift2 = (2. * self._gamma * exp_delta) / (self._sigma[0][0]**2 * (1. - exp_delta**2)) * (self._vT - self._mu - exp_delta * (x - self._mu))
        return drift1 + self._sigma[0][0]**2 * drift2
    
    def sigma(self, t, x):
        return self._sigma
        
        
########################################################
# Cell Diffusion Process
########################################################
class CellSDE(UnconditionalSDE):
    _sigma: jnp.ndarray
    
    def __init__(self, params):
        super().__init__(dim_x=2, dim_w=2)
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
    
    def __init__(self, params):
        super().__init__(dim_x=2, dim_w=2)
        self._chi = params.get("chi", 0.1)
        self._s = params.get("s", 0.0)
        self._gamma = params.get("gamma", 1.5)
        self._alpha = params.get("alpha", 0.8)
        self._sigma = params.get("sigma", 0.3)
        
    def b(self, t, x):
        term1 = jnp.array([[1. / self._chi, -1. / self._chi],
                           [self._gamma,        -1.]])     # (2, 2)
        term2 = jnp.array([(- x[0]**3 + self._s) / self._chi, self._alpha])     # (2,)
        return jnp.dot(term1, x) + term2  # (2,)
    
    def sigma(self, t, x):
        return jnp.array([[0., 0.], [0., self._sigma]])

########################################################
# Stochastic Landmark Process
########################################################
class KunitaSDE(UnconditionalSDE):
    _k_alpha: float
    _k_sigma: float
    n: int              # number of landmarks
    m: int              # dimension of each landmark
    cell_vol: float
    
    def __init__(self, params):
        self._k_alpha = params["k_alpha"]
        self._k_sigma = params["k_sigma"]
        self.n        = params["n"]
        self.m        = params["m"]
        # --- spatial cell volume -------------------------------------------
        # assume a unit box in each dimension; change if your domain is larger
        d           = self.m                      # space dimension
        dx          = 2.0 / self.n               # grid spacing on [0,1]^d
        self.cell_vol = dx**d                    # (Δx)^d
        # -------------------------------------------------------------------
        super().__init__(dim_x=self.m * self.n, dim_w=self.m * self.n)
    
    @classmethod
    def gaussian_kernel(cls, x1, x2, k_alpha, k_sigma):
        return k_alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x1 - x2), axis=0) / k_sigma**2)
    
    @classmethod
    def compute_q_half(cls, x, m, n, k_alpha, k_sigma, cell_vol):
        x_mn = x.reshape(m, n)
        # q_half_block = cls.gaussian_kernel(
        #     x1=x_mn[:, None, :], 
        #     x2=x_mn[:, :, None],
        #     k_alpha=k_alpha,
        #     k_sigma=k_sigma
        # )
        k_ij = cls.gaussian_kernel(
            x1=x_mn[:, None, :],                        # (m, 1, n)
            x2=x_mn[:, :, None],                       # (m, n, 1)
            k_alpha=k_alpha,
            k_sigma=k_sigma,
        )
        # multiply by √(Δx)^d  —  the crucial scale factor
        q_half_block = jnp.sqrt(cell_vol) * k_ij   # (n, n)

        # build a block-diagonal matrix:  I_m ⊗ q_half_block
        # this works for any m (x, y, z, … components)
        sigma = jnp.kron(jnp.eye(m), q_half_block)  # (m n, m n)
        tr = jnp.trace(sigma @ sigma.T)
        s = jnp.sqrt(1.0 / tr)
        return s * sigma
        # return jnp.block([
        #     [q_half_block, jnp.zeros_like(q_half_block)],
        #     [jnp.zeros_like(q_half_block), q_half_block]
        # ])
        
    
    def b(self, t, x):
        return jnp.zeros((self.dim_x, ))
    
    def sigma(self, t, x):
        return self.compute_q_half(
            x=x, 
            m=self.m, 
            n=self.n, 
            k_alpha=self._k_alpha, 
            k_sigma=self._k_sigma,
            cell_vol=self.cell_vol
        )