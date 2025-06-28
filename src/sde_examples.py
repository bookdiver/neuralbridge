import jax.numpy as jnp

from .base import PriorSDE, BridgeSDE, AuxiliarySDE

########################################################
# Brownian Process
########################################################
class BrownianSDE(PriorSDE):
    """ N-D Brownian process with constant drift:
        dX_t = gamma * dt + sigma * dW_t
    """
    def __init__(self, dim_x, dim_w, params):
        super().__init__(dim_x, dim_w)
        self.gamma = params["gamma"]
        self.sigma = params["sigma"]

    def f(self, t, x):
        return self.gamma

    def g(self, t, x):
        return self.sigma
    
class BrownianAuxiliarySDE(AuxiliarySDE):
    """ N-D standard Brownian process:
        dX_t = sigma * dW_t
    """
    def __init__(self, dim_x, dim_w, params):
        super().__init__(dim_x, dim_w)
        self.gamma = params["gamma"]
        
    def B(self, t):
        return jnp.zeros((self.dim_x, self.dim_x))
    
    def beta(self, t):
        return jnp.zeros((self.dim_x, ))
    
    def g(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma
    
    def a(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma**2
    
class BrownianBridgeSDE(BridgeSDE):
    """ N-D Brownian bridge process:
        dX_t = (v - X_t) / (T - t) dt + sigma * dW_t
    """
    def __init__(self, dim_x, dim_w, params, observations):
        super().__init__(dim_x, dim_w, observations)
        self.gamma = params["gamma"]
        self.sigma = params["sigma"]
    
    def f(self, t, x):
        return (self.v - x) / (self.T - t + 1e-6)
    
    def r(self, t, x):
        return (self.v - x - self.gamma * (self.T - t)) / (self.T - t + 1e-6)
    
    def g(self, t, x):
        return jnp.eye(self.dim_x) * self.sigma
    
########################################################
# Ornstein-Uhlenbeck Process
########################################################
class OUSDE(PriorSDE):
    """ N-D Ornstein-Uhlenbeck process:
        dX_t = gamma * (mu - X_t) * dt + sigma * dW_t
        
        with gamma, sigma as scalars, mu as a n-dim vector
    """
    def __init__(self, dim_x, dim_w, params):
        super().__init__(dim_x, dim_w)
        self.gamma = params["gamma"]
        self.mu    = params["mu"]
        self.sigma = params["sigma"]
        
    def f(self, t, x):
        return self.gamma * (self.mu - x)
    
    def g(self, t, x):
        return jnp.eye(self.dim_x) * self.sigma
    
class OUAuxiliarySDE(AuxiliarySDE):
    """ N-D standard Brownian process:
        dX_t = sigma * dW_t
    """
    def __init__(self, dim_x, dim_w, params):
        super().__init__(dim_x, dim_w)
        self.sigma = params["sigma"]
        
    def B(self, t):
        return jnp.zeros((self.dim_x, self.dim_x))
    
    def beta(self, t):
        return jnp.zeros((self.dim_x, ))
    
    def g(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma
    
    def a(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma**2
    
class OUBridgeSDE(BridgeSDE):
    """ N-D Ornstein-Uhlenbeck bridge process:
        dX_t = (gamma * (mu - X_t) + (2 * gamma * exp) / (1 - exp**2) * (v - (X * exp) + mu * (1 - exp))) * dt + sigma * dW_t 
    """
    def __init__(self, dim_x, dim_w, params, observations):
        super().__init__(dim_x, dim_w, observations)
        self.gamma = params["gamma"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        
    def f(self, t, x):
        delta_t = self.T - t
        exp = jnp.exp(-self.gamma * delta_t)
        return (self.gamma * (self.mu - x)
                + (2. * self.gamma * exp) / (1. - exp**2) * (self.v - (x * exp) - self.mu * (1. - exp)))
    
    def r(self, t, x):
        delta_t = self.T - t
        exp = jnp.exp(-self.gamma * delta_t)
        return (2. * self.gamma * exp) / (1. - exp**2) * (self.v - (x * exp) - self.mu * (1. - exp))
    
    def g(self, t, x):
        return jnp.eye(self.dim_x) * self.sigma
    
########################################################
# Sinusoidal Process
########################################################
class SinusoidalSDE(PriorSDE):
    """ 1-D sinusoidal process:
        dX_t = (beta1 - beta2 * sin(8 * X_t)) * dt + sigma * dW_t
    """
    def __init__(self, params):
        super().__init__(dim_x=1, dim_w=1)
        self.beta1 = params["beta1"]
        self.beta2 = params["beta2"]
        self.sigma = params["sigma"]
        
    def f(self, t, x):
        return self.beta1 - self.beta2 * jnp.sin(8. * x)
    
    def g(self, t, x):
        return jnp.eye(self.dim_x) * self.sigma
    
class SinusoidalAuxiliarySDE(AuxiliarySDE):
    """ 1-D standard Brownian process:
        dX_t = sigma * dW_t
    """
    def __init__(self, params):
        super().__init__(dim_x=1, dim_w=1)
        self.sigma = params["sigma"]
        
    def beta(self, t):
        return jnp.zeros((self.dim_x, ))
    
    def B(self, t):
        return jnp.zeros((self.dim_x, self.dim_x))
    
    def g(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma
    
    def a(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma**2
    
########################################################
# Cell Diffusion Process
########################################################
class CellSDE(PriorSDE):
    """ 2-D Cell diffusion process:
        dX^1_t = (u(X^1_t) + 1 - u(X^2_t) - X^1_t) * dt + sigma * dW^1_t
        dX^2_t = (u(X^2_t) + 1 - u(X^1_t) - X^2_t) * dt + sigma * dW^2_t
        
        with u(x) = x**4 / (2.0**(-4) + x**4)
    """
    def __init__(self, params):
        super().__init__(dim_x=2, dim_w=2)
        self.sigma = params["sigma"]

    def f(self, t, x):
        u = lambda x: x**4 / (2.0**(-4) + x**4)
        
        return jnp.array([u(x[0]) + 1.0 - u(x[1]) - x[0],
                          u(x[1]) + 1.0 - u(x[0]) - x[1]])
    
    def g(self, t, x):
        return jnp.eye(self.dim_x) * self.sigma
    
class CellAuxiliarySDE(AuxiliarySDE):
    """ 2-D OU Process:
        dX^1_t = - X^1_t * dt + sigma * dW^1_t
        dX^2_t = - X^2_t * dt + sigma * dW^2_t
    """
    def __init__(self, params):
        super().__init__(dim_x=2, dim_w=2)
        self.sigma = params["sigma"]
    
    def beta(self, t):
        return jnp.zeros((self.dim_x, ))
    
    def B(self, t):
        return -jnp.eye(self.dim_x)
    
    def g(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma
    
    def a(self, t, x=None):
        return jnp.eye(self.dim_x) * self.sigma**2
    
# ########################################################
# # FitzHugh-Nagumo Process
# ########################################################
# class FHNSDE(UnconditionalSDE):
#     _chi: float
#     _s: float
#     _gamma: float
#     _alpha: float
#     _sigma: float
    
#     def __init__(self, params):
#         super().__init__(dim_x=2, dim_w=2)
#         self._chi = params.get("chi", 0.1)
#         self._s = params.get("s", 0.0)
#         self._gamma = params.get("gamma", 1.5)
#         self._alpha = params.get("alpha", 0.8)
#         self._sigma = params.get("sigma", 0.3)
        
#     def b(self, t, x):
#         term1 = jnp.array([[1. / self._chi, -1. / self._chi],
#                            [self._gamma,        -1.]])     # (2, 2)
#         term2 = jnp.array([(- x[0]**3 + self._s) / self._chi, self._alpha])     # (2,)
#         return jnp.dot(term1, x) + term2  # (2,)
    
#     def sigma(self, t, x):
#         return jnp.array([[0., 0.], [0., self._sigma]])

# ########################################################
# # Stochastic Landmark Process
# ########################################################
# class KunitaSDE(UnconditionalSDE):
#     _k_alpha: float
#     _k_sigma: float
#     n: int              # number of landmarks
#     m: int              # dimension of each landmark
#     cell_vol: float
    
#     def __init__(self, params):
#         self._k_alpha = params["k_alpha"]
#         self._k_sigma = params["k_sigma"]
#         self.n        = params["n"]
#         self.m        = params["m"]
#         # --- spatial cell volume -------------------------------------------
#         # assume a unit box in each dimension; change if your domain is larger
#         d           = self.m                      # space dimension
#         dx          = 2.0 / self.n               # grid spacing on [0,1]^d
#         self.cell_vol = dx**d                    # (Δx)^d
#         # -------------------------------------------------------------------
#         super().__init__(dim_x=self.m * self.n, dim_w=self.m * self.n)
    
#     @classmethod
#     def gaussian_kernel(cls, x1, x2, k_alpha, k_sigma):
#         return k_alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x1 - x2), axis=0) / k_sigma**2)
    
#     @classmethod
#     def compute_q_half(cls, x, m, n, k_alpha, k_sigma, cell_vol):
#         x_mn = x.reshape(m, n)
#         # q_half_block = cls.gaussian_kernel(
#         #     x1=x_mn[:, None, :], 
#         #     x2=x_mn[:, :, None],
#         #     k_alpha=k_alpha,
#         #     k_sigma=k_sigma
#         # )
#         k_ij = cls.gaussian_kernel(
#             x1=x_mn[:, None, :],                        # (m, 1, n)
#             x2=x_mn[:, :, None],                       # (m, n, 1)
#             k_alpha=k_alpha,
#             k_sigma=k_sigma,
#         )
#         # multiply by √(Δx)^d  —  the crucial scale factor
#         q_half_block = jnp.sqrt(cell_vol) * k_ij   # (n, n)

#         # build a block-diagonal matrix:  I_m ⊗ q_half_block
#         # this works for any m (x, y, z, … components)
#         sigma = jnp.kron(jnp.eye(m), q_half_block)  # (m n, m n)
#         tr = jnp.trace(sigma @ sigma.T)
#         s = jnp.sqrt(1.0 / tr)
#         return s * sigma
#         # return jnp.block([
#         #     [q_half_block, jnp.zeros_like(q_half_block)],
#         #     [jnp.zeros_like(q_half_block), q_half_block]
#         # ])
        
    
#     def b(self, t, x):
#         return jnp.zeros((self.dim_x, ))
    
#     def sigma(self, t, x):
#         return self.compute_q_half(
#             x=x, 
#             m=self.m, 
#             n=self.n, 
#             k_alpha=self._k_alpha, 
#             k_sigma=self._k_sigma,
#             cell_vol=self.cell_vol
#         )