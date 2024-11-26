from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import (
    ContinuousTimeProcess,
    AuxiliaryProcess
)

class BrownianProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: Optional[float] = 0.0,
                 sigma: Optional[float] = 1.0,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma

    def f(self, t: jnp.ndarray, x: jnp.ndarray, **kwargs):
        return self.gamma * jnp.ones(self.dim, dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class BrownianAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 gamma: Optional[float] = 0.0,
                 sigma: Optional[float] = 1.0,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        
    def beta(self, t: jnp.ndarray, *args, **kwargs):
        return self.gamma * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray, *args, **kwargs):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)

class BrownianBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 score_fn: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None,
                 v: Optional[jnp.ndarray] = None,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        self.v = v
        self.score_fn = score_fn
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        if self.score_fn is not None:
            return self.gamma * jnp.ones(self.dim, dtype=self.dtype) + self.Sigma(t, x, *args, **kwargs) @ self.score_fn(t, x, *args, **kwargs)
        else:
            assert self.v is not None, "v must be provided"
            return self.gamma * jnp.ones(self.dim, dtype=self.dtype) + (self.v - x - self.gamma * (self.T - t)) / (self.T - t + 1e-10)

    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OUProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return - self.gamma * x
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OUAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        
    def beta(self, t: jnp.ndarray, *args, **kwargs):
        return - self.gamma * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray, *args, **kwargs):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    
class OUBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 gamma: float,
                 sigma: float,
                 score_fn: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None,
                 v: Optional[jnp.ndarray] = None,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.gamma = gamma
        self.sigma = sigma
        self.score_fn = score_fn
        self.v = v
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        if self.score_fn is not None:
            return - self.gamma * x + self.Sigma(t, x, *args, **kwargs) @ self.score_fn(t, x, *args, **kwargs)
        else:
            assert self.v is not None, "v must be provided"
            return - self.gamma * (self.v / jnp.sinh(-self.gamma * (self.T - t + 1e-10)) - x / jnp.tanh(-self.gamma * (self.T - t + 1e-10)))
                
    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    

class CellDiffusionProcess(ContinuousTimeProcess):

    def __init__(self, 
                 alpha: float, 
                 sigma: float,
                 T: Optional[float] = 4.0,
                 dim: Optional[int] = 2, 
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.alpha = alpha
        self.sigma = sigma

    def f(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        def u(x, alpha):
            return x**4 / (alpha + x**4)
        
        return jnp.array([u(x[0], self.alpha) + 1.0 - u(x[1], self.alpha) - x[0],
                          u(x[1], self.alpha) + 1.0 - u(x[0], self.alpha) - x[1]],
                          dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return 1.0 / self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class CellDiffusionAuxProcess(AuxiliaryProcess):

    def __init__(self, 
                 alpha: float, 
                 sigma: float, 
                 T: Optional[float] = 4.0,
                 dim: Optional[int] = 2, 
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.alpha = alpha
        self.sigma = sigma

    def beta(self, t: jnp.ndarray, *args, **kwargs):
        return jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray, *args, **kwargs):
        return - 1.0 * jnp.eye(self.dim, dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)

    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)


class LandmarkLagrangianProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 k_alpha: float,
                 k_sigma: float,
                 n_landmarks: Optional[int] = 2,
                 m_landmarks: Optional[int] = 1,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = None,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        self.k_alpha = k_alpha
        self.k_sigma = k_sigma
        self.n_landmarks = n_landmarks
        self.m_landmarks = m_landmarks
        dim = int(n_landmarks * m_landmarks) if dim is None else dim
        super().__init__(T, dim, dtype)
    
    def f(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return jnp.zeros(self.dim, dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        x_landmarks = rearrange(x, "(n m) -> n m", n=self.n_landmarks, m=self.m_landmarks)
        kernel_fn = lambda dist_mat: 0.5 * self.k_alpha * jnp.exp(-jnp.sum(dist_mat**2, axis=-1) / (2.0 * self.k_sigma**2))
        dist_mat = x_landmarks[:, None, :] - x_landmarks[None, :, :]
        corr_mat = kernel_fn(dist_mat)
        Q_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(self.m_landmarks, dtype=self.dtype))
        Q_half = rearrange(Q_half, "i k j l -> (i k) (j l)")
        return Q_half

class LandmarkLagrangianAuxProcess(AuxiliaryProcess):
    
    def __init__(self, 
                 k_alpha: float,
                 k_sigma: float,
                 n_landmarks: Optional[int] = 2,
                 m_landmarks: Optional[int] = 1,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = None,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        self.k_alpha = k_alpha
        self.k_sigma = k_sigma
        self.n_landmarks = n_landmarks
        self.m_landmarks = m_landmarks
        dim = int(n_landmarks * m_landmarks) if dim is None else dim
        self.g0 = None
        super().__init__(T, dim, dtype)
        
    def init_g(self, x0: jnp.ndarray) -> None:
        x0_landmarks = rearrange(x0, "(n m) -> n m", n=self.n_landmarks, m=self.m_landmarks)
        kernel_fn = lambda dist_mat: 0.5 * self.k_alpha * jnp.exp(-jnp.sum(dist_mat**2, axis=-1) / (2.0 * self.k_sigma**2))
        dist_mat = x0_landmarks[:, None, :] - x0_landmarks[None, :, :]
        corr_mat = kernel_fn(dist_mat)
        Q0_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(self.m_landmarks, dtype=self.dtype))
        Q0_half = rearrange(Q0_half, "i k j l -> (i k) (j l)")
        self.g0 = Q0_half
        
    def beta(self, t: jnp.ndarray, *args, **kwargs):
        return jnp.zeros(self.dim, dtype=self.dtype)
    
    def B(self, t: jnp.ndarray, *args, **kwargs):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray, *args, **kwargs):
        assert self.g0 is not None, "g0 must be initialized using .init_g(x0)"
        return self.g0
