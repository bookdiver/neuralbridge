from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import (
    ContinuousTimeProcess,
    AuxiliaryProcess
)

class BrownianProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params

    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class BrownianAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        
    def beta(self, t: jnp.ndarray):
        return self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)

class BrownianBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 v: Optional[jnp.ndarray] = None,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        self.v = v
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        assert self.v is not None, "v must be provided"
        return self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype) + (self.v - x - self.params["gamma"] * (self.T - t)) / (self.T - t + 1e-10)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OUProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        return - self.params["gamma"] * x
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OUAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        
    def beta(self, t: jnp.ndarray):
        return - self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    
class OUBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 v: Optional[jnp.ndarray] = None,
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        self.v = v
    
    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        assert self.v is not None, "v must be provided"
        return - self.params["gamma"] * (self.v / jnp.sinh(-self.params["gamma"] * (self.T - t + 1e-10)) - x / jnp.tanh(-self.params["gamma"] * (self.T - t + 1e-10)))
                
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    

class CellDiffusionProcess(ContinuousTimeProcess):

    def __init__(self, 
                 params: Dict[str, float],
                 T: Optional[float] = 4.0,
                 dim: Optional[int] = 2, 
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params

    def f(self, t: jnp.ndarray, x: jnp.ndarray):
        def u(x, alpha):
            return x**4 / (alpha + x**4)
        
        return jnp.array([u(x[0], self.params["alpha"]) + 1.0 - u(x[1], self.params["alpha"]) - x[0],
                          u(x[1], self.params["alpha"]) + 1.0 - u(x[0], self.params["alpha"]) - x[1]],
                          dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class CellDiffusionAuxProcess(AuxiliaryProcess):

    def __init__(self, 
                 params: Dict[str, float],
                 T: Optional[float] = 4.0,
                 dim: Optional[int] = 2, 
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params

    def beta(self, t: jnp.ndarray):
        return jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: jnp.ndarray):
        return - 1.0 * jnp.eye(self.dim, dtype=self.dtype)

    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma * jnp.eye(self.dim, dtype=self.dtype)

    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray):
        return self.sigma**2 * jnp.eye(self.dim, dtype=self.dtype)

class LandmarkLagrangianProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = None,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        self.params = params
        dim = int(self.params["n_landmarks"] * self.params["m_landmarks"]) if dim is None else dim
        super().__init__(T, dim, dtype)
    
    def f(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((self.dim, ), dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        x_landmarks = rearrange(x, "(n d) -> n d", n=self.params["n_landmarks"])
        kernel_fn = lambda dist_mat: 0.5 * self.params["k_alpha"] * jnp.exp(-jnp.sum(dist_mat**2, axis=-1) / (2.0 * self.params["k_sigma"]**2))
        dist_mat = x_landmarks[:, None, :] - x_landmarks[None, :, :]
        corr_mat = kernel_fn(dist_mat)
        q_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(self.params["m_landmarks"], dtype=self.dtype))
        q_half = rearrange(q_half, "i k j l -> (i k) (j l)")
        return q_half

class LandmarkLagrangianAuxProcess(AuxiliaryProcess):
    
    def __init__(self, 
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = None,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        self.params = params
        dim = int(self.params["n_landmarks"] * self.params["m_landmarks"]) if dim is None else dim
        self.g0 = None
        super().__init__(T, dim, dtype)
        
    def init_g(self, v: jnp.ndarray) -> None:
        v_landmarks = rearrange(v, "(n d) -> n d", n=self.params["n_landmarks"])
        kernel_fn = lambda dist_mat: 0.5 * self.params["k_alpha"] * jnp.exp(-jnp.sum(dist_mat**2, axis=-1) / (2.0 * self.params["k_sigma"]**2))
        dist_mat = v_landmarks[:, None, :] - v_landmarks[None, :, :]
        corr_mat = kernel_fn(dist_mat)
        q_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(self.params["m_landmarks"], dtype=self.dtype))
        q_half = rearrange(q_half, "i k j l -> (i k) (j l)")
        self.g0 = q_half
        
    def beta(self, t: jnp.ndarray):
        return jnp.zeros((self.dim, ), dtype=self.dtype)
    
    def B(self, t: jnp.ndarray):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray):
        assert self.g0 is not None, "g0 must be initialized using .init_g(x0)"
        return self.g0

class FitzHughNagumoProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 10.0,
                 dim: Optional[int] = 2,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        
    def f(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        term1 = jnp.array([[1. / self.params["epsilon"], -1. / self.params["epsilon"]],
                           [self.params["gamma"],        -1.]], 
                            dtype=self.dtype)     # (2, 2)
        term2 = jnp.array([(- x[0]**3 + self.params["s"]) / self.params["epsilon"], self.params["beta"]], dtype=self.dtype)     # (2,)
        return jnp.einsum("i j, j -> i", term1, x) + term2  # (2,)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0., self.params["sigma"]], dtype=self.dtype).reshape(2, 1)     # (2, 1)
        
class FitzHughNagumoAuxProcess(AuxiliaryProcess):
    
    def __init__(self, 
                 params: Dict[str, float],
                 T: Optional[float] = 10.0,
                 dim: Optional[int] = 2,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        
    def B(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[(1. - 3. * self.params["nu"]**2) / self.params["epsilon"], -1. / self.params["epsilon"]],
                          [self.params["gamma"],                                         -1.]], 
                          dtype=self.dtype)     # (2, 2)
        
    def beta(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([(2. * self.params["nu"]**3 + self.params["s"]) / self.params["epsilon"],
                          self.params["beta"]], 
                          dtype=self.dtype)     # (2,)  
        
    def g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[0.],
                          [self.params["sigma"]]], 
                         dtype=self.dtype)


class DoubleWellPotentialDiffusionProcess(ContinuousTimeProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
    
    def f(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return x - x**3
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)


class DoubleWellPotentialDiffusionAuxProcess(AuxiliaryProcess):
    
    def __init__(self,
                 params: Dict[str, float],
                 T: Optional[float] = 1.0,
                 dim: Optional[int] = 1,
                 dtype: Optional[jnp.dtype] = jnp.float32):
        super().__init__(T, dim, dtype)
        self.params = params
        
    def beta(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((self.dim, ), dtype=self.dtype)
    
    def B(self, t: jnp.ndarray) -> jnp.ndarray:
        return - 1.0 * jnp.eye(self.dim, dtype=self.dtype)
    
    def g(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
