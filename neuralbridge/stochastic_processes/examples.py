from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import (
    ContinuousTimeProcess,
    AuxiliaryProcess
)

class BrownianProcess(ContinuousTimeProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params

    def f(self, t: tTYPE, x: xTYPE, **kwargs):
        return self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype)

    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class BrownianAuxProcess(AuxiliaryProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
        
    def beta(self, t: tTYPE):
        return self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: tTYPE):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)

    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)

class BrownianBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self, dim: int, params: Dict[str, float], T: float,v: Optional[jnp.ndarray] = None):
        super().__init__(dim)
        self.params = params
        self.T = T
        self.v = v
        
    def f(self, t: tTYPE, x: xTYPE, **kwargs):
        assert self.v is not None, "v must be provided"
        drift = self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype) + (self.v - x - self.params["gamma"] * (self.T - t)) / (self.T - t + 1e-10)
        return (drift, jnp.zeros_like(drift))

    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OrnsteinUhlenbeckProcess(ContinuousTimeProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
        
    def f(self, t: tTYPE, x: xTYPE, **kwargs):
        return -self.params["gamma"] * x
    
    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class OrnsteinUhlenbeckAuxProcess(AuxiliaryProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
        
    def beta(self, t: tTYPE):
        return -self.params["gamma"] * jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: tTYPE):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    
class OrnsteinUhlenbeckBridgeProcess(ContinuousTimeProcess):
    
    def __init__(self, dim: int, params: Dict[str, float], T: float, v: Optional[jnp.ndarray] = None):
        super().__init__(dim)
        self.params = params
        self.T = T
        self.v = v
    
    def f(self, t: tTYPE, x: xTYPE, **kwargs):
        assert self.v is not None, "v must be provided"
        return - self.params["gamma"] * (self.v / jnp.sinh(-self.params["gamma"] * (self.T - t + 1e-10)) - x / jnp.tanh(-self.params["gamma"] * (self.T - t + 1e-10)))
    # def f(self, t: tTYPE, x: xTYPE, **kwargs):
    #     mean = jnp.exp(-self.params["gamma"] * (self.T - t + 1e-10)) * x + \
    #         self.params["mu"] * (1 - jnp.exp(-self.params["gamma"] * (self.T - t + 1e-10)))
    #     grad_mean = jnp.exp(-self.params["gamma"] * (self.T - t + 1e-10))
    #     var = self.params["sigma"]**2 / (2.0 * self.params["gamma"]) * \
    #         (jnp.exp(-self.params["gamma"] * (self.T - t + 1e-10)) - \
    #          jnp.exp(-self.params["gamma"] * (self.T + t + 1e-10)))
    #     score = (self.v - mean) / var * grad_mean
    #     return self.params["gamma"] * (self.params["mu"] - x) + self.Sigma(t, x) @ score
    
    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    

class CellDiffusionProcess(ContinuousTimeProcess):

    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params

    def f(self, t: tTYPE, x: xTYPE, **kwargs):
        def u(x, alpha):
            return x**4 / (alpha + x**4)
        
        return jnp.array([u(x[0], self.params["alpha"]) + 1.0 - u(x[1], self.params["alpha"]) - x[0],
                          u(x[1], self.params["alpha"]) + 1.0 - u(x[0], self.params["alpha"]) - x[1]],
                          dtype=self.dtype)
    
    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
    def inv_Sigma(self, t: tTYPE, x: xTYPE):
        return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class CellDiffusionAuxProcess(AuxiliaryProcess):

    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params

    def beta(self, t: tTYPE):
        return jnp.ones(self.dim, dtype=self.dtype)

    def B(self, t: tTYPE):
        return - 1.0 * jnp.eye(self.dim, dtype=self.dtype)

    def g(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)

    def Sigma(self, t: tTYPE, x: xTYPE):
        return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class FitzHughNagumoProcess(ContinuousTimeProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
        
    def f(self, t: tTYPE, x: xTYPE, **kwargs) -> xTYPE:
        term1 = jnp.array([[1. / self.params["chi"], -1. / self.params["chi"]],
                           [self.params["gamma"],        -1.]], 
                            dtype=self.dtype)     # (2, 2)
        term2 = jnp.array([(- x[0]**3 + self.params["s"]) / self.params["chi"], self.params["alpha"]], dtype=self.dtype)     # (2,)
        return jnp.einsum("i j, j -> i", term1, x) + term2  # (2,)
    
    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return jnp.array([0., self.params["sigma"]], dtype=self.dtype).reshape(2, 1)     # (2, 1)
        
class FitzHughNagumoAuxProcess(AuxiliaryProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
        
    def B(self, t: tTYPE) -> xTYPE:
        return jnp.array([[(1. - 3. * self.params["v"]**2) / self.params["chi"], -1. / self.params["chi"]],
                          [self.params["gamma"],                                         -1.]], 
                          dtype=self.dtype)     # (2, 2)
        
    def beta(self, t: tTYPE) -> xTYPE:
        return jnp.array([(2. * self.params["v"]**3 + self.params["s"]) / self.params["chi"],
                          self.params["alpha"]], 
                          dtype=self.dtype)     # (2,)  
        
    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        return jnp.array([0., self.params["sigma"]], dtype=self.dtype).reshape(2, 1)

class LandmarkLagrangianProcess(ContinuousTimeProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
    
    def f(self, t: tTYPE, x: xTYPE, **kwargs) -> xTYPE:
        return jnp.zeros((self.dim, ), dtype=self.dtype)
    
    def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
        x_landmarks = rearrange(x, "(n d) -> n d", n=self.params["n_landmarks"])
        kernel_fn = lambda dist_mat: 0.5 * self.params["k_alpha"] * jnp.exp(-jnp.sum(dist_mat**2, axis=-1) / (2.0 * self.params["k_sigma"]**2))
        dist_mat = x_landmarks[:, None, :] - x_landmarks[None, :, :]
        corr_mat = kernel_fn(dist_mat)
        q_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(self.params["m_landmarks"], dtype=self.dtype))
        q_half = rearrange(q_half, "i k j l -> (i k) (j l)")
        return q_half

class LandmarkLagrangianAuxProcess(AuxiliaryProcess):
    
    def __init__(self, dim: int, params: Dict[str, float]):
        super().__init__(dim)
        self.params = params
        self.const_g = None
        
    def initialize_g(self, v: xTYPE) -> None:
        v_landmarks = rearrange(v, "(n d) -> n d", n=self.params["n_landmarks"])
        kernel_fn = lambda dist_mat: 0.5 * self.params["k_alpha"] * jnp.exp(-jnp.sum(dist_mat**2, axis=-1) / (2.0 * self.params["k_sigma"]**2))
        dist_mat = v_landmarks[:, None, :] - v_landmarks[None, :, :]
        corr_mat = kernel_fn(dist_mat)
        q_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(self.params["m_landmarks"], dtype=self.dtype))
        q_half = rearrange(q_half, "i k j l -> (i k) (j l)")
        self.const_g = q_half
        
    def beta(self, t: tTYPE):
        return jnp.zeros((self.dim, ), dtype=self.dtype)
    
    def B(self, t: tTYPE):
        return jnp.zeros((self.dim, self.dim), dtype=self.dtype)
    
    def g(self, t: tTYPE, x: xTYPE):
        assert self.const_g is not None, "constant g must be initialized using .initialize_g(v)"
        return self.const_g


# class DoubleWellPotentialDiffusionProcess(ContinuousTimeProcess):
    
#     def __init__(self, dim: int, params: Dict[str, float]):
#         super().__init__(dim)
#         self.params = params
    
#     def f(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return x - x**3
    
#     def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
#     def Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
#     def inv_Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return 1.0 / self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)


# class DoubleWellPotentialDiffusionAuxProcess(AuxiliaryProcess):
    
#     def __init__(self, dim: int, params: Dict[str, float]):
#         super().__init__(dim)
#         self.params = params
        
#     def beta(self, t: tTYPE) -> xTYPE:
#         return jnp.zeros((self.dim, ), dtype=self.dtype)
    
#     def B(self, t: tTYPE) -> xTYPE:
#         return - 1.0 * jnp.eye(self.dim, dtype=self.dtype)
    
#     def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
# class DoubleWellPotentialDiffusionAuxProcessBetter(AuxiliaryProcess):
    
#     def __init__(self, dim: int, params: Dict[str, Any]):
#         super().__init__(dim)
#         self.params = params

#     def beta(self, t: tTYPE) -> xTYPE:
#         return 2.0 * self.params["v"] * jnp.ones((self.dim, ), dtype=self.dtype)
    
#     def B(self, t: tTYPE) -> xTYPE:
#         return (1.0 - 3.0 * self.params["v"]**2) * jnp.eye(self.dim, dtype=self.dtype)
    
#     def g(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return self.params["sigma"] * jnp.eye(self.dim, dtype=self.dtype)
    
#     def Sigma(self, t: tTYPE, x: xTYPE) -> xTYPE:
#         return self.params["sigma"]**2 * jnp.eye(self.dim, dtype=self.dtype)
    
class SDEFactory:
    X_unc_cls: ContinuousTimeProcess
    X_aux_cls: AuxiliaryProcess
    
    def __init__(self, config: OmegaConf):
        if "brownian" in config.sde.name.lower():
            self.X_unc_cls = BrownianProcess
            self.X_aux_cls = BrownianAuxProcess
        elif "ou" in config.sde.name.lower():
            self.X_unc_cls = OrnsteinUhlenbeckProcess
            self.X_aux_cls = OrnsteinUhlenbeckAuxProcess
        elif "cell" in config.sde.name.lower():
            self.X_unc_cls = CellDiffusionProcess
            self.X_aux_cls = CellDiffusionAuxProcess
        elif "fhn" in config.sde.name.lower():
            self.X_unc_cls = FitzHughNagumoProcess
            self.X_aux_cls = FitzHughNagumoAuxProcess
        elif "landmark" in config.sde.name.lower():
            self.X_unc_cls = LandmarkLagrangianProcess
            self.X_aux_cls = LandmarkLagrangianAuxProcess
        else:
            raise ValueError(f"Model {config.sde.name} not supported")
        
        self.config = config
        
    def get_original_sde(self) -> ContinuousTimeProcess:
        return self.X_unc_cls(
            dim=self.config.sde.X_dim,
            params=self.config.sde.params_X_unc
        )
    
    def get_auxiliary_sde(self) -> AuxiliaryProcess:
        return self.X_aux_cls(
            dim=self.config.sde.X_dim,
            params=self.config.sde.params_X_aux
        )
        
    def get_sde(self) -> Tuple[ContinuousTimeProcess, AuxiliaryProcess]:
        return self.get_original_sde(), self.get_auxiliary_sde()