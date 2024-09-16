import abc

import jax.numpy as jnp

class ContinuousTimeProcess(abc.ABC):
    """
    Abstract base class for continuous-time stochastic processes defined by SDEs:
    dX(t) = f(t, X(t))dt + g(t, X(t))dW(t)

    This class provides the structure for stochastic processes used in modeling.
    It defines abstract methods for key components of the process and implements
    some concrete methods based on these abstractions.

    Attributes:
        dim (int): The dimension of the process.
        dtype (jnp.dtype): The data type used for numerical computations.
        T (float): The total time span of the process.
        dt (float): The time step size.
    """

    dim: int
    dtype: jnp.dtype
    T: float
    dt: float
    
    def __init__(self, T: float, dt: float, dim: int, dtype: jnp.dtype):
        """
        Initialize the ContinuousTimeProcess.

        Args:
            T (float): The total time span of the process.
            dt (float): The time step size.
            dim (int): The dimension of the process.
            dtype (jnp.dtype): The data type for numerical computations.
        """
        self.T = T
        self.dt = dt
        self.dim = dim
        self.dtype = dtype

    @property
    def ts(self):
        """
        Returns the time grid.

        Returns:
            jnp.ndarray: An array representing the time grid.
        """
        assert self.dt is not None, "Time step size must be provided!"
        return jnp.arange(0, self.T + self.dt, self.dt, dtype=self.dtype)
    
    @property
    def reverse_ts(self):
        """
        Returns the reversed time grid.

        Returns:
            jnp.ndarray: An array representing the reversed time grid.
        """
        return jnp.flip(self.ts) 
    
    @property
    def dts(self):
        """
        Returns the time step differences.

        Returns:
            jnp.ndarray: An array of time step differences.
        """
        return jnp.diff(self.ts) 
    
    @abc.abstractmethod
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the drift function.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The drift value at time t and state x.
        """
        pass
    
    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the diffusion function.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The diffusion value at time t and state x.
        """
        pass

    @abc.abstractmethod
    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the covariance matrix.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The covariance matrix at time t and state x.
        """
        pass
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Returns the inverse covariance matrix.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The inverse of the covariance matrix at time t and state x.
        """
        return jnp.linalg.inv(self.Sigma(t, x, *args, **kwargs))
    
class AuxiliaryProcess(ContinuousTimeProcess):
    """
    Represents an auxiliary process in continuous time, inheriting from ContinuousTimeProcess.
    Auxiliary process is defined by the following SDE:
    dX_tilde(t) = (beta(t) + B(t)*X_tilde)dt + g(t)dW(t)
    
    This class defines the structure for auxiliary processes used in stochastic modeling.
    It provides abstract methods for key components of the process and implements
    some concrete methods based on these abstractions.

    Attributes:
        dim (int): The dimension of the process.
        dtype (jnp.dtype): The data type used for numerical computations.
        T (float): The total time span of the process.
        dt (float): The time step size.
    """

    dim: int
    dtype: jnp.dtype
    T: float
    dt: float
    
    def __init__(self, T: float, dt: float, dim: int, dtype: jnp.dtype):
        """
        Initialize the AuxiliaryProcess.

        Args:
            T (float): The total time span of the process.
            dt (float): The time step size.
            dim (int): The dimension of the process.
            dtype (jnp.dtype): The data type for numerical computations.
        """
        super().__init__(T, dt, dim, dtype)

    @abc.abstractmethod
    def beta(self, t: float, *args, **kwargs):
        """
        Abstract method for the beta function of the process.

        Args:
            t (float): The current time.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The beta value at time t.
        """
        pass

    @abc.abstractmethod
    def B(self, t: float, *args, **kwargs):
        """
        Abstract method for the B function of the process.

        Args:
            t (float): The current time.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The B value at time t.
        """
        pass

    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Compute the drift function of the process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            jnp.ndarray: The drift value at time t and state x.
        """
        return self.beta(t, *args, **kwargs) + self.B(t, *args, **kwargs) @ x

    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the diffusion function of the process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The diffusion value at time t and state x.
        """
        pass 

    @abc.abstractmethod
    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the covariance matrix of the process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The covariance matrix at time t and state x.
        """
        pass
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Compute the inverse of the covariance matrix.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            jnp.ndarray: The inverse of the covariance matrix at time t and state x.
        """
        return jnp.linalg.inv(self.Sigma(t, x, *args, **kwargs))

class BridgeProcess(ContinuousTimeProcess):
    """
    Abstract base class for bridge processes.
    Bridge process is defined by the following SDE:
    dX^*(t) = (f(t, X^*(t)) + Sigma(t, X^*(t)) * score(t, X^*(t)))dt + g(t, X^*(t))dW(t)

    A bridge process is a stochastic process conditioned to start at a given initial state
    and end at a specified final state at time T.

    Attributes:
        dim (int): The dimension of the process.
        dtype (jnp.dtype): The data type of the process.
        T (float): The final time of the process.
        dt (float): The time step for discretization.
        xT (jnp.ndarray): The final state of the process.
    """

    dim: int
    dtype: jnp.dtype
    T: float
    dt: float
    xT: jnp.ndarray
    
    def __init__(self, T: float, dt: float, dim: int, dtype: jnp.dtype, xT: jnp.ndarray):
        """
        Initialize the BridgeProcess.

        Args:
            T (float): The final time of the process.
            dt (float): The time step for discretization.
            dim (int): The dimension of the process.
            dtype (jnp.dtype): The data type of the process.
            xT (jnp.ndarray): The final state of the process.
        """
        super().__init__(T, dt, dim, dtype)
        self.xT = xT
        
    @abc.abstractmethod
    def score(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the score function of the process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The score value at time t and state x.
        """
        pass
    
    @abc.abstractmethod
    def _f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the drift function of the unconditioned process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The drift value of the unconditioned process at time t and state x.
        """
        pass
    
    def f(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Compute the drift function of the bridge process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The drift value of the bridge process at time t and state x.
        """
        return self._f(t, x, *args, **kwargs) + jnp.dot(self.Sigma(t, x, *args, **kwargs), self.score(t, x, *args, **kwargs))
    
    @abc.abstractmethod
    def g(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the diffusion function of the process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The diffusion value at time t and state x.
        """
        pass
    
    @abc.abstractmethod
    def Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Abstract method for the covariance matrix of the process.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The covariance matrix at time t and state x.
        """
        pass
    
    def inv_Sigma(self, t: float, x: jnp.ndarray, *args, **kwargs):
        """
        Compute the inverse of the covariance matrix.

        Args:
            t (float): The current time.
            x (jnp.ndarray): The current state.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            jnp.ndarray: The inverse of the covariance matrix at time t and state x.
        """
        return jnp.linalg.inv(self.Sigma(t, x, *args, **kwargs))
