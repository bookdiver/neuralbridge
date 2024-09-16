from typing import Generator
import jax
import jax.numpy as jnp

from .bases import ContinuousTimeProcess
from ..utils.sample_path import SamplePath

class WienerProcess(ContinuousTimeProcess):

    def __init__(self, 
                 dim: int,
                 dtype: jnp.dtype = jnp.float32,
                 T: float = 1.0,
                 dt: float = 1e-3
        ):
        self.dim = dim
        self.dtype = dtype
        self.T = T
        self.dt = dt

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass
    
    def Sigma(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass
    
    def inv_Sigma(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        pass

    def sample_step(self, dt: float, rng_key: jax.Array) -> jnp.ndarray:
        dW = jax.random.normal(rng_key, (self.dim,), dtype=self.dtype) * jnp.sqrt(dt)
        return dW
    
    def sample_path(self, rng_key: jax.Array, ts: jnp.ndarray, n_batches: int = 1) -> SamplePath:
        dts = jnp.diff(ts)
        n_steps = len(dts)
        
        subkeys = jax.random.split(rng_key, n_steps * n_batches).reshape((n_batches, n_steps, -1))
        dWs = jax.vmap(
            jax.vmap(
                self.sample_step,
                in_axes=0
            ),
            in_axes=(None, 0)
        )(dts, subkeys)
        
        return SamplePath(
            ts=ts, 
            xs=dWs, 
            name=f"{self.dim}-dimensional Wiener process"
        )
    
    def path_sampler(self, rng_key: jax.Array, evaluate_ts: jnp.ndarray | None = None, *, n_batches: int = 1) -> Generator[SamplePath, None, None]:
        """
        Generates a sample path of the Wiener process, used for the MCMC sampling.

        Args:
            rng_key (jax.Array): Random number generator key.
            evaluate_ts (jnp.ndarray, optional): Time steps at which to evaluate the process. Defaults to None.
            n_batches (int, optional): Number of batches to generate. Defaults to 1.

        Yields:
            Generator[SamplePath, None, None]: A generator yielding sample paths of the Wiener process.
        """
        ts = evaluate_ts if evaluate_ts is not None else self.ts
            
        while True:
            rng_key, subkey = jax.random.split(rng_key)
            yield self.sample_path(subkey, ts, n_batches)