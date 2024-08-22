import jax
import jax.numpy as jnp

from .bases import ContinuousTimeProcess
from ..utils.sample_path import SamplePath

class WienerProcess(ContinuousTimeProcess):

    def __init__(self, 
                 dim: int,
                 dtype: jnp.dtype = jnp.float32,
                 T: float = 1.0,
                 dt: float = None,
                 *,
                 rng_key: jax.Array = jax.random.PRNGKey(0)):
        self.dim = dim
        self.dtype = dtype
        self.T = T
        self.dt = dt

        self.rng_key = rng_key

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x, dtype=self.dtype)
    
    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(self.dim, dtype=self.dtype)
    
    def Sigma(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(self.dim, dtype=self.dtype)
    
    def update_key(self):
        self.rng_key = jax.random.split(self.rng_key)[0]

    def sample_step(self, dt: float, rng_key: jax.Array) -> jnp.ndarray:
        dW = jax.random.normal(rng_key, (self.dim,), dtype=self.dtype) * jnp.sqrt(dt)
        return dW
    
    def sample_path(self, evaluate_ts: jnp.ndarray) -> jnp.ndarray:
        if evaluate_ts is None and self.ts is not None:
            ts = self.ts
        elif evaluate_ts is not None:
            ts = evaluate_ts
        else:
            raise ValueError("Either provide ts or evaluate_ts to sample path")
        dts = jnp.diff(ts)
        n_steps = len(dts)
        rng_keys = jax.random.split(self.rng_key, n_steps)
        dWs = jax.vmap(self.sample_step)(dts, rng_keys)
        return SamplePath(ts=ts, xs=dWs, name=f"{self.dim}-dimensional Wiener process")