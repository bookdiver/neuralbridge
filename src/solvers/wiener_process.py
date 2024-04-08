import jax
import jax.numpy as jnp

from .sample_path import SamplePath

class WienerProcess:
    def __init__(self, dim: int, dtype: jnp.dtype, rng_key: jax.Array):
        self.dim = dim
        self.dtype = dtype
        self.rng_key = rng_key

    def update_key(self):
        self.rng_key = jax.random.split(self.rng_key)[0]

    def sample(self, t0: float, t1: float, rng_key: jax.Array) -> jnp.ndarray:
        dt = t1 - t0
        dW = jax.random.normal(rng_key, (self.dim,), dtype=self.dtype) * jnp.sqrt(dt)
        return dW
    
    def sample_path(self, ts: jnp.ndarray) -> SamplePath:
        n_steps = len(ts) - 1
        rng_keys = jax.random.split(self.rng_key, n_steps)
        t_now, t_next = ts[:-1], ts[1:]
        dWs = jax.vmap(self.sample)(t_now, t_next, rng_keys)
        return SamplePath(ts=ts, xs=dWs, name=f"{self.dim}-dimensional Wiener process")
