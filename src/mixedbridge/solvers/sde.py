from typing import Optional, Generator
from functools import partial
import abc
import jax
import jax.numpy as jnp

from ..utils.sample_path import SamplePath
from ..stochastic_processes.bases import ContinuousTimeProcess

class WienerProcess:

    def __init__(self,
                 T: float, 
                 dt: float,
                 dim: int,
                 dtype: Optional[jnp.dtype] = jnp.float32,
                 t_scheme: str = "linear"):
        self.T = T
        self.dt = dt
        self.dim = dim
        self.dtype = dtype
        self.t_scheme = t_scheme
        
    @property
    def ts(self):
        if self.t_scheme == "linear":
            return self.linear_t_scheme()
        elif self.t_scheme == "quadratic":
            return self.quadratic_t_scheme()
        else:
            raise ValueError(f"Time scheme: {self.t_scheme} not found!")
        
    @property
    def dts(self):
        return jnp.diff(self.ts)
    
    def linear_t_scheme(self):
        ts = jnp.arange(0.0, self.T + self.dt, self.dt, dtype=self.dtype)
        return ts
    
    def quadratic_t_scheme(self):
        ts = jnp.arange(0.0, self.T + self.dt, self.dt, dtype=self.dtype)
        ts =  ts * (2.0 - ts / self.T)
        return ts

    def sample_step(self, rng_key: jax.Array, dt: float) -> jnp.ndarray:
        dW = jax.random.normal(rng_key, (self.dim,), dtype=self.dtype) * jnp.sqrt(dt)
        return dW
    
    def sample_path(self, rng_key: jax.Array, batch_size: int = 1) -> SamplePath:
        n_steps = len(self.dts)
        
        sub_keys = jax.random.split(rng_key, n_steps * batch_size).reshape((batch_size, n_steps, -1))
        dWs = jax.vmap(
            jax.vmap(
                self.sample_step,
                in_axes=(0, 0)
            ),
            in_axes=(0, None)
        )(sub_keys, self.dts)
        
        Ws = jnp.cumsum(dWs, axis=1)
        Ws = jnp.concatenate([jnp.zeros((batch_size, 1, self.dim)), Ws], axis=1)
        
        return SamplePath(
            ts=self.ts, 
            dts=self.dts,
            xs=Ws,
            dxs=dWs,
            name=f"{self.dim}-dimensional Wiener process"
        )
    
    def path_sampler(self, rng_key: jax.Array, batch_size: int = 1) -> Generator[SamplePath, None, None]:
            
        while True:
            rng_key, sub_key = jax.random.split(rng_key)
            yield self.sample_path(sub_key, batch_size)

class SDESolver(abc.ABC):
    sde: ContinuousTimeProcess
    wiener: WienerProcess

    def __init__(
        self, 
        sde: ContinuousTimeProcess, 
        wiener: WienerProcess
    ):
        self.sde = sde
        self.wiener = wiener
        
    @property
    def T(self):
        return self.wiener.T
    
    @property
    def ts(self):
        return self.wiener.ts

    @property
    def dts(self):
        return self.wiener.dts

    @abc.abstractmethod
    def step(
        self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray, *args, **kwargs
    ):
        pass

    def solve(
        self, 
        x0: jnp.ndarray, 
        rng_key: Optional[jax.Array] = None, 
        dWs: Optional[jnp.ndarray] = None, 
        batch_size: int = 1,
        enforce_end_point: jnp.ndarray = None
    ) -> SamplePath:
        
        if hasattr(self.sde, "G"):
            log_likelihood = True
        else:
            log_likelihood = False
        
        def scan_fn(carry: tuple, val: tuple) -> tuple:
            x, log_ll = carry
            t, dt, dW = val
            x_next = self.step(x, t, dt, dW)
            G = self.sde.G(t, x) if log_likelihood else 0.0
            log_ll += G * dt
            return (x_next, log_ll), (x_next, G)
        
        if dWs is None:
            assert rng_key is not None, "Either dWs or rng_key must be provided"
            wiener_path = self.wiener.sample_path(
                rng_key,
                batch_size=batch_size
            )
            dWs = wiener_path.dxs
        else:
            assert dWs.shape[0] == batch_size, "Number of batches must match the shape of dWs"

        (_, final_log_ll), (xs, G_integrands) = jax.vmap(
            lambda dW: jax.lax.scan(
                scan_fn, 
                init=(x0, 0.0), 
                xs=(self.ts[:-1], self.dts, dW)
            ),
            in_axes=0
        )(dWs)
        x0_expanded = jnp.expand_dims(x0, axis=(0, 1))  
        x0_tiled = jnp.tile(x0_expanded, (batch_size, 1, 1)) 
        xs = jnp.concatenate([x0_tiled, xs], axis=1) 
        
        if enforce_end_point is not None:
            xs = xs.at[:, -1, :].set(enforce_end_point[None, ...])
            
        return SamplePath(
            xs=xs, 
            ts=self.ts, 
            dWs=dWs,
            dts=self.dts,
            log_likelihood=final_log_ll,
            dlog_ll=G_integrands
        )


class Euler(SDESolver):
    @partial(jax.jit, static_argnums=(0,))
    def step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray) -> jnp.ndarray:
        drift     = self.sde.f(t, x)
        diffusion = self.sde.g(t, x)
        return x + drift * dt + diffusion @ dW
    
class ModifiedEuler(SDESolver):
    @partial(jax.jit, static_argnums=(0,))
    def step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray) -> jnp.ndarray:
        drift = self.sde.f(t, x)
        next_t = t + dt
        scaling = jnp.sqrt((self.T - next_t) / (self.T - t))
        diffusion = scaling * self.sde.g(t, x)
        return x + drift * dt + diffusion @ dW

class RungeKutta(SDESolver):
    @partial(jax.jit, static_argnums=(0,))
    def step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Runge-Kutta method not implemented")