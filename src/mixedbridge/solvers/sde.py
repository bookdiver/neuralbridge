from __future__ import annotations
from functools import partial
import abc
import jax
import jax.numpy as jnp

from ..stochastic_processes.bases import ContinuousTimeProcess
from ..stochastic_processes.wiener import WienerProcess
from ..utils.sample_path import SamplePath

class SDESolver(abc.ABC):
    sde: ContinuousTimeProcess
    wiener_process: WienerProcess

    def __init__(
        self, sde: ContinuousTimeProcess, wiener_process: WienerProcess
    ):
        self.sde = sde
        self.wiener_process = wiener_process
        self._dts = jnp.diff(self.sde.ts)  # Cache dts

    @property
    def ts(self):
        return self.sde.ts

    @property
    def dts(self):
        return self._dts

    @abc.abstractmethod
    def step(
        self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray, *args, **kwargs
    ):
        pass

    def solve(
        self, x0: jnp.ndarray, rng_key: jax.Array, dWs: jnp.ndarray | None, *, log_likelihood: bool = False, n_batches: int = 1
    ) -> SamplePath:
        
        def scan_fn(carry: tuple, val: tuple) -> tuple:
            x, log_psi = carry
            t, dt, dW = val
            x_next = self.step(x, t, dt, dW)
            G = self.sde.G(t, x) if log_likelihood else 0.0
            log_psi += G * dt
            return (x_next, log_psi), x_next
        
        if dWs is None:
            assert rng_key is not None, "Either dWs or rng_key must be provided"
            dWs = self.wiener_process.sample_path(
                rng_key,
                ts=self.ts,
                n_batches=n_batches
            ).xs
        else:
            assert dWs.shape[0] == n_batches, "Number of batches must match the shape of dWs"

        (_, log_ll), xs = jax.vmap(
            lambda dW: jax.lax.scan(
                scan_fn, init=(x0, 0.0), xs=(self.ts[:-1], self.dts, dW)
            ),
            in_axes=0
        )(dWs)

        return SamplePath(xs=xs, ts=self.ts[1:], dWs=dWs, log_likelihood=log_ll)


class Euler(SDESolver):
    @partial(jax.jit, static_argnums=(0,))
    def step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray):
        return x + self.sde.f(t, x) * dt + self.sde.g(t, x) @ dW
