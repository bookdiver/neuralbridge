from __future__ import annotations
import abc
import jax
import jax.numpy as jnp

from ..stochastic_processes.bases import ContinuousTimeProcess
from ..stochastic_processes.wiener import WienerProcess
from ..utils.sample_path import SamplePath

class SDESolver(abc.ABC):
    sde: ContinuousTimeProcess
    wiener_process: WienerProcess = None

    def __init__(
        self, sde: ContinuousTimeProcess, wiener_process: WienerProcess = None
    ):
        self.sde = sde
        self.wiener_process = wiener_process

    @property
    def ts(self):
        return self.sde.ts

    @property
    def dts(self):
        return jnp.diff(self.ts)

    @abc.abstractmethod
    def step(
        self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray, *args, **kwargs
    ):
        pass

    def solve(
        self, x0: jnp.ndarray, dWs: jnp.ndarray = None, log_likelihood: bool = False, path: SamplePath = None,
    ) -> SamplePath:
        if dWs is None and self.wiener_process is not None:
            dWs = self.wiener_process.sample_path(self.ts).xs
        else:
            assert dWs is not None

        def scan_fn(carry: tuple, val: tuple) -> tuple:
            x, log_psi = carry
            t, dt, dW = val
            x_next = self.step(x, t, dt, dW)
            G = self.sde.G(t, x) if log_likelihood else 0.0
            log_psi += G * dt
            return (x_next, log_psi), x_next

        (_, log_ll), xs = jax.lax.scan(
            scan_fn, init=(x0, 0.0), xs=(self.ts[:-1], self.dts, dWs)
        )

        if path:
            path.xs = xs
            path.ts = self.ts[1:]
            path.log_likelihood = log_ll
            return path
        else:
            return SamplePath(xs=xs, ts=self.ts[1:], log_likelihood=log_ll)


class Euler(SDESolver):
    def step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray):
        return x + self.sde.f(t, x) * dt + self.sde.g(t, x) @ dW
