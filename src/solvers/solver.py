from collections import namedtuple
import abc

import jax
import jax.numpy as jnp

from ..stochastic_processes.abstract_processes.continuous_time_process import ContinuousTimeProcess
from .wiener_process import WienerProcess
from .sample_path import SamplePath

class SDESolver(abc.ABC):
    sde: ContinuousTimeProcess
    wiener_process: WienerProcess = None

    def __init__(self, 
                 sde: ContinuousTimeProcess, 
                 wiener_process: WienerProcess = None):
        assert sde.dim == wiener_process.dim if wiener_process is not None else True
        self.sde = sde
        self.wiener_process = wiener_process

    @property
    def ts(self):
        return self.sde.ts
    
    @property
    def dts(self):
        return jnp.diff(self.ts)

    @abc.abstractmethod
    def step(self, 
             x: jnp.ndarray, 
             t: float, 
             dt: float,
             dW: jnp.ndarray,
             *args, 
             **kwargs):
        pass

    def solve(self, x0: jnp.ndarray, dWs: jnp.ndarray = None) -> SamplePath:
        if (dWs is None and self.wiener_process is not None):
            dWs = self.wiener_process.sample_path(self.ts).xs
        else:
            assert dWs is not None
        
        def scan_fn(x: jnp.ndarray, c: tuple):
            t, dt, dW = c
            x_next = self.step(x, t, dt, dW)
            return x_next, x_next

        init_c = (self.ts[:-1], self.dts, dWs)

        _, xs = jax.lax.scan(scan_fn, x0, init_c)

        return SamplePath(xs=xs, ts=self.ts[1:])

        

        

