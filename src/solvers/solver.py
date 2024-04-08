from collections import namedtuple
import abc

import jax
import jax.numpy as jnp

from ..stochastic_processes.abstract_processes.continuous_time_process import ContinuousTimeProcess
from ..stochastic_processes.abstract_processes.wiener_process import WienerProcess
from ..utils.sample_path import SamplePath

class SDESolver(abc.ABC):
    sde: ContinuousTimeProcess
    wiener_process: WienerProcess = None

    def __init__(self, 
                 sde: ContinuousTimeProcess, 
                 wiener_process: WienerProcess = None):
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

    def solve(self, x0: jnp.ndarray, dWs: jnp.ndarray = None, log_likelihood: bool=False) -> SamplePath:
        if (dWs is None and self.wiener_process is not None):
            dWs = self.wiener_process.sample_path(self.ts).xs
        else:
            assert dWs is not None
        
        def scan_fn(carry: tuple, val: tuple) -> tuple:
            x, log_psi = carry
            t, t_next, dt, dW = val
            x_next = self.step(x, t, dt, dW)
            G = self.sde.G(t, x) if log_likelihood else 0.0
            log_psi += G * (t_next - t)
            return (x_next, log_psi), x_next

        (_, log_ll), xs = jax.lax.scan(scan_fn, (x0, 0.0), (self.ts[:-1], self.ts[1:], self.dts, dWs))

        return SamplePath(xs=xs, ts=self.ts[1:], log_likelihood=log_ll)

        

        

