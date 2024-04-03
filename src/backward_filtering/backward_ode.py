from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from ..stochastic_processes.continuous_time_process import ContinuousTimeProcess
from ..stochastic_processes.auxiliary_process import AuxiliaryProcess


SolverState = namedtuple('SolverState', ['L', 'M_diag', 'mu'])

class BackwardODE:
    def __init__(
            self, 
            original_process: ContinuousTimeProcess, 
            auxiliary_process: AuxiliaryProcess, 
            L0: jnp.ndarray, 
            target: jnp.ndarray, 
            Sigma: jnp.ndarray):
        
        self.g = partial(original_process.g, x=None)
        self.B = auxiliary_process.B
        self.beta = auxiliary_process.beta
        self.L0 = L0
        self.Sigma = Sigma
    
    def solve_step(self, solver_state: namedtuple, t: float | jnp.ndarray, dt: float) -> namedtuple:
        L = kernel_r3(lambda t, x: -self.B(t) @ x, t, solver_state.L, dt)
        M_diag = kernel_r3(lambda t, x: L @ self.g(t) @ self.g(t).T @ L @ self.B(t), t, solver_state.M_diag, dt)
        M = jnp.linalg.inv(M_diag)
        mu = kernel_r3(lambda t, x: -L @ self.beta(t), t, solver_state.mu, dt)

        new_state = SolverState(L=L, M_diag=M_diag, mu=mu)
        return new_state, (L, M, mu)
    
    def solve(self, reverse_ts: jnp.ndarray):
        init_state = SolverState(
            L=self.L0, 
            M_diag=self.Sigma,
            mu=0.0*self.L0[:, 0]
        )
        dt = reverse_ts[0] - reverse_ts[1]
        solve_step = jax.jit(partial(self.solve_step, dt=dt))
        _, (Ls, Ms, mus) = jax.lax.scan(solve_step, init_state, reverse_ts)

        return Ls, Ms, mus

def kernel_r3(func: callable, t: float | jnp.ndarray, x: jnp.ndarray, dt: float, **kwargs) -> jnp.ndarray:
    k1 = func(t, x, **kwargs)
    k2 = func(t + 0.5 * dt, x + 0.5 * dt * k1, **kwargs)
    k3 = func(t + 0.75 * dt, x + 3 * dt / 4 * k2, **kwargs)
    out = x + dt * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)
    return out
