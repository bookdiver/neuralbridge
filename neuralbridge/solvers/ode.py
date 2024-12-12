from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import AuxiliaryProcess

SolverState = namedtuple("SolverState", ["L", "M_diag", "mu"])

class BackwardODE:
    def __init__(
        self,
        X_tilde: AuxiliaryProcess,
        L0: jnp.ndarray,
        Sigma0: jnp.ndarray,
        mu0: jnp.ndarray,
        kernel: str = "dopri5"
    ):
        self.Sigma = partial(X_tilde.Sigma, x=None)
        self.B = X_tilde.B
        self.beta = X_tilde.beta
        self.L0 = L0
        self.Sigma0 = Sigma0
        self.mu0 = mu0
        self.kernel = self._get_kernel(kernel)
    
    def _get_kernel(self, kernel: str):
        if kernel == "dopri5":
            return kernel_dopri5
        elif kernel == "r3":
            return kernel_r3
        else:
            raise ValueError(f"Kernel {kernel} not found")

    @partial(jax.jit, static_argnums=(0,))
    def solve_step(self, solver_state: namedtuple, t_args: tuple) -> namedtuple:
        t, dt = t_args
        L = self.kernel(
            func=lambda t, y: - jnp.einsum("i j, j ... -> i ...", y, self.B(t)),
            t=t,
            y=solver_state.L,
            dt=dt
        )
        M_diag = self.kernel(
            func=lambda t, y: - jnp.einsum("i j, j k, l k -> i l", solver_state.L, self.Sigma(t), solver_state.L),
            t=t,
            y=solver_state.M_diag,
            dt=dt
        )
        mu = self.kernel(
            func=lambda t, y: - jnp.einsum("i j, j ... -> i ...", solver_state.L, self.beta(t)),
            t=t,
            y=solver_state.mu,
            dt=dt
        )
        M = jnp.linalg.inv(M_diag)
        new_state = SolverState(L=L, M_diag=M_diag, mu=mu)
        return new_state, (L, M, mu)
    
    def solve(self, ts: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dts = jnp.diff(ts)
        init_state = SolverState(L=self.L0, M_diag=self.Sigma0, mu=self.mu0)
        _, (Ls, Ms, mus) = jax.lax.scan(
            f=self.solve_step, 
            init=init_state, 
            xs=(ts[1:], dts),
            reverse=True
        )
        return Ls, Ms, mus


def kernel_r3(
    func: callable, t: float, y: jnp.ndarray, dt: float, **kwargs
) -> jnp.ndarray:
    k1 = func(t, y, **kwargs)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1, **kwargs)
    k3 = func(t + 0.75 * dt, y + 3 * dt / 4 * k2, **kwargs)
    out = y + dt * (2 / 9 * k1 + 1 / 3 * k2 + 4 / 9 * k3)
    return out

def kernel_dopri5(
    func: callable, t: float, y: jnp.ndarray, dt: float, **kwargs
) -> jnp.ndarray:
    k1 = func(t, y, **kwargs)
    k2 = func(t + 1/5 * dt, y + 1/5 * dt * k1, **kwargs)
    k3 = func(t + 3/10 * dt, y + 3/40 * dt * k1 + 9/40 * dt * k2, **kwargs)
    k4 = func(t + 4/5 * dt, y + 44/45 * dt * k1 - 56/15 * dt * k2 + 32/9 * dt * k3, **kwargs)
    k5 = func(t + 8/9 * dt, y + 19372/6561 * dt * k1 - 25360/2187 * dt * k2 + 64448/6561 * dt * k3 - 212/729 * dt * k4, **kwargs)
    k6 = func(t + dt, y + 9017/3168 * dt * k1 - 355/33 * dt * k2 + 46732/5247 * dt * k3 + 49/176 * dt * k4 - 5103/18656 * dt * k5, **kwargs)
    k7 = func(t + dt, y + 35/384 * dt * k1 + 0 * dt * k2 + 500/1113 * dt * k3 + 125/192 * dt * k4 - 2187/6784 * dt * k5 + 11/84 * dt * k6, **kwargs)
    
    out = y + dt * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
    return out
