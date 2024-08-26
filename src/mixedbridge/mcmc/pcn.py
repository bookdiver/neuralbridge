import jax
import jax.numpy as jnp
from functools import partial

from ..stochastic_processes.gbp import GuidedBridgeProcess
from ..stochastic_processes.wiener import WienerProcess
from ..solvers.sde import Euler
from ..utils.sample_path import SamplePath

class PreconditionedCrankNicolson:
    def __init__(self, 
                 rho: float,
                 x0: jnp.ndarray,
                 guided_process: GuidedBridgeProcess,
                 *,
                 rng_key: jax.Array = jax.random.PRNGKey(42),
                 n_batches: int = 1
                ):
        self.rho = rho
        self.x0 = x0
        self.n_batches = n_batches
        
        self.dim = guided_process.dim
        self.dtype = guided_process.dtype
        self.T = guided_process.T
        self.dt = guided_process.dt
        self.ts = guided_process.ts

        self.wiener = WienerProcess(self.dim, self.dtype, self.T, self.dt)
        self.rng_key = rng_key
        key1, key2 = jax.random.split(rng_key)
        self.W_sampler = self.wiener.path_sampler(key1, n_batches=self.n_batches)
        self.Z_sampler = self.wiener.path_sampler(key2, n_batches=self.n_batches)

        self.solver = Euler(guided_process, self.wiener)

    @partial(jax.jit, static_argnums=(0,))
    def update_step(self, rng_key, path_X, ll_X, Ws):
        key1, key2 = jax.random.split(rng_key, 2)
        Zs = self.wiener.sample_path(key1, ts=self.ts, n_batches=self.n_batches).xs
        Wos = self.rho * Ws + jnp.sqrt(1 - self.rho**2) * Zs

        path_Xo = self.solver.solve(self.x0, rng_key=None, dWs=Wos, log_likelihood=True, n_batches=self.n_batches)
        ll_Xo = jnp.mean(path_Xo.log_likelihood)
        log_ratio = ll_Xo - ll_X

        log_u = jnp.log(jax.random.uniform(key2))
        accepted = log_u < log_ratio

        return jax.lax.cond(
            accepted,
            lambda: (path_Xo, Wos, ll_Xo, accepted),
            lambda: (path_X, Ws, ll_X, accepted)
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def update_loop(self, n_iters, init_state):
        def body_fun(i, state):
            path_X, Ws, ll_X, accepts, ll_Xs, rng_key = state
            rng_key, subkey = jax.random.split(rng_key)
            path_X, Ws, ll_X, accepted = self.update_step(subkey, path_X, ll_X, Ws)
            accepts += accepted
            ll_Xs = ll_Xs.at[i+1].set(ll_X)
            return path_X, Ws, ll_X, accepts, ll_Xs, rng_key

        return jax.lax.fori_loop(0, n_iters, body_fun, init_state)

    def update(self, n_iters: int, verbose: bool=True, log_every: int=1):
        key1, key2 = jax.random.split(self.rng_key)
        Ws = self.wiener.sample_path(key1, ts=self.ts, n_batches=self.n_batches).xs
        path_X = self.solver.solve(self.x0, rng_key=None, dWs=Ws, log_likelihood=True, n_batches=self.n_batches)
        ll_X = jnp.mean(path_X.log_likelihood)
        ll_Xs = jnp.zeros(n_iters + 1)
        ll_Xs = ll_Xs.at[0].set(ll_X)

        init_state = (path_X, Ws, ll_X, 0, ll_Xs, self.rng_key)
        path_X, Ws, ll_X, accepts, ll_Xs, self.rng_key = self.update_loop(n_iters, init_state)

        if verbose:
            for i in range(0, n_iters, log_every):
                print(f"Iteration {i+1}/{n_iters}: log likelihood = {ll_Xs[i+1]}")

        print(f"Acceptance rate: {accepts / n_iters:.2%}")
        return path_X, ll_Xs