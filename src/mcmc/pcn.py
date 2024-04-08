import jax
import jax.numpy as jnp

from ..stochastic_processes.abstract_processes.guided_bridge_process import GuidedBridgeProcess
from ..solvers.wiener_process import WienerProcess
from ..solvers.euler import Euler
from ..solvers.sample_path import SamplePath

class PreconditionedCrankNicolson:
    def __init__(self, 
                 rho: float,
                 x0: jnp.ndarray,
                 guided_process: GuidedBridgeProcess,
                 rng_key: jax.Array):
        self.rho = rho
        self.x0 = x0

        rng_key1 , rng_key2 = jax.random.split(rng_key)
        self.W = WienerProcess(dim=guided_process.dim, dtype=guided_process.dtype, rng_key=rng_key1)
        self.Z = WienerProcess(dim=guided_process.dim, dtype=guided_process.dtype, rng_key=rng_key2)

        self.ts = guided_process.ts
        self.sampler = Euler(guided_process)
        self.log_likelihood = guided_process.log_likelihood

    def update_step(self, path_X: SamplePath, Ws: jnp.ndarray):
        Zs = self.Z.sample_path(self.ts).xs
        Wos = self.rho * Ws + jnp.sqrt(1 - self.rho**2) * Zs

        path_Xo = self.sampler.solve(self.x0, dWs=Wos)
        ll_X = self.log_likelihood(path_X)
        ll_Xo = self.log_likelihood(path_Xo)
        log_ratio = ll_Xo - ll_X

        log_u = jnp.log(jax.random.uniform(self.W.rng_key))

        self.W.update_key()
        self.Z.update_key()

        accepted = log_u < log_ratio

        if accepted:
            return path_Xo, Wos, ll_X, accepted
        else:
            return path_X, Ws, ll_X, accepted
        
    def update(self, n_iters: int, verbose: bool=True, log_every: int=1):
        Ws = self.W.sample_path(self.ts).xs
        path_X = self.sampler.solve(self.x0, dWs=Ws)
        ll_Xs = []
        accepts = 0

        for i in range(n_iters):
            path_X, Ws, ll_X, accepted = self.update_step(path_X, Ws)
            if verbose and (i + 1) % log_every == 0:
                print(f"Iteration {i+1}/{n_iters}: log likelihood = {ll_X}, accepted = {accepted}")

            if accepted:
                accepts += 1
            ll_Xs.append(ll_X)

        print(f"Acceptance rate: {accepts / n_iters:.2%}")
        return path_X, ll_Xs


