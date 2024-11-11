from jax_tqdm import loop_tqdm

from neuralbridge.setups import *
from neuralbridge.utils.sample_path import SamplePath
from neuralbridge.stochastic_processes.conds import GuidedBridgeProcess
from neuralbridge.solvers.sde import Euler, WienerProcess

RunState = namedtuple("RunState", ["path_X", "n_accepted", "log_lls","rng_key"])

class PreconditionedCrankNicolson:
    guided_bridge: GuidedBridgeProcess
    run_config: dict
    
    def __init__(self, guided_bridge: GuidedBridgeProcess, run_config: dict):
        self.guided_bridge = guided_bridge

        self.seed = run_config.get("seed", 42)
        self.rho = run_config.get("rho", 0.98)
        self.batch_size = run_config.get("batch_size", 32)
        self.n_iters = run_config.get("n_iters", 10000)
        
        self.wiener = None
        self.path_solver = None

    def initialize_path_solver(self, wiener_proc: WienerProcess) -> None:
        self.wiener = wiener_proc
        self.path_solver = Euler(self.guided_bridge, self.wiener)

    @partial(jax.jit, static_argnums=(0,))
    def run_step(self, run_state: RunState) -> RunState:
        key1, key2, next_key = jax.random.split(run_state.rng_key, 3)
        dZs = self.wiener.sample_path(key1, batch_size=self.batch_size).dxs
        dWos = self.rho * run_state.path_X.dWs + jnp.sqrt(1 - self.rho**2) * dZs

        path_Xo = self.path_solver.solve(
            x0=self.guided_bridge.u, 
            rng_key=None, 
            dWs=dWos, 
            batch_size=self.batch_size
        )
        log_ll_Xo = path_Xo.log_ll
        log_ratio = log_ll_Xo - run_state.path_X.log_ll

        log_u = jnp.log(jax.random.uniform(key2, shape=(self.batch_size,))) 
        accepted = log_u < log_ratio
        n_accepted_new = accepted + run_state.n_accepted

        def update_field(acc: jnp.ndarray, x_new: Union[SamplePath, jnp.ndarray], x_old: Union[SamplePath, jnp.ndarray]) -> Union[SamplePath, jnp.ndarray]:
            if isinstance(x_new, jnp.ndarray) and x_new.shape[0] == accepted.shape[0]:
                return jax.vmap(
                    lambda a, n, o: jax.lax.cond(a, lambda _: n, lambda _: o, None)
                )(acc, x_new, x_old)
            else:
                return jax.lax.cond(acc[0], lambda _: x_new, lambda _: x_old, None)

        path_X_new = jax.tree_map(
            lambda x_new, x_old: update_field(accepted, x_new, x_old),
            path_Xo, run_state.path_X
        )
        new_state = run_state._replace(
            path_X=path_X_new,
            n_accepted=n_accepted_new,
            rng_key=next_key
        )

        return new_state

    @partial(jax.jit, static_argnums=(0, 1))
    def run_loop(self, n_iters: int, init_state: RunState) -> RunState:
        @loop_tqdm(n=n_iters)
        def body_fun(i: int, run_state: RunState):
            run_state = self.run_step(run_state)
            next_log_lls = run_state.log_lls.at[i+1].set(run_state.path_X.log_ll)
            run_state = run_state._replace(log_lls=next_log_lls)
            return run_state

        return jax.lax.fori_loop(0, n_iters, body_fun, init_state)

    def run_pCN(self, log_every: Optional[int] = None) -> RunState:
        assert self.wiener is not None and self.path_solver is not None, "Path solver and Wiener process must be initialized before running pCN, initialize them with .init_path_solver()"
        rng_key = jax.random.PRNGKey(self.seed)
        key1, key2 = jax.random.split(rng_key)
        init_dWs = self.wiener.sample_path(key1, batch_size=self.batch_size).dxs
        init_path_X = self.path_solver.solve(
            x0=self.guided_bridge.u, 
            rng_key=None, 
            dWs=init_dWs, 
            batch_size=self.batch_size
        )
        log_lls = jnp.zeros((self.n_iters + 1, self.batch_size))
        log_lls = log_lls.at[0].set(init_path_X.log_ll)

        init_state = RunState(
            path_X=init_path_X, 
            n_accepted=jnp.zeros(self.batch_size), 
            log_lls=log_lls,
            rng_key=key2
        )
        final_state = self.run_loop(self.n_iters, init_state)

        if log_every is not None:
            for i in range(0, self.n_iters, log_every):
                logging.info(f"Iteration {i}/{self.n_iters}: log likelihood = {final_state.log_lls[i, 0]}")

        logging.info(f"Batch average acceptance rate: {jnp.mean(final_state.n_accepted) / self.n_iters:.2%}")
        return final_state
