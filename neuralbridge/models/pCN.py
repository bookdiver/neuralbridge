from jax_tqdm import loop_tqdm

from neuralbridge.setups import *
from neuralbridge.utils.sample_path import SamplePath
from neuralbridge.stochastic_processes.conds import GuidedBridgeProcess
from neuralbridge.solvers.sde import Euler, WienerProcess

RunState = namedtuple("RunState", ["path", "total_n_accepted", "log_lls","rng_key"])

class PreconditionedCrankNicolson:
    X_circ: GuidedBridgeProcess
    config: dict
    
    def __init__(self, X_circ: GuidedBridgeProcess, config: dict):
        self.X_circ = X_circ

        self.seed = config.get("seed", DEFAULT_SEED)
        self.rho = config.get("rho", 0.98)
        self.batch_size = config.get("batch_size", 32)
        self.n_iters = config.get("n_iters", 1000)
        
        self.W = None
        self.solver = None

    def initialize_solver(self, W: WienerProcess) -> None:
        self.W = W
        self.solver = Euler(
            sde=self.X_circ,
            W=W
        )

    @partial(jax.jit, static_argnums=(0,))
    def run_step(self, run_state: RunState) -> RunState:
        current_path = run_state.path
        key1, key2, new_key = jax.random.split(run_state.rng_key, 3)
        dZs = self.W.sample(rng_key=key1, batch_size=self.batch_size)
        dWos = self.rho * current_path.dWs + jnp.sqrt(1 - self.rho**2) * dZs

        proposed_path = self.solver.solve(
            x0=self.X_circ.u, 
            rng_key=None, 
            dWs=dWos, 
            batch_size=self.batch_size,
            enforce_endpoint=self.X_circ.v_
        )
        log_ll_ratio = proposed_path.log_ll - current_path.log_ll

        log_u = jnp.log(jax.random.uniform(key2, shape=(self.batch_size,))) 
        accepted = log_u < log_ll_ratio
        new_total_n_accepted = accepted + run_state.total_n_accepted

        def update_field(acc: jnp.ndarray, x_new: Union[SamplePath, jnp.ndarray], x_old: Union[SamplePath, jnp.ndarray]) -> Union[SamplePath, jnp.ndarray]:
            if isinstance(x_new, jnp.ndarray) and x_new.shape[0] == accepted.shape[0]:
                return jax.vmap(
                    lambda a, n, o: jax.lax.cond(a, lambda _: n, lambda _: o, None)
                )(acc, x_new, x_old)
            else:
                return jax.lax.cond(acc[0], lambda _: x_new, lambda _: x_old, None)

        new_path = jax.tree_map(
            lambda x_new, x_old: update_field(accepted, x_new, x_old),
            proposed_path, current_path
        )
        new_state = run_state._replace(
            path=new_path,
            total_n_accepted=new_total_n_accepted,
            rng_key=new_key
        )

        return new_state

    def run_loop(self, n_iters: int, init_state: RunState, log_every: Optional[int] = None) -> Union[RunState, List[RunState]]:
        if log_every is not None:
            logs = []
            
            def run_with_logging(i, run_state):
                run_state = self.run_step(run_state)
                next_log_lls = run_state.log_lls.at[i+1].set(run_state.path.log_ll)
                run_state = run_state._replace(log_lls=next_log_lls)
                
                if i % log_every == 0 or i == n_iters - 1:
                    logs.append(run_state)

                return run_state
            
            final_state = init_state
            for i in tqdm(range(n_iters), desc=f"Running pCN with log every {log_every}"):
                final_state = run_with_logging(i, final_state)
            return logs, final_state
        else:
            return None, self._run_loop_no_logging(n_iters, init_state)

    @partial(jax.jit, static_argnums=(0, 1))
    def _run_loop_no_logging(self, n_iters: int, init_state: RunState) -> RunState:
        
        @loop_tqdm(n_iters, print_rate=1, desc="Running pCN", tqdm_type="std")
        def body_fun(i: int, run_state: RunState) -> RunState:
            run_state = self.run_step(run_state)
            next_log_lls = run_state.log_lls.at[i+1].set(run_state.path.log_ll)
            return run_state._replace(log_lls=next_log_lls)

        final_state = jax.lax.fori_loop(0, n_iters, body_fun, init_state)
        return final_state

    def run(self, log_every: Optional[int] = None) -> RunState:
        assert self.W is not None and self.solver is not None, "Path solver and Wiener process must be initialized before running pCN, initialize them with .init_path_solver()"
        rng_key = jax.random.PRNGKey(self.seed)
        key1, key2 = jax.random.split(rng_key)
        init_dWs = self.W.sample(rng_key=key1, batch_size=self.batch_size)
        init_path = self.solver.solve(
            x0=self.X_circ.u, 
            rng_key=None, 
            dWs=init_dWs, 
            batch_size=self.batch_size,
            enforce_endpoint=self.X_circ.v_
        )
        log_lls = jnp.zeros((self.n_iters + 1, self.batch_size))
        log_lls = log_lls.at[0].set(init_path.log_ll)

        init_state = RunState(
            path=init_path, 
            total_n_accepted=jnp.zeros(self.batch_size), 
            log_lls=log_lls,
            rng_key=key2
        )
        (logs, final_state) = self.run_loop(self.n_iters, init_state, log_every)

        accept_rates = jnp.mean(final_state.total_n_accepted) / self.n_iters
        logging.info(f"Batch average acceptance rate: {accept_rates:.2%}")
        return logs, final_state
