from jax_tqdm import loop_tqdm

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.conds import (
    GuidedBridgeProcess, NeuralBridgeProcess
)
from neuralbridge.solvers.sde import Euler, WienerProcess
from neuralbridge.utils.sample_path import SamplePath
from neuralbridge.utils.t_grid import TimeGrid

RunState = namedtuple("RunState", ["iteration","path", "total_n_accepted", "log_likelihood_ratio", "rng_key"])

class PreconditionedCrankNicolson:
    X: GuidedBridgeProcess | NeuralBridgeProcess
    config: OmegaConf
    
    def __init__(self, X: GuidedBridgeProcess | NeuralBridgeProcess, config: OmegaConf):
        self.X = X

        # MCMC parameters
        self.eta = config.mcmc.eta
        self.n_chains = config.mcmc.n_chains
        self.n_iters = config.mcmc.n_iters
        self.n_burn_in = config.mcmc.n_burn_in
        
        self.tGrid = TimeGrid(
            T=config.sde.T,
            dt=config.sde.dt,
            t_scheme=config.sde.t_scheme
        )
        
        self.is_neural_bridge = isinstance(self.X, NeuralBridgeProcess)
        
        self.W = WienerProcess(dim=config.sde.W_dim)
        self.solver = Euler(
            X=self.X,
            W=self.W,
            tGrid=self.tGrid
        )

    @partial(jax.jit, static_argnums=(0,))
    def run_step(self, run_state: RunState) -> RunState:
        current_path = run_state.path
        key1, key2, new_key = jax.random.split(run_state.rng_key, 3)
        dZs = self.W.sample(
            rng_key=key1, 
            dts=self.tGrid.dts,
            batch_size=self.n_chains
        )
        dWos = self.eta * current_path.dWs + jnp.sqrt(1 - self.eta**2) * dZs

        proposed_path = self.solver.solve(
            x0=self.X.u, 
            dWs=dWos, 
            batch_size=self.n_chains,
            enforced_endpoint=None,
            compute_log_likelihood_ratio=True,
            correct_log_likelihood_ratio=self.is_neural_bridge,
            nn_variables=FrozenDict({}),
            training=False,
        )
        log_likelihood_ratio = proposed_path.log_likelihood_ratio - current_path.log_likelihood_ratio

        log_u = jnp.log(jax.random.uniform(key2, shape=(self.n_chains,))) 
        accepted = log_u < log_likelihood_ratio
        
        new_total_n_accepted = jax.lax.cond(
            run_state.iteration > self.n_burn_in,
            lambda _: accepted + run_state.total_n_accepted,
            lambda _: run_state.total_n_accepted,
            None
        )

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
            iteration=run_state.iteration + 1,
            path=new_path,
            total_n_accepted=new_total_n_accepted,
            rng_key=new_key
        )

        return new_state

    def run_loop(self, n_iters: int, init_state: RunState, log_every: Optional[int] = None) -> Union[RunState, List[RunState]]:
        logs = []
        
        def run_with_logging(run_state):
            run_state = self.run_step(run_state)
            next_log_likelihood_ratio = run_state.log_likelihood_ratio.at[run_state.iteration+1].set(run_state.path.log_likelihood_ratio)
            run_state = run_state._replace(log_likelihood_ratio=next_log_likelihood_ratio)
            
            if run_state.iteration >= self.n_burn_in and (run_state.iteration % log_every == 0 or run_state.iteration == n_iters - 1):
                logs.append(run_state)

            return run_state
        
        final_state = init_state
        for _ in tqdm(range(n_iters), desc=f"Running pCN with log every {log_every}"):
            final_state = run_with_logging(final_state)
        return logs, final_state


    # @partial(jax.jit, static_argnums=(0, 1))
    # def _run_loop_no_logging(self, n_iters: int, init_state: RunState) -> RunState:
        
    #     @loop_tqdm(n_iters, print_rate=1, desc="Running pCN", tqdm_type="std")
    #     def body_fun(i: int, run_state: RunState) -> RunState:
    #         run_state = self.run_step(run_state)
    #         next_log_likelihood_ratio = run_state.log_likelihood_ratio.at[i+1].set(run_state.path.log_likelihood_ratio)
    #         return run_state._replace(log_likelihood_ratio=next_log_likelihood_ratio)

    #     final_state = jax.lax.fori_loop(0, n_iters, body_fun, init_state)
    #     return final_state

    def run(self, rng_key: Optional[jax.Array] = DEFAULT_RNG_KEY, log_every: Optional[int] = None) -> RunState:
        key1, key2 = jax.random.split(rng_key)
        init_dWs = self.W.sample(rng_key=key1, dts=self.tGrid.dts, batch_size=self.n_chains)
        init_path = self.solver.solve(
            x0=self.X.u, 
            dWs=init_dWs, 
            batch_size=self.n_chains,
            enforced_endpoint=None,
            compute_log_likelihood_ratio=True,
            correct_log_likelihood_ratio=self.is_neural_bridge,
            nn_variables=FrozenDict({}),
            training=False,
        )
        log_likelihood_ratio = jnp.zeros((self.n_iters + 1, self.n_chains))
        log_likelihood_ratio = log_likelihood_ratio.at[0].set(init_path.log_likelihood_ratio)

        init_state = RunState(
            iteration=0,
            path=init_path, 
            total_n_accepted=jnp.zeros(self.n_chains), 
            log_likelihood_ratio=log_likelihood_ratio,
            rng_key=key2
        )
        (logs, final_state) = self.run_loop(self.n_iters, init_state, log_every)

        accept_rates = jnp.mean(final_state.total_n_accepted) / (self.n_iters - self.n_burn_in)
        logging.info(f"Batch average acceptance rate: {accept_rates:.2%}")
        return logs, final_state
