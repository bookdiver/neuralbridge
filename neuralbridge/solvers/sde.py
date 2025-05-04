from neuralbridge.setups import *
from neuralbridge.utils.sample_path import SamplePath
from neuralbridge.utils.t_grid import TimeGrid
from neuralbridge.stochastic_processes.unconds import ContinuousTimeProcess

StepState = namedtuple("StepState", ["x", "t", "nn_val"])
SolveState = namedtuple("SolveState", ["x", "t", "log_likelihood_ratio", "rng_key"])

class WienerProcess:
    dim: int
    dtype: Optional[jnp.dtype] = DEFAULT_DTYPE

    def __init__(self, dim: int):
        self.dim = dim

    def _sample_step(self, rng_key: jax.Array, dt: tTYPE) -> xTYPE:
        dW = jax.random.normal(rng_key, (self.dim,), self.dtype) * jnp.sqrt(dt)
        return dW
    
    def sample(self, rng_key: jax.Array, dts: Sequence[tTYPE], batch_size: int = 1) -> xTYPE:    
        n_steps = len(dts) 
        sub_keys = jax.random.split(rng_key, n_steps * batch_size).reshape((batch_size, n_steps, -1))
        dWs = jax.vmap(
            jax.vmap(self._sample_step,
            in_axes=(0, 0)
            ),
            in_axes=(0, None)           # vectorize over batch size
        )(sub_keys, dts)    # dWs.shape = (batch_size, n_steps, *self.shape)
        return dWs
    
    def get_path_sampler(self, rng_key: jax.Array, dts: Sequence[tTYPE], batch_size: int = 1) -> Generator[xTYPE, None, None]:
        while True:
            rng_key, sub_key = jax.random.split(rng_key)
            yield self.sample(sub_key, dts, batch_size)

class SDESolver(abc.ABC):
    X: ContinuousTimeProcess
    W: WienerProcess
    tGrid: TimeGrid

    def __init__(
        self, 
        X: ContinuousTimeProcess, 
        W: WienerProcess,
        tGrid: TimeGrid
    ):
        self.X = X
        self.W = W
        self.tGrid = tGrid
    
    @abc.abstractmethod
    def _step(
        self, 
        x: xTYPE, 
        t: tTYPE, 
        dt: tTYPE, 
        dW: xTYPE,
        nn_variables: FrozenDict, 
        training: bool,
        rngs: jax.Array | dict
    ) -> StepState:
        pass
    
    @partial(
        jax.jit, 
        static_argnums=(0, ), 
        static_argnames=(
            "batch_size",
            "compute_log_likelihood_ratio", 
            "correct_log_likelihood_ratio",
            # "nn_variables",
            "training"
        )
    )
    def solve(
        self, 
        x0: jnp.ndarray, 
        rng_key: jax.Array = DEFAULT_RNG_KEY, 
        dWs: jnp.ndarray | None = None, 
        batch_size: int = 1, 
        enforced_endpoint: jnp.ndarray | None = None, 
        compute_log_likelihood_ratio: bool = False, 
        correct_log_likelihood_ratio: bool = False,
        nn_variables: FrozenDict = FrozenDict({}), 
        training: bool = False,
    ) -> SamplePath:
        
        if compute_log_likelihood_ratio:
            assert hasattr(self.X, "G"), "current SDE does not support computing likelihood ratio"
            
        def scan_fn(solver_state: namedtuple, carry_args: tuple) -> tuple:
            dt, dW = carry_args
            key, next_key = jax.random.split(solver_state.rng_key)
            new_step_state = self._step(
                x=solver_state.x, 
                t=solver_state.t, 
                dt=dt, 
                dW=dW,
                nn_variables=nn_variables,
                training=training,
                rngs={
                    "dropout": key
                }
            )
            G_val = self.X.G(solver_state.t, solver_state.x) if compute_log_likelihood_ratio else 0.0
            new_log_likelihood_ratio = solver_state.log_likelihood_ratio + G_val * dt
            new_state = solver_state._replace(
                x=new_step_state.x, 
                t=new_step_state.t,
                log_likelihood_ratio=new_log_likelihood_ratio,
                rng_key=next_key
            )
            return new_state, (new_step_state.x, new_step_state.nn_val)
        
        if dWs is None:
            dWs = self.W.sample(rng_key, self.tGrid.dts, batch_size)
        else:
            assert dWs.shape[0] == batch_size, "Number of batches must match the shape of dWs"

        init_state = SolveState(x=x0, t=0.0, log_likelihood_ratio=0.0, rng_key=rng_key)
        final_state, (xs, nn_vals) = jax.vmap(
            lambda dW: jax.lax.scan(
                scan_fn, 
                init=init_state, 
                xs=(self.tGrid.dts, dW),
                length=self.tGrid.n_steps,
            ),
            in_axes=0
        )(dWs)
        x0_tiled = repeat(x0, "i -> b 1 i", b=batch_size)
        xs = jnp.concatenate([x0_tiled, xs], axis=1)
        assert xs.shape[1] == self.tGrid.n_steps + 1
        
        if enforced_endpoint is not None:
            if enforced_endpoint.size < xs.shape[-1]:
                xs = xs.at[:, -1, :].set(
                    jnp.where(
                        ~jnp.isnan(enforced_endpoint),
                        enforced_endpoint[jnp.newaxis, :],
                        xs[:, -1, :]
                    )
                )
            else:
                xs = xs.at[:, -1, :].set(enforced_endpoint[jnp.newaxis, :])
                
        if correct_log_likelihood_ratio:
            sto_integrand = -jnp.einsum("b t i, b t i -> b t", nn_vals, dWs)
            det_integrand = -0.5 * jnp.sum(jnp.square(nn_vals), axis=-1)
            sto_integral = jnp.sum(sto_integrand, axis=1)
            det_integral = jnp.sum(det_integrand * self.tGrid.dts[jnp.newaxis, :], axis=1)
            log_likelihood_ratio_correction = sto_integral + det_integral
            log_likelihood_ratio = final_state.log_likelihood_ratio + log_likelihood_ratio_correction
        else:
            log_likelihood_ratio = final_state.log_likelihood_ratio
            
        return SamplePath(
            name=f"{self.X.__class__.__name__} sample path",
            xs=xs, 
            ts=self.tGrid.ts,
            dWs=dWs,
            dts=self.tGrid.dts,
            log_likelihood_ratio=log_likelihood_ratio,
            nn_vals=nn_vals
        )

class Euler(SDESolver):
    def _step(
        self, 
        x: xTYPE, 
        t: tTYPE, 
        dt: tTYPE, 
        dW: xTYPE,
        nn_variables: FrozenDict = FrozenDict({}), 
        training: bool = False,
        rngs: jax.Array | dict = {}
    ) -> StepState:
        drifts = self.X.f(t, 
                          x,
                          nn_variables=nn_variables,
                          training=training,
                          rngs=rngs)
        if isinstance(drifts, tuple):
            drifts, nn_val = drifts
        else:
            nn_val = jnp.zeros_like(drifts)
        diffusivity = self.X.g(t, x)
        x_new = x + drifts * dt + jnp.einsum("i j, j -> i", diffusivity, dW)
        t_new = t + dt
        return StepState(x=x_new, t=t_new, nn_val=nn_val)
