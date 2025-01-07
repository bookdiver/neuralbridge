from neuralbridge.setups import *
from neuralbridge.utils.sample_path import SamplePath
from neuralbridge.stochastic_processes.unconds import ContinuousTimeProcess

SolverState = namedtuple("SolverState", ["x", "log_ll"])

class WienerProcess:

    def __init__(self,
                 T: float, 
                 dt: float,
                 shape: Tuple[int, ...],
                 dtype: Optional[jnp.dtype] = DEFAULT_DTYPE,
                 t_scheme: str = "linear"):
        self.T = T
        self.dt = dt
        self.shape = shape
        self.dtype = dtype
        self.t_scheme = t_scheme
        
    @property
    def ts(self):
        if self.t_scheme == "linear":
            return self._linear_t_scheme()
        elif self.t_scheme == "quadratic":
            return self._quadratic_t_scheme()
        else:
            raise ValueError(f"Time scheme: {self.t_scheme} not found!")
        
    @property
    def dts(self):
        return jnp.diff(self.ts)
    
    def _linear_t_scheme(self):
        ts = jnp.arange(0.0, self.T + self.dt, self.dt, dtype=self.dtype)
        return ts
    
    def _quadratic_t_scheme(self):
        ts = jnp.arange(0.0, self.T + self.dt, self.dt, dtype=self.dtype)
        ts =  ts * (2.0 - ts / self.T)
        return ts

    def _sample_step(self, rng_key: jax.Array, dt: float) -> jnp.ndarray:
        dW = jax.random.normal(rng_key, self.shape, self.dtype) * jnp.sqrt(dt)
        return dW
    
    def sample(self, rng_key: jax.Array, batch_size: int = 1) -> jnp.ndarray:
        n_steps = len(self.dts)
        
        sub_keys = jax.random.split(rng_key, n_steps * batch_size).reshape((batch_size, n_steps, -1))
        dWs = jax.vmap(
            jax.vmap(
                self._sample_step,
                in_axes=(0, 0)   # vectorize over time steps
            ),
            in_axes=(0, None)    # vectorize over batch size
        )(sub_keys, self.dts)    # dWs.shape = (batch_size, n_steps, *self.shape)
        
        return dWs
    
    def get_path_sampler(self, rng_key: jax.Array, batch_size: int = 1) -> Generator[jnp.ndarray, None, None]:
            
        while True:
            rng_key, sub_key = jax.random.split(rng_key)
            yield self.sample_path(sub_key, batch_size)

class SDESolver(abc.ABC):
    sde: ContinuousTimeProcess
    W: WienerProcess

    def __init__(
        self, 
        sde: ContinuousTimeProcess, 
        W: WienerProcess
    ):
        self.sde = sde
        self.W = W
        
    @property
    def T(self):
        return self.W.T
    
    @property
    def ts(self):
        return self.W.ts
    
    @property
    def dt(self):
        return self.W.dt
    
    @property
    def dts(self):
        return self.W.dts
    
    @abc.abstractmethod
    def _step(
        self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray, *args, **kwargs
    ):
        """ This method should be implemented by the subclass. 
        It should return the next state given the current state, time, time step, and Wiener increment.
        This method is called when solving an unparameterized SDE. On the contrary, the next method
        is called when solving a parameterized SDE and mainly used for backprop.
        
        Args:
            x (jnp.ndarray): The current state.
            t (float): The current time.
            dt (float): The time step.
            dW (jnp.ndarray): The Wiener increment.
        """
        pass
    
    @abc.abstractmethod
    @partial(jax.grad, argnums=5)
    def _step_with_variables(
        self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray, variables: Any,
    ):
        """ This method should be implemented by the subclass.
        It should return the next state and the next variable given the current state, time, time step, and Wiener increment.
        This method is called when solving a parameterized SDE and mainly used for backprop.
        
        Args:
            x (jnp.ndarray): The current state.
            t (float): The current time.
            dt (float): The time step.
            dW (jnp.ndarray): The Wiener increment.
            variables (Any): The variables for the neural network.
        """
        pass
    
    @partial(jax.jit, static_argnums=(0, 4))
    def solve(
        self, 
        x0: jnp.ndarray, 
        rng_key: Optional[jax.Array] = jax.random.PRNGKey(DEFAULT_SEED), 
        dWs: Optional[jnp.ndarray] = None, 
        batch_size: Optional[int] = 1,
        enforce_endpoint: Optional[jnp.ndarray] = None,
    ) -> SamplePath:
        
        if hasattr(self.sde, "G"):
            record_ll = True
        else:
            record_ll = False
            
        def scan_fn(solver_state: namedtuple, t_W_args: tuple) -> tuple:
            t, dt, dW = t_W_args
            x = self._step(
                x=solver_state.x, 
                t=t, 
                dt=dt, 
                dW=dW
            )
            G = self.sde.G(t, x) if record_ll else 0.0
            log_ll = solver_state.log_ll + G * dt
            new_state = solver_state._replace(
                x=x, 
                log_ll=log_ll, 
            )
            return new_state, x
        
        if dWs is None:
            dWs = self.W.sample(rng_key, batch_size)
        else:
            assert dWs.shape[0] == batch_size, "Number of batches must match the shape of dWs"

        init_state = SolverState(x=x0, log_ll=0.0)
        final_state, xs = jax.vmap(
            lambda dW: jax.lax.scan(
                scan_fn, 
                init=init_state, 
                xs=(self.ts[:-1], self.dts, dW)
            ),
            in_axes=0
        )(dWs)
        x0_tiled = repeat(x0, "i -> b 1 i", b=batch_size)
        xs = jnp.concatenate([x0_tiled, xs], axis=1) 
        
        if enforce_endpoint is not None:
            # Check if enforce_end_point has fewer dimensions than xs
            if enforce_endpoint.size < xs.shape[-1]:
                # Create a boolean mask for the components to enforce
                mask = ~jnp.isnan(enforce_endpoint)
                # Only update the non-NaN components
                xs = xs.at[:, -1].set(
                    jnp.where(
                        mask,
                        enforce_endpoint[None, ...],
                        xs[:, -1]
                    )
                )
            else:
                # Original behavior for full enforcement
                xs = xs.at[:, -1].set(enforce_endpoint[None, ...])
            
        return SamplePath(
            name=f"{self.sde.__class__.__name__} sample path",
            xs=xs, 
            ts=self.ts,
            dWs=dWs,
            dts=self.dts,
            log_ll=final_state.log_ll
        )
        
    def solve_with_variables(
        self,
        x0: jnp.ndarray,
        variables: Any,
        dWs: jnp.ndarray,
        batch_size: Optional[int] = 1,
        enforce_endpoint: Optional[jnp.ndarray] = None,
    ) -> SamplePath:
        if hasattr(self.sde, "G"):
            record_ll = True
        else:
            record_ll = False

        def scan_fn(solver_state: namedtuple, t_W_args: tuple) -> tuple:
            t, dt, dW = t_W_args
            x, nu = self._step_with_variables(
                x=solver_state.x, 
                t=t, 
                dt=dt, 
                dW=dW,
                variables=variables,
            )
            G = self.sde.G(t, x) if record_ll else 0.0
            log_ll = solver_state.log_ll + G * dt
            new_state = solver_state._replace(
                x=x, 
                log_ll=log_ll, 
            )
            return new_state, (x, nu)
        
        assert dWs.shape[0] == batch_size, "Number of batches must match the shape of dWs"

        init_state = SolverState(x=x0, log_ll=0.0)
        final_state, (xs, nus) = jax.vmap(
            lambda dW: jax.lax.scan(
                scan_fn, 
                init=init_state, 
                xs=(self.ts[:-1], self.dts, dW)
            ),
            in_axes=0
        )(dWs)
        x0_tiled = repeat(x0, "i -> b 1 i", b=batch_size)
        xs = jnp.concatenate([x0_tiled, xs], axis=1) 
        
        if enforce_endpoint is not None:
            # Check if enforce_end_point has fewer dimensions than xs
            if enforce_endpoint.size < xs.shape[-1]:
                # Create a boolean mask for the components to enforce
                mask = ~jnp.isnan(enforce_endpoint)
                # Only update the non-NaN components
                xs = xs.at[:, -1].set(
                    jnp.where(
                        mask,
                        enforce_endpoint[None, ...],
                        xs[:, -1]
                    )
                )
            else:
                # Original behavior for full enforcement
                xs = xs.at[:, -1].set(enforce_endpoint[None, ...])
            
        return SamplePath(
            name=f"{self.sde.__class__.__name__} sample path",
            xs=xs, 
            ts=self.ts,
            dWs=dWs,
            dts=self.dts,
            log_ll=final_state.log_ll,
            nus=nus,
        )


class Euler(SDESolver):
    def _step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray) -> jnp.ndarray:
        drift = self.sde.f(t, x)
        diffusion = self.sde.g(t, x)
        return x + drift * dt + jnp.einsum('i j, j -> i', diffusion, dW)
    
    def _step_with_variables(
        self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray, variables: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        drift, nu = self.sde.f(t, x, variables)
        diffusion = self.sde.g(t, x)
        return x + drift * dt + jnp.einsum('i j, j -> i', diffusion, dW), nu
