import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from ..stochastic_processes.bases import ContinuousTimeProcess
from ..stochastic_processes.wiener import WienerProcess
from ..solvers.sde import Euler
from ..utils.sample_path import SamplePath

class ScoreMatchingReverseBridge:
    def __init__(self, 
                 x0: jnp.ndarray,
                 xT: jnp.ndarray,
                 process: ContinuousTimeProcess,
                 *,
                 model: nn.Module,
                 rng_key: jax.Array = jax.random.PRNGKey(42),
                 n_batches: int = 1,
                 lr: float = 1e-3,
                 ):
        self.x0 = x0
        self.xT = xT
        self.model = model
        self.n_batches = n_batches
        self.lr = lr
        
        self.process = process
        self.dim = process.dim
        self.dtype = process.dtype
        self.T = process.T
        self.dt = process.dt
        
        self.wiener = WienerProcess(self.dim, self.dtype, self.T, self.dt)
        self.rng_key = rng_key
        
    def _initialize_model(self):
        dummy_t, dummy_x = jnp.zeros((1, )), jnp.zeros((1, self.dim))
        variables = self.model.init(self.rng_key, dummy_t, dummy_x, train=True)
        self.init_params = variables['params']
        self.state = TrainState.create(
            apply_fn=self.model.apply, 
            params=self.init_params, 
            tx=optax.adam(self.lr)
        )

    def _initialize_solver(self):
        self.solver = Euler(self.process, self.wiener)
        
    def _sample_path(self, rng_key: jax.Array):
        path = self.solver.solve(
            x0=self.x0,
            rng_key=rng_key,
            dWs=None,
            n_batches=self.n_batches,
        )
        ts, xs = path.ts, path.xs
        ts = jnp.broadcast_to(ts, (self.n_batches, *ts.shape))
        return ts, xs
        
    
    def _get_reverse_bridge(self, state):
        f = self.process.f
        g = self.process.g
        Sigma = self.process.Sigma
        inv_Sigma = self.process.inv_Sigma
        
        def score_fn(t, x):
            _t = jnp.array([t])
            _x = jnp.expand_dims(x, axis=0)
            return state.apply_fn(
                {'params': state.params}, 
                _t, 
                _x, 
                train=False
            ).squeeze()
        
        class ReverseBridgeProcess(ContinuousTimeProcess):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.score_fn = score_fn
                
            def f(self, t, x):
                reverse_t = self.T - t
                # TODO: Add the divergence term
                return -f(reverse_t, x) + Sigma(reverse_t, x) @ score_fn(reverse_t, x)
            
            def g(self, t, x):
                reverse_t = self.T - t
                return g(reverse_t, x)
            
            def Sigma(self, t, x):
                reverse_t = self.T - t
                return Sigma(reverse_t, x)
            
            def inv_Sigma(self, t, x):
                reverse_t = self.T - t
                return inv_Sigma(reverse_t, x)
            
        return ReverseBridgeProcess(self.T, self.dt, self.dim, self.dtype)
    
    def _create_functions(self):
        
        def _get_grads(ts, xs):
            euler = xs[1:] - xs[:-1] - jax.vmap(
                    lambda t, x: self.process.f(t, x))(ts[:-1], xs[:-1]) * self.dt
            inv_Sigmas = jax.vmap(
                lambda t, x: self.process.inv_Sigma(t, x)
            )(ts[:-1], xs[:-1])
            grads = - 1 / self.dt * jnp.einsum('tij,tj->ti', inv_Sigmas, euler)
            return grads
        
        def _compute_loss(params, ts, xs, train):
            grads = jax.vmap(_get_grads)(ts, xs)
            
            b_size, t_size, d_size = xs[:, 1:, :].shape
            ts = ts[:, 1:].reshape((b_size * t_size, ))
            xs = xs[:, 1:, :].reshape((b_size * t_size, d_size))
            grads = grads.reshape((b_size * t_size, d_size))
            
            scores = self.model.apply(
                {'params': params}, 
                ts,
                xs, 
                train=train
            )
            
            losses = (scores - grads).reshape((b_size, t_size, d_size))
            loss = 0.5 * jnp.mean(
                jnp.sum(
                    jnp.sum(jnp.square(losses), axis=-1), 
                    axis=-1
                ) * self.dt,
                axis=0
            )
            return loss
        
        def train_step(state, ts, xs):
            loss_fn = jax.value_and_grad(_compute_loss, argnums=0, has_aux=False)
            loss, grads = loss_fn(state.params, ts, xs, train=True)
            state = state.apply_gradients(grads=grads)
            return state, loss
    
        self.train_step = jax.jit(train_step)
    
    def train(self, n_iters: int):
        self._initialize_model()
        self._initialize_solver()
        self._create_functions()
        rng_key = self.rng_key
        
        for iter in range(n_iters):
            rng_key, subkey = jax.random.split(rng_key)
            ts, xs = self._sample_path(subkey)
            self.state, loss = self.train_step(self.state, ts, xs)
            if iter % 100 == 0:
                print(f'Iter {iter+1}/{n_iters}, Loss: {loss}')
        return self.state
        
    def run(self, n_iters: int, state: TrainState | None = None):
        if state is None:
            state = self.train(n_iters)
        
        reverse_bridge = self._get_reverse_bridge(state)
        solver = Euler(reverse_bridge, self.wiener)
        path = solver.solve(
            x0=self.xT,
            rng_key=self.rng_key,
            dWs=None,
            n_batches=self.n_batches,
        )
        return path, reverse_bridge
        
        
    
    
        
    