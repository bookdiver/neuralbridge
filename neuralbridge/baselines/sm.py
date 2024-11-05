from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from ..stochastic_processes.bases import ContinuousTimeProcess
from ..stochastic_processes.wiener import WienerProcess
from ..solvers.sde import Euler

class ScoreMatchingReverseBridge:
    def __init__(self, 
                 x0: jnp.ndarray,
                 xT: jnp.ndarray,
                 proc: ContinuousTimeProcess,
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
        
        self.proc = proc
        self.dim = proc.dim
        self.dtype = proc.dtype
        self.T = proc.T
        self.dt = proc.dt
        
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
        self.solver = Euler(self.proc, self.wiener)
        
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

    @classmethod
    def get_reverse_bridge(cls, state: TrainState, proc: ContinuousTimeProcess):
        f = proc.f
        g = proc.g
        Sigma = proc.Sigma
        div_Sigma = proc.div_Sigma
        
        def transition_score_fn(t, x):
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
                self.transition_score_fn = transition_score_fn
                
            def f(self, t, x):
                reverse_t = self.T - t
                return -f(reverse_t, x) * 0.5 + Sigma(reverse_t, x) @ self.transition_score_fn(reverse_t, x) + div_Sigma(reverse_t, x)
            
            def g(self, t, x):
                reverse_t = self.T - t
                return g(reverse_t, x)
            
        return ReverseBridgeProcess(proc.T, proc.dt, proc.dim, proc.dtype)
    
    def _create_functions(self):
        
        def _get_grads_and_Sigmas(ts, xs):
            euler = xs[1:] - xs[:-1] - jax.vmap(
                    lambda t, x: self.proc.f(t, x))(ts[:-1], xs[:-1]) * self.dt
            Sigmas = jax.vmap(
                lambda t, x: self.proc.Sigma(t, x)
            )(ts[1:], xs[1:])
            inv_Sigmas = jax.vmap(
                lambda t, x: self.proc.inv_Sigma(t, x)
            )(ts[:-1], xs[:-1])
            grads = - jnp.einsum('tij,tj->ti', inv_Sigmas, euler) / self.dt
            return grads, Sigmas
        
        def _compute_loss(params, ts, xs, train):
            grads, Sigmas = jax.vmap(_get_grads_and_Sigmas)(ts, xs)
            
            b_size, t_size, d_size = xs[:, 1:, :].shape
            ts = ts[:, 1:].reshape((b_size * t_size, ))
            xs = xs[:, 1:, :].reshape((b_size * t_size, d_size))
            grads = grads.reshape((b_size * t_size, d_size))
            Sigmas = Sigmas.reshape((b_size * t_size, d_size, d_size))
            
            scores = self.model.apply(
                {'params': params}, 
                ts,
                xs, 
                train=train
            )
            
            losses = scores - grads
            losses = jnp.einsum('bi,bij,bj->b', losses, Sigmas, losses)
            losses = losses.reshape((b_size, t_size))
            loss = 0.5 * jnp.mean(
                jnp.sum(
                    losses,
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
                print(f'Iter {iter}/{n_iters}, Loss: {loss}')
        return self.state
        
    def run(self, n_iters: int, state: TrainState | None = None):
        if state is None:
            state = self.train(n_iters)
        
        reverse_bridge = self.get_reverse_bridge(state, self.proc)
        solver = Euler(reverse_bridge, self.wiener)
        path = solver.solve(
            x0=self.xT,
            rng_key=self.rng_key,
            dWs=None,
            n_batches=self.n_batches,
        )
        return path, reverse_bridge
        
        
class ScoreMatchingForwardBridge:
    def __init__(self,
                 x0: jnp.ndarray,
                 xT: jnp.ndarray,
                 proc: ContinuousTimeProcess,
                 *,
                 reverse_score_state: TrainState,
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
        
        self.proc = proc
        self.dim = proc.dim
        self.dtype = proc.dtype
        self.T = proc.T
        self.dt = proc.dt
        
        self.reverse_bridge_proc = ScoreMatchingReverseBridge.get_reverse_bridge(reverse_score_state, proc)
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
        self.solver = Euler(self.reverse_bridge_proc, self.wiener)
    
    def _sample_path(self, rng_key: jax.Array):
        path = self.solver.solve(
            x0=self.xT,
            rng_key=rng_key,
            dWs=None,
            n_batches=self.n_batches,
        )
        ts, xs = path.ts[::-1], path.xs
        ts = jnp.broadcast_to(ts, (self.n_batches, *ts.shape))
        return ts, xs
    
    @classmethod
    def get_forward_bridge(cls, state: TrainState, proc: ContinuousTimeProcess, transition_score_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
        f = proc.f
        g = proc.g
        Sigma = proc.Sigma
        transition_score_fn = transition_score_fn
        
        def marginal_score_fn(t, x):
            _t = jnp.array([t])
            _x = jnp.expand_dims(x, axis=0)
            return state.apply_fn(
                {'params': state.params}, 
                _t, 
                _x, 
                train=False
            ).squeeze()
            
        class ForwardBridgeProcess(ContinuousTimeProcess):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.score_fn = lambda t, x: marginal_score_fn(t, x) \
                    - transition_score_fn(t, x)
                
            def f(self, t, x):
                return f(t, x) + self.Sigma(t, x) @ self.score_fn(t, x)
            
            def g(self, t, x):
                return g(t, x)
            
            def Sigma(self, t, x):
                return Sigma(t, x)
        
        return ForwardBridgeProcess(proc.T, proc.dt, proc.dim, proc.dtype)
    
    def _create_functions(self):
    
        def _get_grads_and_Sigmas(ts, xs):
            euler = xs[:-1] - xs[1:] - jax.vmap(
                    lambda t, x: self.reverse_bridge_proc.f(t, x))(ts[1:], xs[1:]) * self.dt
            Sigmas = jax.vmap(
                lambda t, x: self.reverse_bridge_proc.Sigma(t, x)
            )(ts[:-1], xs[:-1])
            inv_Sigmas = jax.vmap(
                lambda t, x: self.reverse_bridge_proc.inv_Sigma(t, x)
            )(ts[1:], xs[1:])
            grads = - jnp.einsum('tij,tj->ti', inv_Sigmas, euler) / self.dt
            return grads, Sigmas
        
        def _compute_loss(params, ts, xs, train):
            grads, Sigmas = jax.vmap(_get_grads_and_Sigmas)(ts, xs)
            
            b_size, t_size, d_size = xs[:, 1:, :].shape
            ts = ts[:, :-1].reshape((b_size * t_size, ))
            xs = xs[:, :-1, :].reshape((b_size * t_size, d_size))
            grads = grads.reshape((b_size * t_size, d_size))
            Sigmas = Sigmas.reshape((b_size * t_size, d_size, d_size))
            
            scores = self.model.apply(
                {'params': params}, 
                ts,
                xs, 
                train=train
            )
            
            losses = scores - grads
            losses = jnp.einsum('bi,bij,bj->b', losses, Sigmas, losses)
            losses = losses.reshape((b_size, t_size))
            loss = 0.5 * jnp.mean(
                jnp.sum(
                    losses,
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
        
        forward_bridge = self.get_forward_bridge(state, self.proc, self.reverse_bridge_proc.transition_score_fn)
        solver = Euler(forward_bridge, self.wiener)
        path = solver.solve(
            x0=self.x0,
            rng_key=self.rng_key,
            dWs=None,
            n_batches=self.n_batches,
        )
        return path, forward_bridge

        
    