import jax.numpy as jnp
from .solver import SDESolver

class Euler(SDESolver):

    def step(self, x: jnp.ndarray, t: float, dt: float, dW: jnp.ndarray):
        return x + self.sde.f(t, x) * dt + self.sde.g(t, x) @ dW