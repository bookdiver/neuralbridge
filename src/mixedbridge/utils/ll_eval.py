from __future__ import annotations
import jax
import jax.numpy as jnp

from ..stochastic_processes.bases import ContinuousTimeProcess
from .sample_path import SamplePath

@jax.jit
def normal_logpdf(x: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    dim = x.shape[-1]
    constant = -0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(jnp.linalg.det(var))
    return constant - 0.5 * (x - mean).T @ jnp.linalg.inv(var) @ (x - mean)

@jax.jit
def stable_normal_logpdf(x: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    dim = x.shape[-1]
    diff = x - mean
    chol = jnp.linalg.cholesky(cov)
    quad = jnp.sum(jnp.square(jnp.linalg.solve(chol, diff)))
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(chol)))
    return -0.5 * (dim * jnp.log(2 * jnp.pi) + logdet + quad)

def log_likelihood(
    sde: ContinuousTimeProcess,
    sample_path: SamplePath,
    eval_every: int = 1
):
    log_lls = jnp.zeros(sample_path.n_batches)
    
    for i in range(0, sample_path.n_steps-1, eval_every):
        x_now = sample_path.xs[:, i]
        t_now = sample_path.ts[i]
        x_next = sample_path.xs[:, i+eval_every]
        t_next = sample_path.ts[i+eval_every]
        f = jax.vmap(sde.f, in_axes=(None, 0))(t_now, x_now)
        Sigma = jax.vmap(sde.Sigma, in_axes=(None, 0))(t_now, x_now) + 1e-6 * jnp.broadcast_to(jnp.eye(sde.dim), (sample_path.n_batches, sde.dim, sde.dim))
        euler = x_now + f * (t_next - t_now)
        log_lls += jax.vmap(stable_normal_logpdf)(x_next, euler, Sigma*(t_next - t_now))

    return log_lls