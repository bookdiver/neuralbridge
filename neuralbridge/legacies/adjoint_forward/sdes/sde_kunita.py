import jax
import jax.numpy as jnp

from . import sde_utils
from einops import rearrange


def kunita(T, N, num_landmarks, x0=None, y=None, dim=2, sigma=1.0, kappa=0.1):
    gauss_kernel = gaussian_kernel_2d(kappa, sigma)
    # matern_kernel = matern_kernel_5_2(kappa, sigma)

    def drift(t, x, *args):
        return jnp.zeros_like(x)

    def diffusion(t: float, x: jnp.ndarray) -> jnp.ndarray:
        if x0 is not None:
            x = x.reshape(-1, dim) + x0.reshape(-1, dim)
        x = x.reshape(-1, dim)
        corr_mat = gauss_kernel(x[:, None, :], x[None, :, :])
        q_half = jnp.einsum("i j, k l -> i k j l", corr_mat, jnp.eye(num_landmarks, dtype=x.dtype))
        q_half = rearrange(q_half, "i k j l -> (i k) (j l)")
        return q_half

    def adj_drift(t, x, partials=None):
        if y is not None:
            x = x + y
        if partials is None:
            partials = _partial_derivatives_by_row(t, x)
        kernel_matrix = diffusion(t, x)
        sum_of_partials = partials.sum(axis=0)
        return 1 / x.size * kernel_matrix @ sum_of_partials

    def adj_diffusion(t, x, *args):
        if y is not None:
            x = x + y
        return diffusion(t, x)

    def correction(t, y, corr, partials=None):
        if partials is None:
            partials = _partial_derivatives_by_row(t, y)
        sum_rows = jnp.sum(partials, axis=0)
        total_sum = sum_rows @ sum_rows.T
        c = 0.5 * total_sum
        return c * corr

    def _partial_derivatives_by_row(t, x):
        """Computes matrix A_ij = d/dx_i k(x_i, y_j)"""
        identity = jnp.identity(x.size)

        def diffusion_row(y, row_i: int):
            """Computes sigm, i.e. the ith row of the diffusion matrix"""
            return diffusion(t, y)[row_i, :]

        rows = jnp.arange(0, x.size)

        def partial_derivative_per_row(i):
            dxi = identity[i, :]
            diffusion_i = lambda y: diffusion_row(y, i)
            _diff, d_diff = jax.jvp(diffusion_i, (x,), (dxi,))
            return d_diff

        matrix = jax.vmap(partial_derivative_per_row)(rows)
        return matrix

    return sde_utils.SDE(
        T,
        N,
        num_landmarks * dim,
        drift,
        diffusion,
        adj_drift,
        adj_diffusion,
        correction,
        None,
        (num_landmarks * dim,),
        None,
    )


def matern_kernel_5_2(alpha, sigma):
    def k(x, y):
        dist = jnp.sum(jnp.abs(x - y), axis=-1)
        return (
            alpha
            * (1 + jnp.sqrt(5) * dist / sigma + 5 * dist**2 / (3 * sigma**2))
            * jnp.exp(-jnp.sqrt(5) * dist / sigma)
        )

    return k


def gaussian_kernel_2d(alpha: float, sigma: float) -> callable:
    def k(x, y):
        return 0.5 * alpha * jnp.exp(-0.5 * jnp.sum(jnp.square(x - y), axis=-1) / (sigma**2))

    return k
