import jax
import jax.numpy as jnp
from flax import linen as nn

class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    t_emb_dim: int
    scaling: float
    max_period: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, t: jnp.ndarray):
        """ t shape: (batch, 1) """
        pe = jnp.empty((len(t), self.t_emb_dim), dtype=self.dtype)
        factor = self.scaling * jnp.einsum('... j, k -> ... k', 
                                           t, 
                                           jnp.exp(jnp.arange(0, self.t_emb_dim, 2) \
                                                   * (-jnp.log(self.max_period) / self.t_emb_dim)))
        pe = pe.at[..., 0::2].set(jnp.sin(factor))
        pe = pe.at[..., 1::2].set(jnp.cos(factor))
        return pe


class TimeEmbeddingMLP(nn.Module):
    output_dim: int
    # act_fn: str

    @nn.compact
    def __call__(self, t_emb: jnp.ndarray) -> tuple[jax.Array, jax.Array]:
        scale_shift = nn.Dense(
            2 * self.output_dim, kernel_init=nn.initializers.xavier_normal()
        )(t_emb)
        scale, shift = jnp.array_split(scale_shift, 2, axis=-1)
        return scale, shift