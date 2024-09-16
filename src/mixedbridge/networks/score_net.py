from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

class ScoreNetSmall(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, train: bool = True):
        t = jnp.expand_dims(t, axis=-1)
        x = jnp.concatenate([t, x], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            x = nn.swish(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x
    
    