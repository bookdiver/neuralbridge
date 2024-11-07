from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .time_mlp import TimeEmbedding, TimeEmbeddingMLP

class ScoreNetSmall(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    norm: str | None = None
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, training: bool = True):
        
        x = jnp.concatenate([t, x], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            if self.norm == "batch":
                x = nn.BatchNorm(use_running_average=not training)(x)
            elif self.norm == "layer":
                x = nn.LayerNorm()(x)
            x = nn.swish(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

class ScoreNet(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    norm: str | None = None
    t_emb_dim: int = 32
    t_emb_max_period: float = 100.0
    t_emb_scaling: float = 100.0
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, training: bool = True):
        t_emb = TimeEmbedding(
            self.t_emb_dim,
            self.t_emb_max_period,
            self.t_emb_scaling
        )(t)
        
        x = nn.Dense(self.t_emb_dim, dtype=self.dtype)(x)
        
        x = jnp.concatenate([t_emb, x], axis=-1)
        
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            if self.norm == "batch":
                x = nn.BatchNorm(use_running_average=not training)(x)
            elif self.norm == "layer":
                x = nn.LayerNorm()(x)
            x = nn.tanh(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x