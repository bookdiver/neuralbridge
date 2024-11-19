from flax import linen as nn

from neuralbridge.setups import *
from neuralbridge.networks.time_embedding import TimeEmbedding, TimeEmbeddingMLP

def get_activation(activation: str) -> Callable:
    if activation == "relu":
        return nn.relu
    elif activation == "tanh":
        return nn.tanh
    elif activation == "gelu":
        return nn.gelu
    elif activation == "silu":
        return nn.silu
    elif activation == "leaky_relu":
        return nn.leaky_relu
    elif activation == "lipswish":
        return lambda x: 0.909 * nn.swish(x)
    else:
        raise ValueError(f"Activation {activation} not supported")

class MLPSmall(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    norm: Optional[str] = None
    activation: Optional[str] = "tanh"
    dtype: Optional[jnp.dtype] = jnp.float32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, training: bool = True):
        
        x = jnp.concatenate([t, x], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            if self.norm == "batch":
                x = nn.BatchNorm(use_running_average=not training)(x)
            elif self.norm == "layer":
                x = nn.LayerNorm()(x)
            x = get_activation(self.activation)(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

class MLP(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    norm: Optional[str] = None
    activation: Optional[str] = "tanh"
    t_emb_dim: int = 32
    t_emb_max_period: float = 100.0
    t_emb_scaling: float = 100.0
    dtype: Optional[jnp.dtype] = jnp.float32
    
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
            x = get_activation(self.activation)(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x