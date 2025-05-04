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
    activation: Optional[str] = "tanh"
    normalization_type: Optional[Union[str, None]] = None
    dropout_rate: Optional[float] = 0.0
    dtype: Optional[jnp.dtype] = DEFAULT_DTYPE
    
    @nn.compact
    def __call__(self, t: tTYPE, x: xTYPE, training: bool):
        
        x = jnp.concatenate([t, x], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            if self.normalization_type == "batch":
                x = nn.BatchNorm(use_running_average=not training)(x)
            elif self.normalization_type == "layer":
                x = nn.LayerNorm()(x)
            elif self.normalization_type == None:
                pass
            x = get_activation(self.activation)(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        
        return x

class MLP(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    activation: Optional[str] = "tanh"
    t_emb_dim: int = 32
    t_emb_max_period: float = 10000.0
    t_emb_scaling: float = 100.0
    normalization_type: Optional[str] = "batch"
    dropout_rate: Optional[float] = 0.0
    dtype: Optional[jnp.dtype] = jnp.float32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, training: bool = True):
        t_emb = TimeEmbedding(
            t_emb_dim=self.t_emb_dim,
            max_period=self.t_emb_max_period,
            scaling=self.t_emb_scaling,
            dtype=self.dtype
        )(t)
        
        x = nn.Dense(self.t_emb_dim, dtype=self.dtype)(x)
        
        x = jnp.concatenate([t_emb, x], axis=-1)
        
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            if self.normalization_type == "batch":
                x = nn.BatchNorm(use_running_average=not training)(x)
            elif self.normalization_type == "layer":
                x = nn.LayerNorm()(x)
            elif self.normalization_type == None:
                pass
            x = get_activation(self.activation)(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

# Modified MLP from https://github.com/jeremyhengjm/DiffusionBridge/blob/master/DiffusionBridge/neuralnet.py
class MLPLayer(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: bool = False
    activation: str = "relu" 
    normalization_type: Optional[str] = None
    dropout_rate: Optional[float] = 0.0
    dtype: Optional[jnp.dtype] = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        activation_fn = get_activation(self.activation)
        
        for hidden_dim in self.hidden_dims[:-1]:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            x = activation_fn(x)
            if self.normalization_type == "batch":
                x = nn.BatchNorm(use_running_average=not training)(x)
            elif self.normalization_type == "layer":
                x = nn.LayerNorm()(x)
            elif self.normalization_type == None:
                pass
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dims[-1], dtype=self.dtype)(x)
        if self.activate_final:
            x = activation_fn(x)
            
        return x

class MLPLarge(nn.Module):
    out_dim: int
    encoder_dims: Sequence[int] = (16,)
    decoder_dims: Sequence[int] = (128, 128)
    t_emb_dim: int = 16
    t_emb_max_period: float = 10000.0
    t_emb_scaling: float = 100.0
    activation: str = "tanh"
    normalization_type: Optional[str] = None
    dropout_rate: Optional[float] = 0.0
    dtype: Optional[jnp.dtype] = jnp.float32

    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, training: bool):
        # Time embedding
        t_emb = TimeEmbedding(
            t_emb_dim=self.t_emb_dim,
            max_period=self.t_emb_max_period,
            scaling=self.t_emb_scaling,
            dtype=self.dtype
        )(t)  # size (N, t_emb_dim)

        # Time encoder
        t_emb = MLPLayer(
            hidden_dims=list(self.encoder_dims) + [2 * self.t_emb_dim],
            activate_final=False,
            activation=self.activation,
            normalization_type=self.normalization_type,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )(t_emb, training)  # size (N, 2 * t_emb_dim)

        # State encoder
        x_emb = MLPLayer(
            hidden_dims=list(self.encoder_dims) + [2 * self.t_emb_dim],
            activate_final=False,
            activation=self.activation,
            normalization_type=self.normalization_type,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )(x, training)  # size (N, 2 * t_emb_dim)

        # Concatenate embeddings
        h = jnp.concatenate([x_emb, t_emb], axis=-1)  # size (N, 4 * t_emb_dim)

        # Final network
        out = MLPLayer(
            hidden_dims=list(self.decoder_dims) + [self.out_dim],
            activate_final=False,
            activation=self.activation,
            normalization_type=self.normalization_type,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )(h, training)  # size (N, dimension)

        return out

class NetworkFactory:
    def __init__(self, config):
        self.config = config

    def create(self) -> nn.Module:
        mlp_type = self.config["mlp_type"]
        out_dim = self.config["out_dim"]
        activation = self.config["activation"]
        normalization_type = self.config["normalization_type"]
        dropout_rate = self.config["dropout_rate"]
        
        if mlp_type == "mlp_small":
            hidden_dims = self.config["hidden_dims"]
            return MLPSmall(
                out_dim=out_dim,
                hidden_dims=hidden_dims, 
                activation=activation,
                normalization_type=normalization_type,
                dropout_rate=dropout_rate,
            )
        elif mlp_type == "mlp":
            hidden_dims = self.config["hidden_dims"]
            t_emb_dim = self.config["t_emb_dim"]
            t_emb_max_period = self.config["t_emb_max_period"]
            t_emb_scaling = self.config["t_emb_scaling"]

            return MLP(
                out_dim=out_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                t_emb_dim=t_emb_dim,
                t_emb_max_period=t_emb_max_period,
                t_emb_scaling=t_emb_scaling,
                normalization_type=normalization_type,
                dropout_rate=dropout_rate,
            )
        elif mlp_type == "mlp_large":
            encoder_dims = self.config["encoder_dims"]
            decoder_dims = self.config["decoder_dims"]
            t_emb_dim = self.config["t_emb_dim"]
            t_emb_max_period = self.config["t_emb_max_period"]
            t_emb_scaling = self.config["t_emb_scaling"]
            
            return MLPLarge(
                out_dim=out_dim,
                encoder_dims=encoder_dims,
                decoder_dims=decoder_dims,
                t_emb_dim=t_emb_dim,
                t_emb_max_period=t_emb_max_period,
                t_emb_scaling=t_emb_scaling,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown MLP type: {mlp_type}")