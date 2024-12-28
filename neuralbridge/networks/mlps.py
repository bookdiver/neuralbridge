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
    dtype: Optional[jnp.dtype] = jnp.float32
    
    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray):
        
        x = jnp.concatenate([t, x], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, dtype=self.dtype)(x)
            x = get_activation(self.activation)(x)
        x = nn.Dense(self.out_dim, dtype=self.dtype)(x)
        return x

class MLP(nn.Module):
    out_dim: int
    hidden_dims: Sequence[int]
    activation: Optional[str] = "tanh"
    t_emb_dim: int = 32
    t_emb_max_period: float = 100.0
    t_emb_scaling: float = 100.0
    norm: Optional[str] = "batch"
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

# Modified MLP from https://github.com/jeremyhengjm/DiffusionBridge/blob/master/DiffusionBridge/neuralnet.py
class ScoreLayer(nn.Module):
    input_dim: int
    layer_widths: Sequence[int]
    activate_final: bool = False
    activation: str = "relu" 
    dtype: Optional[jnp.dtype] = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        activation_fn = get_activation(self.activation)
        
        for i, width in enumerate(self.layer_widths[:-1]):
            x = nn.Dense(width, dtype=self.dtype)(x)
            x = activation_fn(x)
            
        x = nn.Dense(self.layer_widths[-1], dtype=self.dtype)(x)
        if self.activate_final:
            x = activation_fn(x)
            
        return x

class ScoreNetwork(nn.Module):
    dimension: int
    encoder_layers: Sequence[int] = (16,)
    pos_dim: int = 16
    decoder_layers: Sequence[int] = (128, 128)
    activation: str = "leaky_relu"
    dtype: Optional[jnp.dtype] = jnp.float32

    @nn.compact
    def __call__(self, t: jnp.ndarray, x: jnp.ndarray, training: bool = True):
        """
        Parameters
        ----------
        t : time step (N, 1)
        x : state (N, dimension)
                        
        Returns
        -------    
        out :  score (N, dimension)
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x[None, :]

        t_emb_dim = self.pos_dim * 2

        # Time embedding
        t_emb = TimeEmbedding(
            t_emb_dim=t_emb_dim,
            scaling=100.0,
            max_period=10000.0
        )(t)  # size (N, t_emb_dim)

        # Time encoder
        t_emb = ScoreLayer(
            input_dim=self.pos_dim,
            layer_widths=list(self.encoder_layers) + [t_emb_dim],
            activate_final=False,
            activation=self.activation,
            dtype=self.dtype
        )(t_emb)  # size (N, t_enc_dim)

        # State encoder
        x_emb = ScoreLayer(
            input_dim=self.dimension,
            layer_widths=list(self.encoder_layers) + [t_emb_dim],
            activate_final=False,
            activation=self.activation,
            dtype=self.dtype
        )(x)  # size (N, t_enc_dim)

        # Concatenate embeddings
        h = jnp.concatenate([x_emb, t_emb], axis=-1)  # size (N, 2 * t_enc_dim)

        # Final network
        out = ScoreLayer(
            input_dim=2 * t_emb_dim,
            layer_widths=list(self.decoder_layers) + [self.dimension],
            activate_final=False,
            activation=self.activation,
            dtype=self.dtype
        )(h)  # size (N, dimension)

        return out

class NetworkFactory:
    def __init__(self, config: dict):
        self.config = config
        
    def create(self) -> nn.Module:
        mlp_type = self.config.get("mlp_type", "mlp")
        out_dim = self.config.get("out_dim")
        hidden_dims = self.config.get("hidden_dims")
        activation = self.config.get("activation", "tanh")
        dtype = self.config.get("dtype", jnp.float32)
        
        if mlp_type == "mlp_small":
            return MLPSmall(
                out_dim=out_dim,
                hidden_dims=hidden_dims, 
                activation=activation,
                dtype=dtype
            )
        elif mlp_type == "mlp":
            t_emb_dim = self.config.get("t_emb_dim", 32)
            t_emb_max_period = self.config.get("t_emb_max_period", 100.0)
            t_emb_scaling = self.config.get("t_emb_scaling", 100.0)
            norm = self.config.get("norm", "batch")
            
            return MLP(
                out_dim=out_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                t_emb_dim=t_emb_dim,
                t_emb_max_period=t_emb_max_period,
                t_emb_scaling=t_emb_scaling,
                norm=norm,
                dtype=dtype
            )
        elif mlp_type == "score":
            pos_dim = self.config.get("pos_dim", 32)
            encoder_layers = self.config.get("encoder_layers", (128, 128))
            decoder_layers = self.config.get("decoder_layers", (128, 128))
            
            return ScoreNetwork(
                dimension=out_dim,
                pos_dim=pos_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                activation=activation,
                dtype=dtype
            )
        else:
            raise ValueError(f"Unknown MLP type: {mlp_type}")