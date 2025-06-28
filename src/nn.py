import jax
import jax.numpy as jnp
from flax import nnx


def get_activation(act_type):
    if act_type == "relu":
        return nnx.relu
    elif act_type == "tanh":
        return nnx.tanh
    elif act_type == "gelu":
        return nnx.gelu
    elif act_type == "silu":
        return nnx.silu
    elif act_type == "leaky_relu":
        return nnx.leaky_relu
    elif act_type == "lipswish":
        return lambda x: 0.909 * nnx.swish(x)
    else:
        raise ValueError(f"Activation {act_type} not supported")
    
    
class SinusoidalTimeEmbedding(nnx.Module):
    """ Sinusoidal time step embedding """
    embed_dim: int
    scale: float
    freqs: nnx.Param
    
    def __init__(self, embed_dim, scale, min_freq, max_freq, *, rngs):
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.scale = scale
        num_freqs = embed_dim // 2
        self.freqs = nnx.Param(
            2.0 * jnp.pi * jnp.exp(
            jnp.linspace(
                jnp.log(min_freq),
                jnp.log(max_freq),
                num_freqs,
            ))
        )
        self.freqs = jax.lax.stop_gradient(self.freqs) # freeze the frequencies

    def __call__(self, t):
        """ input t shape: (batch, ) """
        t = jnp.expand_dims(t, axis=0)
        t = self.scale * t[:, jnp.newaxis] # (batch_size, 1)
        arg = t * self.freqs  # (batch_size, num_freqs)
        sin_emb = jnp.sin(arg)
        cos_emb = jnp.cos(arg)
        embedding = jnp.stack([sin_emb, cos_emb], axis=-1) # (batch_size, num_freqs, 2)
        embedding = embedding.reshape(-1, self.embed_dim)  # (batch_size, embed_dim)
        return embedding
    
class TimeEmbeddingFiLM(nnx.Module):
    """ Feature-wise linear modulation from time embedding """
    fc: nnx.Module
    
    def __init__(self, input_dim, output_dim, *, rngs):
        self.fc = nnx.Linear(
            input_dim,
            int(2 * output_dim),
            kernel_init=nnx.initializers.normal(stddev=0.01),
            bias_init=nnx.initializers.zeros,
            rngs=rngs
        )
    
    def __call__(self, t_emb):
        scale_shift = self.fc(t_emb)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        return scale, shift

class DenseLayer(nnx.Module):
    """ A dense layer with optional normalization and activation """
    fc: nnx.Module
    norm: nnx.Module
    act: nnx.Module
    
    def __init__(self, input_dim, output_dim, *, act_type, rngs):
        self.fc = nnx.Linear(
            input_dim, 
            output_dim, 
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal()
        )
        self.act = get_activation(act_type)
        
    def __call__(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x
    
class MLPSmall(nnx.Module):
    """ A simple fully connected MLP without time embedding """
    layers: list[nnx.Module]
    in_fc: nnx.Module
    out_fc: nnx.Module
    
    def __init__(self, input_dim, output_dim, hidden_dims, *, act_type, rngs):
        self.in_fc = nnx.Linear(
            input_dim + 1, 
            hidden_dims[0], 
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal()
        )
        self.layers = []
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(
                DenseLayer(
                    hidden_dims[i - 1], hidden_dims[i], 
                    act_type=act_type, 
                    rngs=rngs
                )
            )
            
        self.out_fc = nnx.Linear(
            hidden_dims[-1], 
            output_dim, 
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal()
        )
        
    def __call__(self, t, x):
        t_expand = jnp.expand_dims(t, axis=-1)
        tx = jnp.concatenate([t_expand, x], axis=-1)
        tx = self.in_fc(tx)
        for layer in self.layers:
            tx = layer(tx)
        out = self.out_fc(tx)
        return out

class MLPMedium(nnx.Module):
    """ A fully connected MLP with time embedding """
    layers: list[nnx.Module]
    t_emb_layer: nnx.Module
    out_fc: nnx.Module
    
    def __init__(self, input_dim, output_dim, hidden_dims, *, t_emb_type, t_emb_kwargs, act_type, rngs):
        self.t_emb_layer = get_time_embedding_layer(t_emb_type, t_emb_kwargs)
        self.layers = []
        self.layers.append(
            nnx.Linear(
                int(input_dim + t_emb_kwargs["embed_dim"]), 
                hidden_dims[0], 
                rngs=rngs,
                kernel_init=nnx.initializers.kaiming_normal()
            )
        )
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(
                DenseLayer(hidden_dims[i - 1], hidden_dims[i], act_type=act_type, rngs=rngs)
            )
            
        self.out_fc = nnx.Linear(
            hidden_dims[-1], 
            output_dim, 
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal()   
        )
        
    def __call__(self, t, x):
        """
        input t shape: (batch_size, )
        input x shape: (batch_size, x_dim)
        
        output shape: (batch_size, out_dim)
        """
        t_emb = self.t_emb_layer(t)
        t_emb = t_emb.squeeze()
        
        x = jnp.concatenate([t_emb, x], axis=-1)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.out_fc(x)
        return x

class MLPLarge(nnx.Module):
    """ A fully connected MLP with feature-wise linear modulation from time embedding """
    layers: list[nnx.Module]
    in_fc: nnx.Module
    t_emb_layer: nnx.Module
    t_emb_film_layers: list[nnx.Module]
    out_fc: nnx.Module
    
    def __init__(self, input_dim, output_dim, hidden_dims, *, t_emb_type, t_emb_kwargs, act_type, rngs):
        self.t_emb_layer = get_time_embedding_layer(t_emb_type, t_emb_kwargs)
        self.layers = []
        self.t_emb_film_layers = []
        self.in_fc = nnx.Linear(
            input_dim, 
            hidden_dims[0], 
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal()
        )
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(
                DenseLayer(hidden_dims[i - 1], hidden_dims[i], act_type=act_type, rngs=rngs)
            )
            
            self.t_emb_film_layers.append(
                TimeEmbeddingFiLM(t_emb_kwargs["embed_dim"], hidden_dims[i], rngs=rngs)
            )
            
        self.out_fc = nnx.Linear(
            hidden_dims[-1], 
            output_dim, 
            rngs=rngs,
            kernel_init=nnx.initializers.kaiming_normal
        )
        
    def __call__(self, t, x):
        """
        input t shape: (batch_size, )
        input x shape: (batch_size, x_dim)
        
        output shape: (batch_size, out_dim)
        """
        t_emb = self.t_emb_layer(t)
        x = self.in_fc(x)
        
        for layer, film_layer in zip(self.layers, self.t_emb_film_layers):
            x = layer(x)
            t_emb_scale, t_emb_shift = film_layer(t_emb)
            x = x * t_emb_scale + t_emb_shift
            
        x = self.out_fc(x)
        return x