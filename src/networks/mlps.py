import jax.numpy as jnp
from flax import nnx

from .time_embeddings import (
    SinusoidalTimeEmbedding,
    GaussianRandomTimeEmbedding,
    TimeEmbeddingFiLM
)

from .spectral_norm import SpectralNorm

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
    
def get_time_embedding_layer(t_emb_type, t_emb_kwargs):
    if t_emb_type == "sinusoidal":
        return SinusoidalTimeEmbedding(**t_emb_kwargs)
    elif t_emb_type == "random":
        return GaussianRandomTimeEmbedding(**t_emb_kwargs)
    else:
        raise ValueError(f"Time embedding {t_emb_type} not supported")
    

class DenseLayer(nnx.Module):
    """ A dense layer with optional normalization and activation """
    fc: nnx.Module
    norm: nnx.Module
    act: nnx.Module
    
    def __init__(self, input_dim, output_dim, *, act_type, rngs):
        # self.fc = SpectralNorm(nnx.Linear(input_dim, output_dim, rngs=rngs), rngs=rngs)
        self.fc = nnx.Linear(input_dim, output_dim, rngs=rngs)
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
        # self.in_fc = SpectralNorm(nnx.Linear(input_dim + 1, hidden_dims[0], rngs=rngs), rngs=rngs)
        self.in_fc = nnx.Linear(input_dim + 1, hidden_dims[0], rngs=rngs)
        self.layers = []
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(
                DenseLayer(
                    hidden_dims[i - 1], hidden_dims[i], 
                    act_type=act_type, 
                    rngs=rngs
                )
            )
            
        self.out_fc = nnx.Linear(hidden_dims[-1], output_dim, rngs=rngs)
        
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
    
    def __init__(self, input_dim, output_dim, hidden_dims, *, t_emb_type, t_emb_kwargs, act_type, norm_type, rngs):
        self.t_emb_layer = get_time_embedding_layer(t_emb_type, t_emb_kwargs)
        self.layers = []
        self.layers.append(
            nnx.Linear(int(input_dim + t_emb_kwargs["embed_dim"]), hidden_dims[0], rngs=rngs)
        )
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(
                DenseLayer(hidden_dims[i - 1], hidden_dims[i], act_type=act_type, norm_type=norm_type, rngs=rngs)
            )
            
        self.out_fc = nnx.Linear(hidden_dims[-1], output_dim, rngs=rngs)
        
    def __call__(self, t, x):
        """
        input t shape: (batch_size, )
        input x shape: (batch_size, x_dim)
        
        output shape: (batch_size, out_dim)
        """
        t_emb = self.t_emb_layer(t)
        
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
    
    def __init__(self, input_dim, output_dim, hidden_dims, *, t_emb_type, t_emb_kwargs, act_type, norm_type, rngs):
        self.t_emb_layer = get_time_embedding_layer(t_emb_type, t_emb_kwargs)
        self.layers = []
        self.t_emb_film_layers = []
        self.in_fc = nnx.Linear(input_dim, hidden_dims[0], rngs=rngs)
        
        for i in range(1, len(hidden_dims)):
            self.layers.append(
                DenseLayer(hidden_dims[i - 1], hidden_dims[i], act_type=act_type, norm_type=norm_type, rngs=rngs)
            )
            
            self.t_emb_film_layers.append(
                TimeEmbeddingFiLM(t_emb_kwargs["embed_dim"], hidden_dims[i], rngs=rngs)
            )
            
        self.out_fc = nnx.Linear(hidden_dims[-1], output_dim, rngs=rngs)
        
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