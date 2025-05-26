import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

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
    
class GaussianRandomTimeEmbedding(nnx.Module):
    """ Gaussian random features time step embedding """
    embed_dim: int
    scale: float
    freqs: nnx.Param
    
    def __init__(self, embed_dim, scale, init_std, *, rngs):
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.scale = scale
        num_freqs = embed_dim // 2
        self.freqs = nnx.Param(
            jr.normal(rngs.t_emb(), (num_freqs)) * init_std
        )
        self.freqs = jax.lax.stop_gradient(self.freqs) # freeze the frequencies
        
    def __call__(self, t):
        """ input t shape: (batch, ) """
        if t.ndim == 1:
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