from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .time_embedding import TimeEmbeddingMLP, get_time_embedding


class ScoreUNet(nn.Module):
    output_dim: int
    time_embedding_dim: int
    init_embedding_dim: int
    activation: str
    encoder_layer_dims: Sequence[int]
    decoder_layer_dims: Sequence[int]
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, train: bool) -> jnp.ndarray:
        assert self.encoder_layer_dims[-1] == self.decoder_layer_dims[0], "Bottleneck dim does not match"
        x_init = x
        time_embedding = get_time_embedding(self.time_embedding_dim)
        t = jax.vmap(time_embedding, in_axes=0)(t)
        x = x.reshape((x.shape[0], -1))  # !!! Flatten happens here
        x = InputDense(self.init_embedding_dim, self.activation)(x)

        # downsample
        downs = []
        for dim in self.encoder_layer_dims:
            x = Downsample(
                output_dim=dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
            )(x, t, train)
            downs.append(x)

        # bottleneck
        bottleneck_dim = self.encoder_layer_dims[-1]
        x_out = Dense(bottleneck_dim)(x)
        x = _get_activation(self.activation)(x_out) + x

        # upsample
        for dim, x_skip in zip(self.decoder_layer_dims, downs[::-1]):
            x = Upsample(
                output_dim=dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
            )(x, x_skip, t, train)

        # out
        score = Dense(self.output_dim)(x)
        score = score.reshape(x_init.shape)
        return score


class Upsample(nn.Module):
    output_dim: int
    activation: str
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, x_skip: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        x_in = jnp.concatenate([x, x_skip], axis=-1)
        input_dim = x_in.shape[-1]

        x = Dense(input_dim)(x_in)

        scale, shift = TimeEmbeddingMLP(input_dim, _get_activation(self.activation))(t_emb)
        x = x * (1.0 + scale) + shift
        x = _get_activation(self.activation)(x) + x_in
        x = Dense(self.output_dim)(x)
        x = _get_activation(self.activation)(x)
        if self.batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x


class Downsample(nn.Module):
    output_dim: int
    activation: str
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray, train: bool) -> jnp.ndarray:
        input_dim = x.shape[-1]
        x_in = x.copy()
        x = Dense(input_dim)(x)

        scale, shift = TimeEmbeddingMLP(input_dim, _get_activation(self.activation))(t_emb)
        x = x * (1.0 + scale) + shift
        x = _get_activation(self.activation)(x) + x_in
        x = Dense(self.output_dim)(x)
        x = _get_activation(self.activation)(x)
        if self.batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = nn.LayerNorm()(x)
        return x


class InputDense(nn.Module):
    output_dims: int
    activation: str
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.output_dims, kernel_init=self.kernel_init)(x)
        x = _get_activation(self.activation)(x)
        return x


class Dense(nn.Module):
    output_dims: int
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.output_dims, kernel_init=self.kernel_init)(x)
        return x


def _get_activation(name: str) -> nn.activation:
    if name == "relu":
        return nn.relu
    elif name == "leaky_relu":
        return nn.leaky_relu
    elif name == "elu":
        return nn.elu
    elif name == "gelu":
        return nn.gelu
    elif name == "silu":
        return nn.silu
    elif name == "tanh":
        return nn.tanh
    elif name == "sigmoid":
        return nn.sigmoid
    elif name == "none":
        return lambda x: x
    else:
        raise ValueError(f"Activation {name} not recognized.")
