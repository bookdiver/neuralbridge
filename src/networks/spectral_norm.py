'''
Spectral normalization code for flax.nnx from PR: https://github.com/google/flax/pull/4623
'''
import jax
import typing as tp
from jax import lax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Array,
  Dtype,
  Initializer,
  Axes,
)

def _l2_normalize(x, axis=None, eps=1e-12):
   """Normalizes along dimension `axis` using an L2 norm.
   This specialized function exists for numerical stability reasons.
   Args:
     x: An input ndarray.
     axis: Dimension along which to normalize, e.g. `1` to separately normalize
       vectors in a batch. Passing `None` views `t` as a flattened vector when
       calculating the norm (equivalent to Frobenius norm).
     eps: Epsilon to avoid dividing by zero.
   Returns:
     An array of the same shape as 'x' L2-normalized along 'axis'.
   """
   return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)

class SpectralNorm(Module):
    """Spectral normalization.
    See:
    - https://arxiv.org/abs/1802.05957
    - https://arxiv.org/abs/1805.08318
    - https://arxiv.org/abs/1809.11096
    Spectral normalization normalizes the weight params so that the spectral
    norm of the matrix is equal to 1. This is implemented as a layer wrapper
    where each wrapped layer will have its params spectral normalized before
    computing its ``__call__`` output.
    .. note::
        The initialized variables dict will contain, in addition to a 'params'
        collection, a separate 'batch_stats' collection that will contain a
        ``u`` vector and ``sigma`` value, which are intermediate values used
        when performing spectral normalization. During training, we pass in
        ``update_stats=True`` so that ``u`` and ``sigma`` are updated with
        the most recently computed values using power iteration. This will
        help the power iteration method approximate the true singular value
        more accurately over time. During eval, we pass in ``update_stats=False``
        to ensure we get deterministic behavior from the model.
    Example usage::
        >>> from flax import nnx
        >>> import jax
        >>> rngs = nnx.Rngs(0)
        >>> x = jax.random.normal(jax.random.key(0), (3, 4))
        >>> layer = nnx.SpectralNorm(nnx.Linear(4, 5, rngs=rngs),
        ...                          rngs=rngs)
        >>> nnx.state(layer, nnx.OfType(nnx.Param))
        State({
        'layer_instance': {
            'bias': VariableState( # 5 (20 B)
            type=Param,
            value=Array([0., 0., 0., 0., 0.], dtype=float32)
            ),
            'kernel': VariableState( # 20 (80 B)
            type=Param,
            value=Array([[ 0.5350889 , -0.48486355, -0.4022262 , -0.61925626, -0.46665004],
                    [ 0.31773907,  0.38944173, -0.54608804,  0.84378934, -0.93099   ],
                    [-0.67658   ,  0.0724705 , -0.6101737 ,  0.12972134,  0.877074  ],
                    [ 0.27292168,  0.32105306, -0.2556603 ,  0.4896752 ,  0.19558711]],      dtype=float32)
            )
        }
        })
        >>> y = layer(x, update_stats=True)
    Args:
        layer_instance: Module instance that is wrapped with SpectralNorm
        n_steps: How many steps of power iteration to perform to approximate the
        singular value of the weight params.
        epsilon: A small float added to l2-normalization to avoid dividing by zero.
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        error_on_non_matrix: Spectral normalization is only defined on matrices. By
        default, this module will return scalars unchanged and flatten
        higher-order tensors in their leading dimensions. Setting this flag to
        True will instead throw an error if a weight tensor with dimension greater
        than 2 is used by the layer.
        collection_name: Name of the collection to store intermediate values used
        when performing spectral normalization.
        rngs: The rng key.
    """

    def __init__(
        self,
        layer_instance: Module,
        *,
        n_steps: int = 1,
        epsilon: float = 1e-12,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        error_on_non_matrix: bool = False,
        collection_name: str = 'batch_stats',
        rngs: rnglib.Rngs,
    ):
        self.layer_instance = layer_instance
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.error_on_non_matrix = error_on_non_matrix
        self.collection_name = collection_name
        self.update_stats = False
        self.rngs = rngs

    def __call__(self, x, *args, update_stats: tp.Optional[bool] = None, **kwargs):
        """Compute the largest singular value of the weights in ``self.layer_instance``
        using power iteration and normalize the weights using this value before
        computing the ``__call__`` output.
        Args:
        x: the input array of the nested layer
        *args: positional arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
        update_stats: if True, update the internal ``u`` vector and ``sigma``
            value after computing their updated values using power iteration. This
            will help the power iteration method approximate the true singular value
            more accurately over time.
        **kwargs: keyword arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
        Returns:
        Output of the layer using spectral normalized weights.
        """
        if update_stats is not None:
            # If update_stats is explicitly passed as a __call__ argument, it takes precedence.
            effective_update_stats = update_stats
        else:
            # Otherwise, use the instance's current state (e.g., set by .train() or .eval()).
            effective_update_stats = self.update_stats
            
        state = nnx.state(self.layer_instance)  

        def spectral_normalize(path, vs):
            value = jnp.asarray(vs.value)
            value_shape = value.shape

            # Skip and return value if input is scalar, vector or if number of power
            # iterations is less than 1
            if value.ndim <= 1 or self.n_steps < 1:
                return value
            # Handle higher-order tensors.
            elif value.ndim > 2:
                if self.error_on_non_matrix:
                    raise ValueError(
                        f'Input is {value.ndim}D but error_on_non_matrix is set to True'
                    )
                else:
                    value = jnp.reshape(value, (-1, value.shape[-1]))

            u_var_name = (
                self.collection_name
                + '/'
                + '/'.join(str(k) for k in path)
                + '/u'
            )

            try:
                u = state[u_var_name].value
            except KeyError:
                u = jax.random.normal(
                self.rngs.params(),
                (1, value.shape[-1]),
                self.param_dtype,
                )

            sigma_var_name = (
                self.collection_name
                + '/'
                + '/'.join(str(k) for k in path)
                + '/sigma'
            )

            try:
                sigma = state[sigma_var_name].value
            except KeyError:
                sigma = jnp.ones((), self.param_dtype)

            for _ in range(self.n_steps):
                v = _l2_normalize(
                jnp.matmul(u, value.transpose([1, 0])), eps=self.epsilon
                )
                u = _l2_normalize(jnp.matmul(v, value), eps=self.epsilon)

            u = lax.stop_gradient(u)
            v = lax.stop_gradient(v)

            sigma = jnp.matmul(jnp.matmul(v, value), jnp.transpose(u))[0, 0]

            value /= jnp.where(sigma != 0, sigma, 1)
            value_bar = value.reshape(value_shape)

            if effective_update_stats: # Use the determined effective_update_stats
                state[u_var_name] = nnx.Param(u)
                state[sigma_var_name] = nnx.Param(sigma) # This is line 197

            dtype = dtypes.canonicalize_dtype(vs.value, u, v, sigma, dtype=self.dtype)
            return vs.replace(jnp.asarray(value_bar, dtype))

        state = nnx.map_state(spectral_normalize, state)
        nnx.update(self.layer_instance, state)

        return self.layer_instance(x, *args, **kwargs)  # type: ignore