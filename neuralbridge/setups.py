import os
import pdb
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)

from typing import List, Tuple, Dict, Union, Optional, Callable, Sequence, Generator, Any
from functools import partial
from collections import namedtuple
import abc

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from einops import repeat, rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

tTYPE = Union[float, jnp.ndarray]
xTYPE = Union[jnp.ndarray]

DEFAULT_DTYPE = jnp.float32

if DEFAULT_DTYPE == jnp.float64:
    jax.config.update("jax_enable_x64", True)
    
DEFAULT_RNG_KEY = jax.random.PRNGKey(42)