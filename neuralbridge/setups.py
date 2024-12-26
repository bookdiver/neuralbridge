import os
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
from einops import repeat, rearrange
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

DEFAULT_DTYPE = jnp.float32
DEFAULT_SEED = 42
DEFAULT_COLOR_WHEELS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
    '#17becf'   # cyan
]
DEFAULT_CMAP_WHEELS = [
    'GnBu',
    'OrRd',
    'BrBG',
    'PuOr',
    'RdGy',
    'PiYG',
    'PRGn',
]
DEFAULT_FIGURE_SIZE = (10, 8)