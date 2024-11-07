import os
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('absl').setLevel(logging.WARNING)

from typing import Tuple, Union, Optional, Callable, Sequence, Generator
from functools import partial
from collections import namedtuple
import abc

import numpy as np
import jax
import jax.numpy as jnp
from einops import repeat, rearrange