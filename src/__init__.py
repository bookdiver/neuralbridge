"""
Neural Guided Diffusion Bridge package.

A JAX/Flax implementation of neural guided diffusion bridges for conditional 
stochastic process simulation, particularly excelling in rare event conditioning
and multi-modal scenarios.
"""

__version__ = "0.2.0"
__author__ = "Gefan Yang"
__email__ = "gy@di.ku.dk"

__all__ = [
    "sde_examples", 
    "base",
    "neural_bridge",
    "sde_solver",
    "pcn"
]
