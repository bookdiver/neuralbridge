from dataclasses import dataclass, field
from typing import Dict

import jax
import jax.numpy as jnp

@dataclass
class SamplePath:
    name: str = field(default="")
    path: Dict[str, jnp.ndarray] = field(default_factory=dict)
    
    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.path = {}
        for key, value in kwargs.items():
            self.add(key, value)

    def __str__(self) -> str:
        info = [f"{self.name} sample path contains {self.n_batches} samples，each sample runs {self.n_steps} steps:"]
        info.extend(f"{key}.shape: {value.shape}" for key, value in self.path.items())
        return "\n ".join(info)
    
    def __getitem__(self, idx: int) -> Dict[str, jnp.ndarray]:
        if idx >= self.n_steps:
            raise IndexError(f"Index out of range: {idx} >= {self.n_steps}")
        return {key: value[:, idx] for key, value in self.path.items()}
    
    def __getattr__(self, key: str) -> jnp.ndarray:
        try:
            return self.path[key]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")
    
    @property
    def n_steps(self) -> int:
        return next(iter(self.path.values())).shape[1] if self.path else 0
        
    @property
    def n_batches(self) -> int:
        return next(iter(self.path.values())).shape[0] if self.path else 0
    
    def add(self, key: str, val: jnp.ndarray) -> None:
        if not isinstance(val, jnp.ndarray):
            raise ValueError(f"Only jnp.ndarray is allowed, but received {type(val)}")
        self.path[key] = val

    def copy(self) -> "SamplePath":
        return SamplePath(self.name, {k: v.copy() for k, v in self.path.items()})
    
    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.path)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(name=aux_data, **jax.tree_util.tree_unflatten(aux_data, children))

jax.tree_util.register_pytree_node(SamplePath, SamplePath.tree_flatten, SamplePath.tree_unflatten)

    
    
    