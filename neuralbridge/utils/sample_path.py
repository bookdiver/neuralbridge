from dataclasses import dataclass, field
from typing import Dict, Union

import jax
import jax.numpy as jnp

@dataclass
class SamplePath:
    name: str = field(default="Default")
    path: Dict[str, jnp.ndarray] = field(default_factory=dict)
    
    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.path = {}
        for key, value in kwargs.items():
            self.add(key, value)

    def __str__(self) -> str:
        info = [f"{self.name} sample path contains {self.n_samples} samples，each sample runs {self.n_steps} steps:"]
        info.extend(f"{key}.shape: {value.shape}" for key, value in self.path.items())
        return "\n ".join(info)
    
    def __getitem__(self, idx: Union[int, slice]) -> "SamplePath":
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.n_samples)
            return SamplePath(self.name, **{
                key: value[start:stop:step, ...] if len(value.shape) > 1 else value 
                for key, value in self.path.items()
            })
        else:
            if idx < 0:
                idx += self.n_samples
            if idx >= self.n_samples or idx < 0:
                raise IndexError(f"Index out of range: {idx} >= {self.n_samples}")
            return SamplePath(self.name, **{
                key: value[idx:idx+1, ...] if len(value.shape) > 1 else value
                for key, value in self.path.items()
            })
    
    def __getattr__(self, key: str) -> jnp.ndarray:
        try:
            return self.path[key]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")
    
    @property
    def n_steps(self) -> int:
        return next(iter(self.path.values())).shape[1] if self.path else 0
        
    @property
    def n_samples(self) -> int:
        return next(iter(self.path.values())).shape[0] if self.path else 0
    
    def add(self, key: str, val: jnp.ndarray) -> None:
        self.path[key] = val

    def copy(self) -> "SamplePath":
        return SamplePath(self.name, {k: v.copy() for k, v in self.path.items()})
    
    def tree_flatten(self):
        return jax.tree_util.tree_flatten(self.path)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(name=aux_data, **jax.tree_util.tree_unflatten(aux_data, children))

jax.tree_util.register_pytree_node(SamplePath, SamplePath.tree_flatten, SamplePath.tree_unflatten)

    
    
    