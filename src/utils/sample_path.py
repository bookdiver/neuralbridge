from dataclasses import dataclass, field

import jax.numpy as jnp

@dataclass
class SamplePath:
    name: str = field(default_factory=str)
    path: dict = field(default_factory=dict)

    def __init__(self, name: str = "", **kwargs):
        self.name = name
        self.path = {}
        for key, value in kwargs.items():
            self.add(key, value)

    def __str__(self) -> str:
        info = [f"{self.name} sample path contains {self.n_steps} steps:"]
        for key, value in self.path.items():
            info.append(f"{key}.shape: {value.shape}")
        return "\n ".join(info)
    
    def __getitem__(self, idx: int) -> dict:
        # TODO: Doesn't work when only one value is stored in path, e.g. log likelihood
        assert idx < self.n_steps, f"Index out of range: {idx} > {self.n_steps}"
        return {key: value[idx] for key, value in self.path.items()}
    
    def __getattr__(self, key: str) -> jnp.ndarray:
        try:
            return self.path[key]
        except KeyError:
            raise AttributeError(f"{type(self).__name__} has no attribute '{key}'")
    
    @property
    def n_steps(self) -> int:
        try:
            return len(self.xs)
        except AttributeError:
            return 0
    
    def add(self, name: str, value: jnp.ndarray) -> None:
        assert isinstance(value, jnp.ndarray), "Only jnp.ndarray is allowed"
        self.path[name] = value

    def copy(self) -> "SamplePath":
        return SamplePath(self.name, **self.path)

    
    
    