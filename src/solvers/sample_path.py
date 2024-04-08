from dataclasses import dataclass, field

import jax.numpy as jnp

@dataclass
class SamplePath:
    name: str = field(default_factory=str)
    ts: jnp.ndarray = field(default_factory=lambda: jnp.array([]))
    xs: jnp.ndarray = field(default_factory=lambda: jnp.array([]))

    def __init__(self, ts: jnp.ndarray, xs: jnp.ndarray, name: str = ""):
        self.ts = ts
        self.xs = xs

        self.name = name

    def __str__(self):
        info = [f"{self.name} sample path contains {self.n_steps} steps:"]
        for name, value in self.__dict__.items():
            info.append(f"{name}.shape: {value.shape if isinstance(value, jnp.ndarray) else value}")
        return "\n ".join(info)
    
    @property
    def n_steps(self):
        return len(self.ts)
    
    def add(self, name: str, value: any):
        setattr(self, name, value)

    
    
    