from dataclasses import dataclass
from neuralbridge.setups import *

@dataclass
class TimeGrid:
    T: float
    dt: float
    t_scheme: Optional[str] = "linear"
    dtype: Optional[jnp.dtype] = DEFAULT_DTYPE
    
    def __post_init__(self):
        ts = jnp.arange(0.0, self.T + self.dt, self.dt, dtype=self.dtype)
        if self.t_scheme == "linear":
            self.ts = ts
        elif self.t_scheme == "quadratic":
            self.ts = ts * (2.0 - ts / self.T)
        else:
            raise ValueError(f"Time scheme: {self.t_scheme} not found!")
        self.dts = jnp.diff(self.ts)
        assert len(self.ts) == len(self.dts) + 1

    @property
    def n_steps(self):
        return len(self.dts)