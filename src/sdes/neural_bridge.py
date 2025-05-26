import jax
import jax.numpy as jnp
import jax.random as jr

from flax import nnx
from ..networks.mlps import *

class VectorField(nnx.Module):
    nn_config: dict
    nn: nnx.Module
    b_fn: callable
    sigma_fn: callable

    def __init__(self, b_fn, sigma_fn, nn_config):
        self.nn = self.create_model(nn_config)
        self.b_fn = b_fn
        self.sigma_fn = sigma_fn

    def __call__(self, t, x):
        b = self.b_fn(t, x)
        sigma = self.sigma_fn(t, x)
        v = self.nn(t, x)
        t = jnp.asarray(t)
        return b + jnp.dot(sigma.T, v), v
    
    def create_model(self, config):
        kwargs = {k: v for k, v in config.items() if k != "model_name"}
        if config["model_name"] == "MLPSmall":
            nn = MLPSmall(**kwargs)
        elif config["model_name"] == "MLPMedium":
            nn = MLPMedium(**kwargs)
        elif config["model_name"] == "MLPLarge":
            nn = MLPLarge(**kwargs)
        else:
            raise ValueError(f"Model name {config['model_name']} not recognized")
        return nn
    
class ControlVectorField(nnx.Module):
    sigma_fn: callable

    def __init__(self, sigma_fn):
        self.sigma_fn = sigma_fn

    def __call__(self, t, x):
        return self.sigma_fn(t, x)
    
class NeuralBridge(nnx.Module):
    nn: nnx.Module
    vf: VectorField
    cvf: ControlVectorField
    G_fn: callable
    dim_x: int
    dim_w: int
    rngs: nnx.Rngs
    
    def __init__(self, b_fn, sigma_fn, G_fn, nn_config):
        
        self.vf = VectorField(
            b_fn, sigma_fn, nn_config
        )
        self.cvf = ControlVectorField(sigma_fn)
        self.G_fn = G_fn
        self.dim_x = nn_config["input_dim"]
        self.dim_w = nn_config["output_dim"]
        self.rngs = nn_config["rngs"]
    
    def solve(self, ts, x0, key):
        dts = jnp.diff(ts)
        n_steps = len(dts)
        dWs = jr.normal(key, (n_steps, self.dim_w)) * jnp.sqrt(dts[:, None])
        
        def scan_body(carry, vals):
            t, x, ll = carry
            dt, dW = vals
            drift, v_new = self.vf(t, x)
            x_new = x + drift * dt + jnp.dot(self.cvf(t, x), dW)
            ll_new = ll + self.G_fn(t, x) * dt
            t_new = t + dt
            return (t_new, x_new, ll_new), (x_new, v_new)
        
        (_, _, ll), (xs, vs) = jax.lax.scan(
            scan_body,
            init=(0.0, x0, 0.0),
            xs=(dts, dWs),
            length=n_steps
        )
        return xs, (vs, ll)
    
    @property
    def nn(self):
        return self.vf.nn
    
    @nn.setter
    def nn(self, value):
        self.vf.nn = value
        
    # def train(self, **attributes):
    #     return self.set_attributes(
    #         deterministic=False,
    #         use_running_average=False,
    #         update_stats=True,
    #         **attributes,
    #         raise_if_not_found=False,
    #     )
    # def eval(self, **attributes):
    #     return self.set_attributes(
    #         deterministic=True,
    #         use_running_average=True,
    #         update_stats=False,
    #         **attributes,
    #         raise_if_not_found=False,
    #     )
    
    def b(self, t, x):
        return self.vf(t, x)[0]
    
    def sigma(self, t, x):
        return self.cvf(t, x)