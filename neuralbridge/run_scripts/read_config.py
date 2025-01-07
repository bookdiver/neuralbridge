from omegaconf import OmegaConf

from neuralbridge.setups import *

def ellipse(n_landmarks: int, a: float = 1.0, b: float = 1.0, rotation: float = 0.0, shift_x: float = 0.0, shift_y: float = 0.0):
    u = jnp.linspace(0.0, 2.0*jnp.pi, num=n_landmarks, endpoint=False)
    x = a * jnp.sin(u)
    y = b * jnp.cos(u)
    x_rot = x * jnp.cos(rotation) - y * jnp.sin(rotation) 
    y_rot = x * jnp.sin(rotation) + y * jnp.cos(rotation)
    x_shift = x_rot + shift_x
    y_shift = y_rot + shift_y
    u = jnp.stack([x_shift, y_shift], axis=-1)
    return u

def read_config(config_path):
    config = OmegaConf.load(config_path)
    if 'sde' in config:
        if 'n_landmarks' in config.sde:
            config.sde.u = ellipse(
                config.sde.u.n_landmarks,
                config.sde.u.a,
                config.sde.u.b,
                config.sde.u.rotation,
                config.sde.u.shift_x,
                config.sde.u.shift_y
            ).flatten().tolist()
            config.sde.v = ellipse( 
                config.sde.v.n_landmarks,
                config.sde.v.a,
                config.sde.v.b,
                config.sde.v.rotation,
                config.sde.v.shift_x,
                config.sde.v.shift_y
            ).flatten().tolist()
        else:
            u_values = [float('nan') if x.lower() == 'nan' else float(x) 
                        for x in config.sde.u]
            v_values = [float('nan') if x.lower() == 'nan' else float(x) 
                        for x in config.sde.v]
            config.sde.u = u_values
            config.sde.v = v_values
        
    return config