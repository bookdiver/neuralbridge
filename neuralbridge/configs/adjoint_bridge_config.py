import jax.numpy as jnp

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

def get_adjoint_bridge_ou_config():
    config = {
        "sde": {
            "dim": 1,
            "T": 1.0,
            "N": 200,
            "alpha": 1.0, # alpha here is the gamma in the paper
            "sigma": 1.0
        },
        "training": {
            "y": jnp.array([0.0]),
            "batch_size": 100,
            "epochs_per_load": 1,
            "lr": 0.01,
            "num_reloads": 100,
            "load_size": 1000
        },
        "network": {
            "output_dim": 1,
            "time_embedding_dim": 32,
            "init_embedding_dim": 16,
            "activation": "leaky_relu",
            "encoder_layer_dims": [16],
            "decoder_layer_dims": [128, 128],
            "batch_norm": False
        }
    }
    return config

def get_adjoint_bridge_cell_normal_config():
    config = {
        "sde": {
            "T": 4.0,
            "N": 400,
            "alpha": 1./ 16.,
            "sigma": 0.1
        },
        "training": {
            "y": jnp.array([2.0, -0.1]),
            "batch_size": 100,
            "epochs_per_load": 1,
            "lr": 0.01,
            "num_reloads": 100,
            "load_size": 10000
        },
        "network": {
            "output_dim": 2,
            "time_embedding_dim": 32,
            "init_embedding_dim": 16,
            "activation": "leaky_relu",
            "encoder_layer_dims": [16],
            "decoder_layer_dims": [128, 128],
        }
    }
    return config

def get_adjoint_bridge_landmark_config():
    config = {
        "sde": {
            "T": 1.0,
            "N": 100,
            "alpha": 0.3,
            "sigma": 0.5,
            
        }
    }