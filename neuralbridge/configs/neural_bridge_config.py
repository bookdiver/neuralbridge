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

def get_neural_bridge_ou_config():
    config = {
        "sde": {
            "name": "ou",
            "X_dim": 1,
            "W_dim": 1,
            "T": 1.0,
            "dt": 5e-3,
            "t_scheme": "linear",
            "params_X_unc": {
                "gamma": 1.0,
                "sigma": 1.0
            },
            "params_X_aux": {
                "gamma": 0.0,
                "sigma": 1.0
            },
            "u": jnp.array([0.0]),
            "v": jnp.array([0.0]),
            "eps": 1e-10
        },
        "network": {
            "mlp_type": "mlp_small",
            "out_dim": 1,
            "hidden_dims": (20, 20, 20, 20),
            "activation": "tanh",
            "normalization_type": None,
            "dropout_rate": 0.0,
        },
        "training": {
            "save_name": "ou_benchmark",
            "learning_rate": 0.001,
            "batch_size": 20,
            "n_iters_per_epoch": 400,
            "n_epochs": 10,
            "ema_decay": 0.995,
            "optimizer": "adam",
            "warmup_steps_ratio": 0.1,
            "clip_norm": 1.0
        }
    }
    return config

def get_neural_bridge_cell_normal_config():
    config = {
        "sde": {
            "name": "cell",
            "X_dim": 2,
            "W_dim": 2,
            "T": 4.0,
            "dt": 1e-2,
            "t_scheme": "linear",
            "params_X_unc": {
                "alpha": 0.0625,
                "sigma": 0.1,
            },
            "params_X_aux": {
                "alpha": 0.0625,
                "sigma": 0.1,
            },
            "u": jnp.array([0.1, -0.1]),
            "v": jnp.array([2.0, -0.1]),
            "eps": 1e-10
        },
        "network": {
            "mlp_type": "mlp_small",
            "out_dim": 2,
            "hidden_dims": (32, 32, 32, 32),
            "activation": "tanh",
            "normalization_type": None,
            "dropout_rate": 0.0,
        },
        "training": {
            "save_name": "cell_benchmark",
            "learning_rate": 5.0e-4,
            "batch_size": 20,
            "n_iters_per_epoch": 500,
            "n_epochs": 10,
            "ema_decay": 0.995,
            "optimizer": "adam",
            "warmup_steps_ratio": 0.1,
            "clip_norm": 1.0
        }
    }
    return config

def get_neural_bridge_landmark_config():
    config = {
        "sde": {
            "name": "landmark",
            "n_landmarks": 50,
            "T": 1.0,
            "dt": 0.01,
            "X_dim": 100,
            "W_dim": 100,
            "t_scheme": "linear",
            "params_X_unc": {
                "k_alpha": 0.3,
                "k_sigma": 0.5,
                "n_landmarks": 50,
                "m_landmarks": 2
            },
            "params_X_aux": {
                "k_alpha": 0.3,
                "k_sigma": 0.5,
                "n_landmarks": 50,
                "m_landmarks": 2
            },
            "u": ellipse(50, 1.0, 0.5, 0.0, 0.0, 0.0).flatten(),
            "v": ellipse(50, 1.0, 0.5, jnp.pi / 3, -1.0, 0.5).flatten(),
            "eps": 2e-3
        },
        "network": {
            "mlp_type": "mlp_large",
            "out_dim": 100,
            "encoder_dims": (128,),
            "decoder_dims": (256, 256, 128),
            "t_emb_dim": 32,
            "t_emb_max_period": 100.0,
            "t_emb_scaling": 100.0,
            "activation": "tanh",
            "normalization_type": None,
            "dropout_rate": 0.0,
        },
        "training": {
            "save_name": "landmark_benchmark",
            "learning_rate": 7.0e-4,
            "batch_size": 8,
            "n_iters_per_epoch": 1500,
            "n_epochs": 1,
            "ema_decay": 0.995,
            "optimizer": "adam",
            "warmup_steps_ratio": 0.1,
            "clip_norm": 1.0
        }
    }
    return config
