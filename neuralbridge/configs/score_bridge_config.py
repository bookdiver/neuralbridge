import jax.numpy as jnp

def get_score_bridge_ou_config():
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
                "sigma": 1.0,
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
            "mlp_type": "mlp_large",
            "out_dim": 2,
            "encoder_dims": (16,),
            "decoder_dims": (128, 128),
            "t_emb_dim": 16,
            "t_emb_max_period": 10000.0,
            "t_emb_scaling": 100.0,
            "activation": "tanh",
            "normalization_type": None,
            "dropout_rate": 0.0,
        },
        "training": {
            "save_name": "ou_benchmark",
            "learning_rate": 0.01,
            "batch_size": 100,
            "n_iters_per_epoch": 500,
            "n_epochs": 10,
            "ema_decay": 0.99,
            "optimizer": "adam",
            "warmup_steps_ratio": 0.0,
            "clip_norm": None
        }
    }
    return config

def get_score_bridge_cell_normal_config():
    config = {
        "sde": {
            "name": "cell",
            "X_dim": 2,
            "W_dim": 2,
            "T": 4.0,
            "dt": 1e-2,
            "t_scheme": "linear",
            "params_X_unc": {
                "alpha": 1. / 16.,
                "sigma": 0.1
            },
            "params_X_aux": {
                "alpha": 1. / 16.,
                "sigma": 0.1
            },
            "u": jnp.array([0.1, -0.1]),
            "v": jnp.array([2.0, -0.1])
        },
        "network": {
            "mlp_type": "mlp_large",
            "out_dim": 2,
            "encoder_dims": (16,),
            "decoder_dims": (128, 128),
            "t_emb_dim": 16,
            "t_emb_max_period": 10000.0,
            "t_emb_scaling": 100.0,
            "activation": "leaky_relu",
            "normalization_type": None,
            "dropout_rate": 0.0,
        },
        "training": {
            "save_name": "cell_benchmark",
            "learning_rate": 0.01,
            "batch_size": 100,
            "n_iters_per_epoch": 1000,
            "n_epochs": 10,
            "ema_decay": 0.99,
            "optimizer": "adam",
            "warmup_steps_ratio": 0.0,
            "clip_norm": None
        }
    }
    return config
    