import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import jax
import jax.numpy as jnp
import optax

from neuralbridge.setups import *
from neuralbridge.configs.adjoint_bridge_config import *

from legacies.adjoint_forward.models.score_mlp import ScoreMLP
from legacies.adjoint_forward.models.score_unet import ScoreUNet
from legacies.adjoint_forward.sdes import (
    sde_ornstein_uhlenbeck, sde_cell_model, sde_kunita, sde_data
)
from legacies.adjoint_forward.training import train_loop, train_utils

main_rng_key = DEFAULT_RNG_KEY

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="cell_normal", choices=["ou", "cell_normal", "cell_rare", "cell_mm"])

def get_model(model_name):
    if "ou" in model_name.lower():
        sde_cls = sde_ornstein_uhlenbeck.ornstein_uhlenbeck
    elif "cell" in model_name.lower():
        sde_cls = sde_cell_model.cell_model
    elif "landmark" in model_name.lower():
        sde_cls = sde_kunita
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return sde_cls

def get_config(model_name):
    if model_name == "ou":
        return get_adjoint_bridge_ou_config()
    elif model_name == "cell_normal":
        return get_adjoint_bridge_cell_normal_config()
    elif model_name == "landmark_ellipse":
        return get_adjoint_bridge_landmark_config()
    else:
        raise ValueError(f"Model {model_name} not supported in adjoint bridge training")

def train_adjoint_forward(model_name):
    sde_cls = get_model(model_name)
    config = get_config(model_name)

    ckpt_path = os.path.abspath(f"../assets/ckpts/adjoint/{model_name}_new")
            
    sde = sde_cls(**config["sde"])
    model = ScoreMLP(**config["network"])
    
    data_fn = sde_data.data_adjoint(config["training"]["y"], sde)
    
    opt = optax.chain(
        optax.adam(learning_rate=config["training"]["lr"])
    )
    score_fn = train_utils.get_score(sde)
    
    x_shape = jnp.empty(shape=(1, sde.dim))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)
    
    (loop_key, train_key) = jax.random.split(main_rng_key, 2)
    
    dt = config["sde"]["T"] / config["sde"]["N"]
    
    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key,
        model,
        opt,
        *model_init_sizes,
        dt=dt,
        score=score_fn
    )
    
    train_loop.train(
        key=loop_key,
        training=config["training"],
        data_fn=data_fn,
        train_step=train_step,
        params=params,
        batch_stats=batch_stats,
        opt_state=opt_state,
        sde=sde,
        network=model,
        checkpoint_path=ckpt_path
    )
        
if __name__ == "__main__":
    args = args.parse_args()
    train_adjoint_forward(args.model)
