import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import jax
import jax.numpy as jnp
import optax

from neuralbridge.setups import *

from read_config import read_config

from legacies.adjoint_forward.models.score_mlp import ScoreMLP
from legacies.adjoint_forward.sdes import sde_cell_model, sde_data, sde_utils
from legacies.adjoint_forward.training import train_loop, train_utils

seed = DEFAULT_SEED
main_rng_key = jax.random.PRNGKey(seed)

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="cell_normal", choices=["cell_normal", "cell_rare", "cell_mm", "fhn", "landmark_circle"])

def get_model(model_name: str):
    if "cell" in model_name.lower():
        sde_cls = sde_cell_model.cell_model
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return sde_cls

def train_adjoint_forward(model_name, config):
    sde_cls = get_model(model_name)
    sde = sde_cls(
        T=config.sde.T,
        N=int(config.sde.T / config.sde.dt),
        alpha=config.sde.alpha,
        sigma=config.sde.sigma
    )
    
    ckpt_path = os.path.abspath(f"../assets/ckpts/adjoint/{model_name}_0.1sigma")
    
    model = ScoreMLP(**config.network)
    training = {
        "y": jnp.array(config.training.v).tolist(),
        "batch_size": config.training.batch_size,
        "epochs_per_load": config.training.epochs_per_load,
        "lr": config.training.lr,
        "num_reloads": config.training.num_reloads,
        "load_size": config.training.load_size
    }
    
    data_fn = sde_data.data_adjoint(jnp.array(config.training.v), sde)
    
    opt = optax.chain(
        optax.adam(learning_rate=config.training.lr)
    )
    score_fn = train_utils.get_score(sde)
    
    x_shape = jnp.empty(shape=(1, sde.dim))
    t_shape = jnp.empty(shape=(1, 1))
    model_init_sizes = (x_shape, t_shape)
    
    (loop_key, train_key) = jax.random.split(main_rng_key, 2)
    
    train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
        train_key,
        model,
        opt,
        *model_init_sizes,
        dt=config.sde.dt,
        score=score_fn
    )
    
    train_loop.train(
        key=loop_key,
        training=training,
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
    config = read_config(f"../../configs/{args.model}_adjoint.yaml")
    train_adjoint_forward(args.model, config)
