import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from read_config import read_config

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.examples import *
from neuralbridge.solvers.sde import WienerProcess
from neuralbridge.models import scorematching

seed = DEFAULT_SEED
main_rng_key = jax.random.PRNGKey(seed)

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="cell_normal", choices=["cell_normal", "cell_rare", "cell_mm", "landmark_circle"])

def get_model(model_name: str):
    if "cell" in model_name.lower():
        X_cls = CellDiffusionProcess
    elif "landmark" in model_name.lower():
        X_cls = LandmarkLagrangianProcess
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return X_cls

def train_score(model_name, config):
    X_cls = get_model(model_name)
    X = X_cls(
        params=config.sde.params_X,
        T=config.sde.T,
        dim=config.sde.dim
    )
    
    W = WienerProcess(
        T=config.sde.T,
        dt=config.sde.dt,
        shape=(config.sde.dim, )
    )
    
    score_model = scorematching.ScoreMatchingReversedBridge(
        X=X,
        u=jnp.array(config.sde.u),
        v=jnp.array(config.sde.v),
        config=config
    )
    score_model.initialize_solver(W)
    _ = score_model.train(mode="train")

if __name__ == "__main__":
    args = args.parse_args()
    config = read_config(f"../../configs/{args.model}_score.yaml")
    train_score(args.model, config)
