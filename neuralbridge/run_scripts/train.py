import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from read_config import read_config

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.examples import *
from neuralbridge.stochastic_processes.conds import GuidedBridgeProcess
from neuralbridge.solvers.sde import WienerProcess
from neuralbridge.models import neurb

seed = DEFAULT_SEED
main_rng_key = jax.random.PRNGKey(seed)

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="brownian", choices=["brownian", "ou", "cell_normal", "cell_rare", "cell_mm", "fhn_normal", "fhn_rare", "landmark_circle"])

def get_model(model_name: str):
    if "brownian" in model_name.lower():
        X_cls = BrownianProcess
        X_tilde_cls = BrownianAuxProcess
    elif "ou" in model_name.lower():
        X_cls = OrnsteinUhlenbeckProcess
        X_tilde_cls = OrnsteinUhlenbeckAuxProcess
    elif "cell" in model_name.lower():
        X_cls = CellDiffusionProcess
        X_tilde_cls = CellDiffusionAuxProcess
    elif "fhn" in model_name.lower():
        X_cls = FitzHughNagumoProcess
        X_tilde_cls = FitzHughNagumoAuxProcess
    elif "landmark" in model_name.lower():
        X_cls = LandmarkLagrangianProcess
        X_tilde_cls = LandmarkLagrangianAuxProcess
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return X_cls, X_tilde_cls

def train_neural_bridge(model_name, config):
    X_cls, X_tilde_cls = get_model(model_name)
    X = X_cls(
        params=config.sde.params_X, 
        T=config.sde.T, 
        dim=config.sde.dim
    )
    X_tilde = X_tilde_cls(
        params=config.sde.params_X_tilde, 
        T=config.sde.T, 
        dim=config.sde.dim
    )
    X_circ = GuidedBridgeProcess(
        X=X,
        X_tilde=X_tilde,
        u=jnp.array(config.sde.u),
        v=jnp.array(config.sde.v),
        eps=config.sde.eps,
        ts=jnp.arange(0, config.sde.T + config.sde.dt, config.sde.dt)
    )
    
    W = WienerProcess(config.sde.T, config.sde.dt, shape=(config.sde.dim, ))
    X_diamond = neurb.NeuralBridge(
        X_circ=X_circ,
        config=config
    )
    X_diamond.initialize_solver(W)
    _ = X_diamond.train(mode="train")
        
if __name__ == "__main__":
    args = args.parse_args()
    config = read_config(f"../../configs/{args.model}.yaml")
    train_neural_bridge(args.model, config)
