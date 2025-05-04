import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse


from neuralbridge.configs.neural_bridge_config import *
from neuralbridge.setups import DEFAULT_RNG_KEY
from neuralbridge.models import neurb

main_rng_key = DEFAULT_RNG_KEY

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="brownian", choices=["brownian", "ou", "cell_normal", "cell_rare", "cell_mm", "fhn_normal", "fhn_extreme", "landmark_ellipse"])


def main(config):
    neural_bridge_model = neurb.NeuralBridge(
        config=config
    )
    _ = neural_bridge_model.train(mode="train")
        
if __name__ == "__main__":
    args = args.parse_args()
    if args.model == "ou":
        config = get_neural_bridge_ou_config()
    elif args.model == "cell_normal":
        config = get_neural_bridge_cell_normal_config()
    elif args.model == "landmark_ellipse":
        config = get_neural_bridge_landmark_config()
    else:
        raise ValueError(f"Model {args.model} not supported")
    main(config)
