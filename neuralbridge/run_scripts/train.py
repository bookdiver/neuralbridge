import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from read_config import read_config

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
    config = read_config(f"../../configs/{args.model}.yaml")
    main(config)
