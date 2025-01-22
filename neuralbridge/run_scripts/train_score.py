import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from read_config import read_config

from neuralbridge.setups import DEFAULT_RNG_KEY
from neuralbridge.models import score

main_rng_key = DEFAULT_RNG_KEY

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="cell_normal", choices=["cell_normal", "cell_rare", "cell_mm", "landmark_circle"])

def main(config):
    score_model = score.ScoreMatchingReversedBridge(
        config=config
    )
    _ = score_model.train(mode="train")

if __name__ == "__main__":
    args = args.parse_args()
    config = read_config(f"../../configs/{args.model}_score.yaml")
    main(config)
