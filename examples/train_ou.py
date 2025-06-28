#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax import nnx
import argparse
import os
import time
from pathlib import Path

from neuralbridge.sdes.examples import OUSDE, OUBridgeSDE
from neuralbridge.sdes.base import AuxiliarySDE, GuidedBridgeSDE
from neuralbridge.sdes.neural_bridge import NeuralBridge
from neuralbridge.sdes.solver import SDESolver
from neuralbridge.utils.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train Ornstein-Uhlenbeck neural bridge")
    parser.add_argument("--gamma", type=float, default=1.7, 
                        help="Gamma parameter (mean reversion speed)")
    parser.add_argument("--mu", type=float, default=1.0,
                        help="Mu parameter (mean reversion level)")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Sigma parameter (volatility)")
    parser.add_argument("--x0", type=float, default=0.0,
                        help="Starting point")
    parser.add_argument("--vT", type=float, default=1.0,
                        help="Observation point")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for training")
    parser.add_argument("--n_iters", type=int, default=5000,
                        help="Number of iterations for training")
    parser.add_argument("--hidden_dims", type=int, default=20,
                        help="Number of hidden dimensions in each layer")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of hidden layers in the network")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, 
                        default="../assets/ckpts/neural_bridge/ou",
                        help="Directory to store checkpoints and results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save visualization of trajectories")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark the model")
    args = parser.parse_args()
    
    # Create output directory
    if args.benchmark:
        output_dir = Path(args.output_dir) / f"benchmark"
    else:
        output_dir = Path(args.output_dir) / f"theta_{args.theta}_mu_{args.mu}_sigma_{args.sigma}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    rngs = nnx.Rngs(args.seed)
    
    # Define the SDE parameters
    dim = 1
    sde = OUSDE(
        params={
            "gamma": jnp.array(args.gamma),  # For 1D, theta is a 1x1 matrix
            "mu": jnp.array([args.mu]),
            "sigma": jnp.array([[args.sigma]])
        }
    )
    
    # A standard Brownian motion as the auxiliary SDE
    aux_sde = AuxiliarySDE(
        dim_x=1,
        dim_w=1,
        coeffs={
            "T": 1.0,
            "beta": lambda t: jnp.array([0.0]),
            "B": lambda t: jnp.array([[0.0]]),
            "a_tilde_0": None,
            "a_tilde_T": jnp.array([[args.sigma**2]])
        }
    )
    
    # Define starting point and observation
    x0 = jnp.array([args.x0])   # start point
    vT = jnp.array([args.vT])   # observation point at the mean
    n_steps = 500           # number of time steps
    ts = jnp.linspace(0., 1., n_steps+1, endpoint=True)
    epsilon = 1e-5          # observation noise
    
    guided_sde = GuidedBridgeSDE(
        sde, aux_sde,
        obs_params={
            "vT": vT,
            "LT": jnp.eye(1),   # fully observed
            "SigmaT": epsilon**2 * jnp.eye(1)
        },
        ts=ts,
        eq_type="uLM",          # u, L, M formulation of BIF ODE
    )
    
    b_fn = guided_sde.b
    sigma_fn = guided_sde.sigma
    G_fn = guided_sde.G
    
    # Define Neural Network configuration
    nn_config = {
        "model_name": "MLPSmall",
        "input_dim": dim,
        "output_dim": dim,
        "hidden_dims": [args.hidden_dims] * args.num_layers,
        "act_type": "lipswish",
        "rngs": rngs
    }
    
    # Create Neural Bridge model
    neural_bridge = NeuralBridge(
        b_fn, sigma_fn, G_fn, nn_config
    )
    
    # Define training configuration
    train_config = {
        "ckpt_dir": str(output_dir),
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "n_iters": args.n_iters,
        "patience": None
    }
    
    # Print training information
    print(f"Training Ornstein-Uhlenbeck bridge with gamma={args.gamma}, mu={args.mu}, sigma={args.sigma}")
    params = nnx.state(neural_bridge.nn, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"Network: {args.num_layers} layers with {args.hidden_dims} hidden dims, total params: {total_params}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print(f"Number of iterations: {args.n_iters}")
    print(f"Output directory: {output_dir}")
    print("Starting training...")
    
    # Train the model
    losses = train(neural_bridge, x0, ts, train_config, rngs=rngs)
    
    # Save the losses
    loss_path = os.path.join(output_dir, f"losses.npy")
    jnp.save(loss_path, losses)
    
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")
    print(f"Losses saved to {loss_path}")
    
    if args.visualize:
        # Evaluate the model
        neural_bridge.eval()
        
        bridge_sde = OUBridgeSDE(
            params={
                "gamma": jnp.array(args.gamma),
                "mu": jnp.array([args.mu]),
                "sigma": jnp.array([[args.sigma]])
            },
            vT=vT,
            T=1.0
        )
        
        # Create solvers
        nb_solve = SDESolver(neural_bridge)
        tb_solve = SDESolver(bridge_sde)
        
        # Generate trajectories
        n_samples = 10
        test_key = jr.PRNGKey(args.seed + 100)  # Different key for testing
        nb_traj = nb_solve.batch_solve(test_key, x0, ts, n_samples)
        tb_traj = tb_solve.batch_solve(test_key, x0, ts, n_samples)
        
        # Create a visualization
        plt.figure(figsize=(10, 6))
        for i in range(n_samples):
            plt.plot(ts, nb_traj[i], color="C0", alpha=0.7)
            plt.plot(ts, tb_traj[i], color="C1", alpha=0.7)
        
        # Show the mean reversion level
        plt.axhline(y=args.mu, color='r', linestyle='--', label=f"Mean level (μ={args.mu})")
        
        plt.plot([], [], color="C0", label="neural bridge")
        plt.plot([], [], color="C1", label="ou bridge")
        plt.legend()
        plt.title(fr"OU Bridge with $\gamma$={args.gamma}, $\mu$={args.mu}, $\sigma$={args.sigma}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        # Save figure
        fig_path = os.path.join(output_dir, f"trajectories.png")
        plt.savefig(fig_path)
        print(f"Visualization saved to {fig_path}")
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(fr"Training Loss ($\gamma$={args.gamma}, $\mu$={args.mu}, $\sigma$={args.sigma})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Save loss figure
        loss_fig_path = os.path.join(output_dir, f"loss_curve.png")
        plt.savefig(loss_fig_path)
        print(f"Loss curve saved to {loss_fig_path}")

if __name__ == "__main__":
    main() 