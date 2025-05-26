#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax import nnx
import argparse
import os
from pathlib import Path

from neuralbridge.sdes.examples import CellSDE
from neuralbridge.sdes.base import AuxiliarySDE, GuidedProposalSDE
from neuralbridge.sdes.neural_bridge import NeuralBridge
from neuralbridge.sdes.solver import SDESolver
from neuralbridge.utils.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train Cell diffusion neural bridge")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Sigma parameter (volatility)")
    parser.add_argument("--x0", type=str, default="[0.1, -0.1]",
                        help="Starting point as a list of 2 elements")
    parser.add_argument("--vT", type=str, default="[2.0, -0.1]",
                        help="Observation point as a list of 2 elements")
    parser.add_argument("--T", type=float, default=4.0,
                        help="Time horizon")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for training")
    parser.add_argument("--n_iters", type=int, default=5000,
                        help="Number of iterations for training")
    parser.add_argument("--hidden_dims", type=int, default=32,
                        help="Number of hidden dimensions in each layer")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of hidden layers in the network")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, 
                        default="../assets/ckpts/neural_bridge/cell",
                        help="Directory to store checkpoints and results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save visualization of trajectories")
    args = parser.parse_args()
    
    # Parse x0 and vT from string to array
    x0 = jnp.array(eval(args.x0))
    vT = jnp.array(eval(args.vT))
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"sigma_{args.sigma}_vT_{vT[0]:.1f}_{vT[1]:.1f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    rngs = nnx.Rngs(args.seed)
    
    # Define the SDE parameters
    dim = 2  # Cell model is 2-dimensional
    sde = CellSDE(
        params={
            "sigma": jnp.array([[args.sigma, 0.0], [0.0, args.sigma]])
        }
    )
    
    # A standard Brownian motion as the auxiliary SDE
    aux_sde = AuxiliarySDE(
        dim_x=dim,
        dim_w=dim,
        coeffs={
            "T": 1.0,
            "beta": lambda t: jnp.ones((dim,)),
            "B": lambda t: -jnp.eye(dim),
            "a_tilde_0": None,
            "a_tilde_T": jnp.array([[args.sigma**2, 0.0], [0.0, args.sigma**2]])
        }
    )
    
    # Define simulation parameters
    dt = 0.01          # time step
    n_steps = int(args.T / dt)
    ts = jnp.linspace(0., args.T, n_steps+1, endpoint=True)
    epsilon = 1e-5          # observation noise
    
    guided_sde = GuidedProposalSDE(
        sde, aux_sde,
        obs_params={
            "vT": vT,
            "LT": jnp.eye(dim),   # fully observed
            "SigmaT": epsilon**2 * jnp.eye(dim)
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
    print(f"Training Cell diffusion bridge with sigma={args.sigma}")
    print(f"Starting point: {x0}")
    print(f"Conditioned point: {vT}")
    params = nnx.state(neural_bridge.nn, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
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
        
        # Create solver for neural bridge model
        nb_solve = SDESolver(neural_bridge)
        
        # Generate trajectories
        n_samples = 10
        test_key = jr.PRNGKey(args.seed + 100)  # Different key for testing
        nb_traj = nb_solve.batch_solve(test_key, x0, ts, n_samples)
        
        # Create a visualization
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        
        # Plot trajectories
        for i in range(n_samples):
            ax[0].plot(ts, nb_traj[i, :, 0], alpha=0.7, color="C0")
            ax[0].plot(ts, nb_traj[i, :, 1], alpha=0.7, color="C1")
        ax[0].scatter(ts[0], x0[0], color='green', s=100, label='Start (x0)')
        ax[0].scatter(ts[0], x0[1], color='green', s=100)
        ax[0].scatter(ts[-1], vT[0], color='red', s=100, label='End (vT)')
        ax[0].scatter(ts[-1], vT[1], color='red', s=100)
        ax[0].set_title("trajectories")
        ax[0].set_xlabel("x1")
        ax[0].set_ylabel("x2")
        ax[0].legend()
        ax[0].grid(True)
        
        # Plot x1 component over time
        for i in range(n_samples):
            ax[1].plot(ts, nb_traj[i, :, 0], alpha=0.7, color="C0")
        ax[1].set_title("x1 component over time")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("x1")
        ax[1].grid(True)
        
        # Plot x2 component over time
        for i in range(n_samples):
            ax[2].plot(ts, nb_traj[i, :, 1], alpha=0.7, color="C1")
        ax[2].set_title("x2 component over time")
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("x2")
        ax[2].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f"trajectories.png")
        plt.savefig(fig_path)
        print(f"Visualization saved to {fig_path}")
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f"Training Loss (sigma={args.sigma})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Save loss figure
        loss_fig_path = os.path.join(output_dir, f"loss_curve.png")
        plt.savefig(loss_fig_path)
        print(f"Loss curve saved to {loss_fig_path}")

if __name__ == "__main__":
    main() 