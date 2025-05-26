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

from neuralbridge.sdes.examples import FHNSDE
from neuralbridge.sdes.base import AuxiliarySDE, GuidedProposalSDE
from neuralbridge.sdes.neural_bridge import NeuralBridge
from neuralbridge.sdes.solver import SDESolver
from neuralbridge.utils.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train FitzHugh-Nagumo diffusion neural bridge")
    parser.add_argument("--chi", type=float, default=0.1,
                        help="Chi parameter")
    parser.add_argument("--s", type=float, default=0.0,
                        help="S parameter")
    parser.add_argument("--gamma", type=float, default=1.5,
                        help="Gamma parameter")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Alpha parameter")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Sigma parameter (noise intensity)")
    parser.add_argument("--x0", type=str, default="[-0.5, -0.6]",
                        help="Starting point as a list of 2 elements")
    parser.add_argument("--vT", type=str, default="[-1.0]",
                        help="Observation point as a list (1 element for partially observed)")
    parser.add_argument("--T", type=float, default=2.0,
                        help="Time horizon")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for training")
    parser.add_argument("--n_iters", type=int, default=5000,
                        help="Number of iterations for training")
    parser.add_argument("--hidden_dims", type=int, default=32,
                        help="Number of hidden dimensions in each layer")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of hidden layers in the network")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, 
                        default="../assets/ckpts/neural_bridge/fhn",
                        help="Directory to store checkpoints and results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save visualization of trajectories")
    args = parser.parse_args()
    
    # Parse x0 and vT from string to array
    x0 = jnp.array(eval(args.x0))
    vT = jnp.array(eval(args.vT))
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"chi_{args.chi}_s_{args.s}_gamma_{args.gamma}_alpha_{args.alpha}_sigma_{args.sigma}_vT_{vT[0]:.1f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    rngs = nnx.Rngs(args.seed)
    
    # Define the SDE parameters
    dim = 2  # FHN model is 2-dimensional
    sde = FHNSDE(
        params={
            "chi": args.chi,
            "s": args.s,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "sigma": args.sigma
        }
    )
    
    # Define auxiliary SDE for FHN with vT-dependent coefficients
    aux_sde = AuxiliarySDE(
        dim_x=dim,
        dim_w=dim,
        coeffs={
            "T": args.T,
            "beta": lambda t: jnp.array([(2*vT[0]**3 + args.s) / args.chi, args.alpha]),
            "B": lambda t: jnp.array([[(1-3*vT[0]**2)/args.chi, -1/args.chi], [args.gamma, -1]]),
            "a_tilde_0": None,
            "a_tilde_T": jnp.array([[0., 0.], [0., args.sigma**2]])
        }
    )
    
    # Define simulation parameters
    dt = 0.01          # time step (finer for FHN)
    n_steps = int(args.T / dt)
    ts = jnp.linspace(0., args.T, n_steps+1, endpoint=True)
    ts = ts * (2. - ts / args.T)
    epsilon = 2 * 1e-2          # observation noise
    
    guided_sde = GuidedProposalSDE(
        sde, aux_sde,
        obs_params={
            "vT": vT,
            "LT": jnp.array([[1., 0.]]),   # partially observed (only first component)
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
    print(f"Training FitzHugh-Nagumo diffusion bridge")
    print(f"Parameters: chi={args.chi}, s={args.s}, gamma={args.gamma}, alpha={args.alpha}, sigma={args.sigma}")
    print(f"Starting point: {x0}")
    print(f"Conditioned point: {vT} (partially observed - first component only)")
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
        
        # Plot trajectories in phase space
        for i in range(n_samples):
            ax[0].plot(nb_traj[i, :, 0], nb_traj[i, :, 1], alpha=0.7)
        ax[0].scatter(x0[0], x0[1], color='green', s=100, label='Start (x0)', marker='o')
        ax[0].axvline(x=vT[0], color='red', linestyle='--', alpha=0.7, label=f'Observed endpoint x1={vT[0]:.1f}')
        ax[0].set_title("Phase space trajectories")
        ax[0].set_xlabel("x1 (voltage)")
        ax[0].set_ylabel("x2 (recovery)")
        ax[0].legend()
        ax[0].grid(True)
        
        # Plot x1 component over time (observed component)
        for i in range(n_samples):
            ax[1].plot(ts, nb_traj[i, :, 0], alpha=0.7, color="C0")
        ax[1].axhline(y=vT[0], color='red', linestyle='--', alpha=0.7, label=f'Target x1={vT[0]:.1f}')
        ax[1].set_title("x1 (voltage) component over time")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("x1")
        ax[1].legend()
        ax[1].grid(True)
        
        # Plot x2 component over time (unobserved component)
        for i in range(n_samples):
            ax[2].plot(ts, nb_traj[i, :, 1], alpha=0.7, color="C1")
        ax[2].set_title("x2 (recovery) component over time")
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
        plt.title(f"Training Loss (FHN: chi={args.chi}, sigma={args.sigma})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Save loss figure
        loss_fig_path = os.path.join(output_dir, f"loss_curve.png")
        plt.savefig(loss_fig_path)
        print(f"Loss curve saved to {loss_fig_path}")

if __name__ == "__main__":
    main() 