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

from neuralbridge.sdes.examples import LandmarkGaussianSDE # Changed import
from neuralbridge.sdes.base import AuxiliarySDE, GuidedProposalSDE
from neuralbridge.sdes.neural_bridge import NeuralBridge
from neuralbridge.sdes.solver import SDESolver
from neuralbridge.utils.trainer import train

# Added helper function from landmarks.ipynb
def sampling_ellipse(n, a, b, rotation, shift_x, shift_y):
    u = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x = a * jnp.sin(u)
    y = b * jnp.cos(u)
    x_rot = x * jnp.cos(rotation) - y * jnp.sin(rotation)
    y_rot = x * jnp.sin(rotation) + y * jnp.cos(rotation)
    x_shift = x_rot + shift_x
    y_shift = y_rot + shift_y
    # Ensure concatenation happens along a new axis for proper (m,n) shape later if m=2
    u = jnp.stack([x_shift, y_shift], axis=0) # if m=2, this results in (2,n)
    return u.flatten() # Flatten to (m*n,) for SDE input


def main():
    parser = argparse.ArgumentParser(description="Train Landmark Gaussian SDE neural bridge") # Changed description
    parser.add_argument("--k_alpha", type=float, default=0.3, # Changed argument
                        help="k_alpha parameter for Gaussian kernel")
    parser.add_argument("--k_sigma", type=float, default=0.5, # Changed argument
                        help="k_sigma parameter for Gaussian kernel")
    parser.add_argument("--n_landmarks", type=int, default=50, # Changed argument
                        help="Number of landmarks")
    parser.add_argument("--m_dim", type=int, default=2, # Changed argument
                        help="Dimension of each landmark")
    # Removed x0 and vT string arguments, will generate them
    parser.add_argument("--T", type=float, default=1.0, # Changed default
                        help="Time horizon")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for training")
    parser.add_argument("--n_iters", type=int, default=5000,
                        help="Number of iterations for training")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 256, 256, 128],
                        help="List of hidden dimensions for each layer")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str,
                        default="../assets/ckpts/neural_bridge/landmark", # Changed default
                        help="Directory to store checkpoints and results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save visualization of trajectories")
    args = parser.parse_args()

    # Generate x0 and vT using sampling_ellipse
    x0 = sampling_ellipse(n=args.n_landmarks, a=1.0, b=0.5, rotation=0.0, shift_x=0.0, shift_y=0.0)
    vT = sampling_ellipse(n=args.n_landmarks, a=1.0, b=0.5, rotation=jnp.pi / 3, shift_x=-1.0, shift_y=0.5)

    # Create output directory
    # Simplified output directory naming
    output_dir = Path(args.output_dir) / f"k_alpha_{args.k_alpha}_k_sigma_{args.k_sigma}_n_{args.n_landmarks}_m_{args.m_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    rngs = nnx.Rngs(args.seed)

    # Define the SDE parameters
    dim = args.n_landmarks * args.m_dim # Calculate dimension
    sde = LandmarkGaussianSDE( # Changed SDE type
        params={
            "k_alpha": args.k_alpha,
            "k_sigma": args.k_sigma,
            "n": args.n_landmarks,
            "m": args.m_dim
        }
    )

    # Define auxiliary SDE
    q_half_0 = LandmarkGaussianSDE.compute_q_half(x0, args.m_dim, args.n_landmarks, args.k_alpha, args.k_sigma)
    q_half_T = LandmarkGaussianSDE.compute_q_half(vT, args.m_dim, args.n_landmarks, args.k_alpha, args.k_sigma)

    aux_sde = AuxiliarySDE(
        dim_x=sde.dim_x,
        dim_w=sde.dim_w,
        coeffs={
            "T": args.T,
            "beta": lambda t: jnp.zeros((sde.dim_x, )),
            "B": lambda t: jnp.zeros((sde.dim_x, sde.dim_x)),
            "a_tilde_0": q_half_0 @ q_half_0.T,
            "a_tilde_T": q_half_T @ q_half_T.T
        }
    )

    # Define simulation parameters
    dt = 0.01
    n_steps = int(args.T / dt)
    ts = jnp.linspace(0., args.T, n_steps+1, endpoint=True)
    # Removed ts modification: ts = ts * (2. - ts / args.T)
    epsilon = 2.0e-3 # Changed observation noise

    guided_sde = GuidedProposalSDE(
        sde, aux_sde,
        obs_params={
            "vT": vT,
            "LT": jnp.eye(sde.dim_x), # Fully observed
            "SigmaT": epsilon * jnp.eye(sde.dim_x) # Adjusted SigmaT
        },
        ts=ts,
        eq_type="close" # Changed eq_type as in notebook
    )

    b_fn = guided_sde.b
    sigma_fn = guided_sde.sigma
    G_fn = guided_sde.G

    # Define Neural Network configuration
    # nn_config = {
    #     "model_name": "MLPSmall", 
    #     "input_dim": dim,
    #     "output_dim": dim,
    #     "hidden_dims": args.hidden_dims,
    #     "act_type": "lipswish",
    #     "rngs": rngs
    # }
    nn_config = {
        'model_name': 'MLPMedium',
        'input_dim': dim,
        'output_dim': dim,
        'hidden_dims': args.hidden_dims,
        't_emb_type': 'sinusoidal',
        't_emb_kwargs': {
            'embed_dim': 64,
            'scale': 100.0,
            'max_freq': 1000.0,
            'min_freq': 1.0,
            'rngs': rngs
        },
        'act_type': 'lipswish',
        'rngs': rngs
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
    print(f"Training Landmark Gaussian SDE bridge") # Changed print statement
    print(f"Parameters: k_alpha={args.k_alpha}, k_sigma={args.k_sigma}, n_landmarks={args.n_landmarks}, m_dim={args.m_dim}") # Changed print
    print(f"Starting point shape: {x0.shape}") # Changed print
    print(f"Conditioned point shape: {vT.shape}") # Changed print
    params = nnx.state(neural_bridge.nn, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"Network: {len(args.hidden_dims)} layers with {args.hidden_dims} hidden dims, total params: {total_params}")
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
        n_samples = 4 # Reduced samples for landmark viz
        test_key = jr.PRNGKey(args.seed + 100)  # Different key for testing
        nb_traj = nb_solve.batch_solve(test_key, x0, ts, n_samples)
        nb_traj = nb_traj.reshape(n_samples, n_steps + 1, args.m_dim, args.n_landmarks) # Reshape for plotting

        # Create a visualization (simplified for landmarks)
        # Plotting only the first sample's landmarks
        fig, ax = plt.subplots(figsize=(6, 6)) # Single plot

        for i in range(n_samples):
            for j in range(args.n_landmarks):
                ax.plot(nb_traj[i, :, 0, j], nb_traj[i, :, 1, j], color=f"C{i}", alpha=0.4)

        # Plot initial and final positions for the first sample
        ax.scatter(x0.reshape(args.m_dim, args.n_landmarks)[0,:], x0.reshape(args.m_dim, args.n_landmarks)[1,:], color='green', s=50, label='Start (x0)', marker='o')
        ax.scatter(vT.reshape(args.m_dim, args.n_landmarks)[0,:], vT.reshape(args.m_dim, args.n_landmarks)[1,:], color='red', s=50, label='End (vT)', marker='x')

        ax.set_title(f"Landmark trajectories (Sample 0)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        ax.grid(True)
        ax.axis('equal')


        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, f"trajectories.png")
        plt.savefig(fig_path)
        print(f"Visualization saved to {fig_path}")

        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f"Training Loss (Landmark: k_alpha={args.k_alpha}, k_sigma={args.k_sigma})") # Changed title
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, which="both", ls="--", alpha=0.3)

        # Save loss figure
        loss_fig_path = os.path.join(output_dir, f"loss_curve.png")
        plt.savefig(loss_fig_path)
        print(f"Loss curve saved to {loss_fig_path}")

if __name__ == "__main__":
    main() 