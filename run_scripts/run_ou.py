import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.examples import (
    OrnsteinUhlenbeckProcess, OrnsteinUhlenbeckAuxProcess, OrnsteinUhlenbeckBridgeProcess
)
from neuralbridge.stochastic_processes.conds import GuidedBridgeProcess
from neuralbridge.solvers.sde import WienerProcess, Euler
from neuralbridge.networks.mlps import MLPSmall
from neuralbridge.models import neurb
from neuralbridge.utils.plotting import plot_sample_path, plot_sample_path_histogram

seed = DEFAULT_SEED
main_rng_key = jax.random.PRNGKey(seed)

args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

dim = 1
T = 1.0
dt = 1.0 / 250
params_X = {
    "gamma": 1.0,
    "sigma": 1.0
}
params_X_tilde = {
    "gamma": 0.0,
    "sigma": 1.0
}

u = jnp.array([0.0])
v = jnp.array([1.0])

def run_ou_neural_bridge(mode: str = "train"):
    X = OrnsteinUhlenbeckProcess(params=params_X, T=T, dim=dim)
    X_tilde = OrnsteinUhlenbeckAuxProcess(params=params_X_tilde, T=T, dim=dim)
    X_star = OrnsteinUhlenbeckBridgeProcess(params=params_X, v=v, T=T, dim=dim)
    X_circ = GuidedBridgeProcess(
        X=X,
        X_tilde=X_tilde,
        u=u,
        v=v,
        L0=jnp.eye(dim),
        Sigma0=jnp.eye(dim) * 1e-10,
        ts=jnp.arange(0, T + dt, dt)
    )
    
    neural_net = MLPSmall(
        out_dim=dim,
        hidden_dims=(20, 20, 20),
        activation="tanh"
    )
    
    train_config = {
        "save_name": "ou",
        "seed": seed,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "n_iters": 500,
        "n_epochs": 10,
        "ema_decay": 0.995,
        "optimizer": "adam",
        "clip_norm": None,
        "warmup_steps": 500
    }
    
    W = WienerProcess(T, dt, shape=(dim, ))
    X_diamond = neurb.NeuralBridge(
        X_circ=X_circ,
        neural_net=neural_net,
        config=train_config
    )
    X_diamond.initialize_solver(W)
    
    if mode == "train":
        _ = X_diamond.train(mode="train")
    elif mode == "eval":
        _ = X_diamond.train(mode="pretrained")
        X_diamond_path = X_diamond.solve(
            x0=u, 
            rng_key=jax.random.PRNGKey(seed), 
            batch_size=8, 
            enforce_endpoint=v
        )

        X_star_solver = Euler(X_star, W)
        X_star_path = X_star_solver.solve(
            x0=u, 
            rng_key=jax.random.PRNGKey(seed), 
            batch_size=8,
            enforce_endpoint=v
        )
        fig1, ax1 = plt.subplots(1, 1, figsize=DEFAULT_FIGURE_SIZE)
        plot_sample_path(X_star_path, ax=ax1, colors='C0', label="Truth", linewidth=2.5)
        plot_sample_path(X_diamond_path, ax=ax1, colors='C1', label="Learned", linewidth=2.0, linestyle="--")
        fig1.savefig("../assets/figures/ou/ou_bridge_vis.pdf", dpi=300)
        
        X_diamond_path = X_diamond.solve(x0=u, rng_key=jax.random.PRNGKey(seed), batch_size=1000, enforce_endpoint=v)
        X_star_path = X_star_solver.solve(x0=u, rng_key=jax.random.PRNGKey(seed), batch_size=1000, enforce_endpoint=v)

        ts = X_diamond_path.ts
        X_diamond_xs = X_diamond_path.xs
        X_star_xs = X_star_path.xs
        
        fig2, ax2 = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE, sharex=True)
        plot_sample_path_histogram(X_diamond_path, ax=ax2[0], plot_dim=0, vertical_bins=200, norm="linear", title="Learned Bridge Process")
        plot_sample_path_histogram(X_star_path, ax=ax2[1], plot_dim=0, vertical_bins=200, norm="linear", title="True Bridge Process")

        ax2[0].set_ylabel(r'$X^{\diamond}_t$', fontsize=14, labelpad=0.5)
        ax2[1].set_xlabel(r'$t$', fontsize=14, labelpad=0.5)
        ax2[1].set_ylabel(r'$X^{\star}_t$', fontsize=14, labelpad=0.5)
        fig2.savefig("../assets/figures/ou/ou_bridge_histogram.pdf", dpi=300)
        
        t_slices = [0.2, 0.4, 0.6, 0.8]

        fig3, ax3 = plt.subplots(2, 2, figsize=DEFAULT_FIGURE_SIZE, sharex=True, sharey=True)
        for i, t_slice in enumerate(t_slices):
            t_idx = jnp.argmin(jnp.abs(ts - t_slice * T))
            ax3[i // 2, i % 2].hist(X_diamond_xs[:, t_idx, 0], bins=70, density=True, color='C0', alpha=0.5, label="Learned", edgecolor='C0', histtype='stepfilled')
            ax3[i // 2, i % 2].hist(X_star_xs[:, t_idx, 0], bins=70, density=True, color='C1', alpha=0.5, label="Truth", edgecolor='C1', histtype='stepfilled')
            ax3[i // 2, i % 2].set_title(f"t = {t_slice}")
            ax3[i // 2, i % 2].legend()

        ax3[0, 0].set_ylabel(r'Density', fontsize=14, labelpad=0.5)
        ax3[1, 0].set_ylabel(r'Density', fontsize=14, labelpad=0.5)
        ax3[1, 0].set_xlabel(r'$X_t$', fontsize=14, labelpad=0.5)
        ax3[1, 1].set_xlabel(r'$X_t$', fontsize=14, labelpad=0.5)
        fig3.savefig("../assets/figures/ou/ou_bridge_histogram_slices.pdf", dpi=300)
        
if __name__ == "__main__":
    args = args.parse_args()
    run_ou_neural_bridge(args.mode)