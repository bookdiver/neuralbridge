import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from flax.training import checkpoints
import optax

# Neural Guided + Score Matching Reverse + Guided Proposal
from neuralbridge.setups import *
from neuralbridge.stochastic_processes.examples import (
    CellDiffusionProcess, CellDiffusionAuxProcess
)
from neuralbridge.stochastic_processes.conds import GuidedBridgeProcess
from neuralbridge.solvers.sde import WienerProcess, Euler
from neuralbridge.networks.mlps import MLPSmall, MLP, ScoreNetwork
from neuralbridge.models import neurb, pCN, scorematching
from neuralbridge.utils.plotting import plot_sample_path, plot_mcmc_sample_path, plot_sample_path_histogram

# Adjoint Forward
from legacies.adjoint_forward.models.score_mlp import ScoreMLP
from legacies.adjoint_forward.sdes import sde_cell_model, sde_data, sde_utils
from legacies.adjoint_forward.training import train_loop, train_utils

seed = DEFAULT_SEED
main_rng_key = jax.random.PRNGKey(seed)

args = argparse.ArgumentParser()
args.add_argument("--method", type=str, default="neural_bridge", choices=["neural_bridge", "adjoint_forward", "guided_proposal", "guided_proposal_logs", "score_matching", "all"])
args.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
args.add_argument("--conditional", type=str, default="normal_events", choices=["normal_events", "rare_events", "multi_modality"])

dt = 1.0 / 100
sde_params = {
    "alpha": 1. / 16,
    "sigma": 0.1
}

def get_events(conditional: str):
    if conditional == "normal_events":
        return 4.0, jnp.array([0.1, -0.1]), jnp.array([2.0, -0.1])
    elif conditional == "rare_events":
        return 4.0, jnp.array([0.1, -0.1]), jnp.array([1.0, -0.1])
    elif conditional == "multi_modality":
        return 5.0, jnp.array([-1.0, -1.0]), jnp.array([-1.0, -1.0])
    else:
        raise ValueError(f"Invalid conditional: {conditional}")

def run_cell_neural_bridge(mode: str = "train", conditional: str = "normal_events"):
    T, u, v = get_events(conditional)
    
    W = WienerProcess(T, dt, shape=(2, ))
    X = CellDiffusionProcess(params=sde_params, T=T, dim=2)
    X_tilde = CellDiffusionAuxProcess(params=sde_params, T=T, dim=2)
    X_circ = GuidedBridgeProcess(
        X=X,
        X_tilde=X_tilde,
        u=u,
        v=v,
        L0=jnp.eye(2),
        Sigma0=jnp.eye(2) * 1e-10,
        ts=jnp.arange(0, T + dt, dt)
    )
    
    network = MLPSmall(
        out_dim=2,
        hidden_dims=(20, 20, 20, 20),
        activation="tanh"
    )
    
    train_config = {
        "save_name": f"cell_model_{conditional}",
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
    
    X_diamond = neurb.NeuralBridge(
        X_circ=X_circ,
        neural_net=network,
        config=train_config
    )
    X_diamond.initialize_solver(W)
    
    if mode == "train":
        _ = X_diamond.train(mode="train")
    elif mode == "eval":
        _ = X_diamond.train(mode="pretrained")
        X_diamond_path = X_diamond.solve(x0=u, rng_key=main_rng_key, batch_size=16)
        
        # Brute force to get the "true" conditional paths
        X_solver = Euler(X, W)
        X_path = X_solver.solve(x0=u, rng_key=main_rng_key, batch_size=100_000)
        # Create masks for each component's end point conditions
        x1_mask = (X_path.xs[:, -1, 0] >= v[0] - 1e-2) & (X_path.xs[:, -1, 0] <= v[0] + 1e-2)
        x2_mask = (X_path.xs[:, -1, 1] >= v[1] - 1e-2) & (X_path.xs[:, -1, 1] <= v[1] + 1e-2)
        final_mask = x1_mask & x2_mask
        X_star_path = X_path[final_mask]
        print(f"Found {X_star_path.n_samples} samples that satisfy the endpoint conditions out of {X_path.n_samples} samples")
        
        # Plot the results
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")
        plot_sample_path(X_star_path[:16], ax1, colors=["grey", "grey"], alpha=0.5, linewidth=1.5, zorder=1)
        plot_sample_path(X_diamond_path, ax1, label=[r"$X_{t,1}$", r"$X_{t,2}$"], title="Neural Bridge", linewidth=1.5, zorder=2)

        ax1.set_xlabel(r"$t$", fontsize=14)
        ax1.set_ylabel(r"$X_t$", fontsize=14)
        fig1.savefig(f"../assets/figures/cell_model/{conditional}_neural_bridge.pdf", dpi=300)
        
        if conditional == "multi_modality":
            X_diamond_path_hist = X_diamond.solve(x0=u, rng_key=main_rng_key, batch_size=1000)
            fig2, ax2 = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
            plot_sample_path_histogram(X_diamond_path_hist, plot_dim=0, ax=ax2[0], title=r"$X_{t,1}$", norm="linear")
            plot_sample_path_histogram(X_diamond_path_hist, plot_dim=1, ax=ax2[1], title=r"$X_{t,2}$", norm="linear")
            fig2.savefig(f"../assets/figures/cell_model/{conditional}_neural_bridge_histogram.pdf", dpi=300)
            
            return (fig1, ax1), (fig2, ax2)
        
        return (fig1, ax1)
        
def run_cell_guided_proposal(conditional: str = "normal_events"):
    T, u, v = get_events(conditional)
    
    W = WienerProcess(T, dt, shape=(2, ))
    X = CellDiffusionProcess(params=sde_params, T=T, dim=2)
    X_tilde = CellDiffusionAuxProcess(params=sde_params, T=T, dim=2)
    X_circ = GuidedBridgeProcess(
        X=X,
        X_tilde=X_tilde,
        u=u,
        v=v,
        L0=jnp.eye(2),
        Sigma0=jnp.eye(2) * 1e-10,
        ts=jnp.arange(0, T + dt, dt)
    )
    
    # Brute force to get the "true" conditional paths
    X_solver = Euler(X, W)
    X_path = X_solver.solve(x0=u, rng_key=main_rng_key, batch_size=100_000)
    # Create masks for each component's end point conditions
    x1_mask = (X_path.xs[:, -1, 0] >= v[0] - 1e-2) & (X_path.xs[:, -1, 0] <= v[0] + 1e-2)
    x2_mask = (X_path.xs[:, -1, 1] >= v[1] - 1e-2) & (X_path.xs[:, -1, 1] <= v[1] + 1e-2)
    final_mask = x1_mask & x2_mask
    X_star_path = X_path[final_mask]
    print(f"Found {X_star_path.n_samples} samples that satisfy the endpoint conditions out of {X_path.n_samples} samples")
    
    pCN_config = {
        "rho": 0.98,
        "batch_size": 16,
        "n_iters": 5000
    }
    pCN_model = pCN.PreconditionedCrankNicolson(X_circ=X_circ, config=pCN_config)
    pCN_model.initialize_solver(W)
    _, pCN_result = pCN_model.run()
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")
    plot_sample_path(X_star_path[:16], ax1, colors=["grey", "grey"], alpha=0.5, linewidth=1.5, zorder=1)
    plot_sample_path(pCN_result.path, ax1, label=[r"$X_{t,1}$", r"$X_{t,2}$"], title="Guided Proposal", linewidth=1.5, zorder=2)
    ax1.set_xlabel(r"$t$", fontsize=14)
    ax1.set_ylabel(r"$X_t$", fontsize=14)
    fig1.savefig(f"../assets/figures/cell_model/{conditional}_guided_proposal.pdf", dpi=300)
    
    if conditional == "multi_modality":
        pCN_config = {
            "rho": 0.98,
            "batch_size": 1000,
            "n_iters": 5000
        }
        pCN_model = pCN.PreconditionedCrankNicolson(X_circ=X_circ, config=pCN_config)
        pCN_model.initialize_solver(W)
        _, pCN_result_hist = pCN_model.run()
        fig2, ax2 = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
        plot_sample_path_histogram(pCN_result_hist.path, plot_dim=0, ax=ax2[0], title=r"$X_{t,1}$", norm="linear")
        plot_sample_path_histogram(pCN_result_hist.path, plot_dim=1, ax=ax2[1], title=r"$X_{t,2}$", norm="linear")
        fig2.savefig(f"../assets/figures/cell_model/{conditional}_guided_proposal_histogram.pdf", dpi=300)
        
        return (fig1, ax1), (fig2, ax2)
    
    return (fig1, ax1)
    
def run_cell_score_matching(mode: str = "train", conditional: str = "normal_events"):
    T, u, v = get_events(conditional)
    
    W = WienerProcess(T, dt, shape=(2, ))
    X = CellDiffusionProcess(params=sde_params, T=T, dim=2)
    
    train_config = {
        "save_name": f"cell_model_{conditional}",
        "seed": seed,
        "learning_rate": 5e-4,
        "batch_size": 16,
        "n_iters": 1000,
        "n_epochs": 10,
        "ema_decay": 0.995,
        "optimizer": "adam",
        "clip_norm": None,
        "warmup_steps": 1000
    }
    
    # NOTE: Original structure from https://github.com/jeremyhengjm/DiffusionBridge/blob/master/DiffusionBridge/neuralnet.py
    # network = ScoreNetwork(
    #     dimension=2,
    #     encoder_layers=[16],
    #     pos_dim=16,
    #     decoder_layers=[128, 128],
    #     activation="leaky_relu"
    # )
    network = MLP(
        out_dim=2, 
        hidden_dims=(32, 32, 32, 32),
        t_emb_dim=32,
        t_emb_max_period=100.0,
        t_emb_scaling=100.0,
        norm="batch"
    )
    
    score_matching_model = scorematching.ScoreMatchingReversedBridge(
        X=X,
        u=u,
        v=v,
        neural_net=network,
        config=train_config
    )
    score_matching_model.initialize_solver(W)
    
    if mode == "train":
        _ = score_matching_model.train(mode="train")
    elif mode == "eval":
        _ = score_matching_model.train(mode="pretrained")
        Y_star = score_matching_model.build_reversed_bridge()
        Y_star_solver = Euler(sde=Y_star, W=W)
        Y_star_path = Y_star_solver.solve(
            x0=v, 
            rng_key=main_rng_key, 
            batch_size=16, 
            enforce_endpoint=u
        )
        
        # Brute force to get the "true" conditional paths
        X_solver = Euler(X, W)
        X_path = X_solver.solve(x0=u, rng_key=main_rng_key, batch_size=100_000)
        # Create masks for each component's end point conditions
        x1_mask = (X_path.xs[:, -1, 0] >= v[0] - 1e-2) & (X_path.xs[:, -1, 0] <= v[0] + 1e-2)
        x2_mask = (X_path.xs[:, -1, 1] >= v[1] - 1e-2) & (X_path.xs[:, -1, 1] <= v[1] + 1e-2)
        final_mask = x1_mask & x2_mask
        X_star_path = X_path[final_mask]
        print(f"Found {X_star_path.n_samples} samples that satisfy the endpoint conditions out of {X_path.n_samples} samples")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")
        plot_sample_path(X_star_path[:16], ax, colors=["grey", "grey"], alpha=0.5, linewidth=1.5, zorder=1)
        ax.plot(Y_star_path.ts, Y_star_path.xs[:, ::-1, 0].T, color="C0", alpha=0.7, linewidth=1.5, zorder=2)
        ax.plot(Y_star_path.ts, Y_star_path.xs[:, ::-1, 1].T, color="C1", alpha=0.7, linewidth=1.5, zorder=2)
        ax.plot([], [], color="C0", alpha=0.7, linewidth=1.5, label=r"$X_{t,1}$")
        ax.plot([], [], color="C1", alpha=0.7, linewidth=1.5, label=r"$X_{t,2}$")
        ax.legend(fontsize=14)
        ax.set_xlabel(r"$t$", fontsize=14)
        ax.set_ylabel(r"$X_t$", fontsize=14)
        fig.savefig(f"../assets/figures/cell_model/{conditional}_score_matching.pdf", dpi=300)
        
        return (fig, ax)

def run_cell_guided_proposal_with_logs(conditional: str = "normal_events"):
    T, u, v = get_events(conditional)
    
    W = WienerProcess(T, dt, shape=(2, ))
    X = CellDiffusionProcess(params=sde_params, T=T, dim=2)
    X_tilde = CellDiffusionAuxProcess(params=sde_params, T=T, dim=2)
    X_circ = GuidedBridgeProcess(
        X=X,
        X_tilde=X_tilde,
        u=u,
        v=v,
        L0=jnp.eye(2),
        Sigma0=jnp.eye(2) * 1e-10,
        ts=jnp.arange(0, T + dt, dt)
    )
    
    pCN_config = {
        "rho": 0.98,
        "batch_size": 1,
        "n_iters": 5000
    }
    pCN_model = pCN.PreconditionedCrankNicolson(X_circ=X_circ, config=pCN_config)
    pCN_model.initialize_solver(W)
    pCN_logs, _ = pCN_model.run(log_every=100)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")
    plot_mcmc_sample_path(
        pCN_logs, 
        ax=ax, 
        label=[r"$X_{t,1}$", r"$X_{t,2}$"], 
        title="MCMC updates", 
        cmaps=["spring", "winter"],
        linewidth=1.5, 
        alpha=0.5, 
        n_iters=5000)
    ax.set_xlabel(r"$t$", fontsize=14)
    ax.set_ylabel(r"$X_t$", fontsize=14)
    fig.savefig(f"../assets/figures/cell_model/{conditional}_guided_proposal_logs.pdf", dpi=300)
    
def run_cell_adjoint_forward(mode: str = "train", conditional: str = "normal_events"):
    T, u, v = get_events(conditional)
    
    cell_sde = sde_cell_model.cell_model(
        T=T, 
        N=int(T / dt), 
        alpha=sde_params["alpha"], 
        sigma=sde_params["sigma"]
    )
    
    ckpt_path = os.path.abspath(f"../assets/ckpts/adjoint_forward/cell_model_{conditional}")
    
    network = {
        "output_dim": cell_sde.dim,
        "time_embedding_dim": 32,
        "init_embedding_dim": 16,
        "activation": "leaky_relu",
        "encoder_layer_dims": [16],
        "decoder_layer_dims": [128, 128],
    }
    
    if mode == "train":
        training = {
            "y": v.tolist(),
            "batch_size": 100,
            "epochs_per_load": 1,
            "lr": 1e-2,
            "num_reloads": 100,
            "load_size": 10000,
        }

        data_fn = sde_data.data_adjoint(v, cell_sde)
        
        model = ScoreMLP(**network)
        opt = optax.chain(
            optax.adam(learning_rate=training["lr"]),
        )    
        
        score_fn = train_utils.get_score(cell_sde)
        
        x_shape = jnp.empty(shape=(1, cell_sde.dim))
        t_shape = jnp.empty(shape=(1, 1))
        model_init_sizes = (x_shape, t_shape)
        
        (loop_key, train_key) = jax.random.split(main_rng_key, 2)
        
        train_step, params, opt_state, batch_stats = train_utils.create_train_step_reverse(
            train_key,
            model,
            opt,
            *model_init_sizes,
            dt=dt,
            score=score_fn
        )
        
        train_loop.train(
            key=loop_key,
            training=training,
            data_fn=data_fn,
            train_step=train_step,
            params=params,
            batch_stats=batch_stats,
            opt_state=opt_state,
            sde=cell_sde,
            network=model,
            checkpoint_path=ckpt_path
        )
    elif mode == "eval":
        model = ScoreMLP(**network)
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_path,
            target=None,
            step=0,
            prefix="checkpoint_"
        )
        params = ckpt["params"]
        batch_stats = ckpt["batch_stats"]
        trained_score = train_utils.trained_score(model, params, batch_stats)
        
        n_samples = 16
        rng_keys = jax.random.split(main_rng_key, n_samples)
        xs_star = jax.vmap(
            sde_utils.conditioned,
            in_axes=(0, None, None, None)
        )(rng_keys, u, cell_sde, trained_score)
        ts = jnp.arange(0, T, dt)
        
        # Brute force to get the "true" conditional paths
        W = WienerProcess(T, dt, shape=(2, ))
        X = CellDiffusionProcess(params=sde_params, T=T, dim=2) 
        X_solver = Euler(X, W)
        X_path = X_solver.solve(x0=u, rng_key=main_rng_key, batch_size=100_000)
        # Create masks for each component's end point conditions
        x1_mask = (X_path.xs[:, -1, 0] >= v[0] - 1e-2) & (X_path.xs[:, -1, 0] <= v[0] + 1e-2)
        x2_mask = (X_path.xs[:, -1, 1] >= v[1] - 1e-2) & (X_path.xs[:, -1, 1] <= v[1] + 1e-2)
        final_mask = x1_mask & x2_mask
        X_star_path = X_path[final_mask]
        print(f"Found {X_star_path.n_samples} samples that satisfy the endpoint conditions out of {X_path.n_samples} samples")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")
        plot_sample_path(X_star_path[:16], ax, colors=["grey", "grey"], alpha=0.5, linewidth=1.5, zorder=1)
        ax.plot(ts, xs_star[:, :, 0].T, color="C0", alpha=0.7, linewidth=1.5, zorder=2)
        ax.plot(ts, xs_star[:, :, 1].T, color="C1", alpha=0.7, linewidth=1.5, zorder=2)
        ax.plot([], [], color="C0", alpha=0.7, linewidth=1.5, label=r"$X_{t,1}$")
        ax.plot([], [], color="C1", alpha=0.7, linewidth=1.5, label=r"$X_{t,2}$")
        ax.legend(fontsize=14)
        ax.set_xlabel(r"$t$", fontsize=14)
        ax.set_ylabel(r"$X_t$", fontsize=14)
        ax.set_title(r"Adjoint Forward")
        fig.savefig(f"../assets/figures/cell_model/{conditional}_adjoint_forward.pdf", dpi=300)
        
        return (fig, ax)
    
if __name__ == "__main__":
    args = args.parse_args()
    
    if args.method == "neural_bridge":
        run_cell_neural_bridge(args.mode, args.conditional)
    elif args.method == "guided_proposal":
        run_cell_guided_proposal(args.conditional)
    elif args.method == "guided_proposal_logs":
        run_cell_guided_proposal_with_logs(args.conditional)
    elif args.method == "score_matching":
        run_cell_score_matching(args.mode, args.conditional)
    elif args.method == "adjoint_forward":
        run_cell_adjoint_forward(args.mode, args.conditional)
    elif args.method == "all":
        fig_neural_bridge, ax_neural_bridge = run_cell_neural_bridge(args.mode, args.conditional)
        fig_guided_proposal, ax_guided_proposal = run_cell_guided_proposal(args.conditional)
        fig_score_matching, ax_score_matching = run_cell_score_matching(args.mode, args.conditional)
        fig_adjoint_forward, ax_adjoint_forward = run_cell_adjoint_forward(args.mode, args.conditional)
        
        fig_all = plt.figure(figsize=(12, 8), layout="constrained")
        gs = fig_all.add_gridspec(2, 2)
        
        # Neural Bridge subplot
        ax_all1 = fig_all.add_subplot(gs[0, 0])
        ax_all1.clear()
        for line in ax_neural_bridge.lines:
            ax_all1.plot(line.get_xdata(), line.get_ydata(), 
                        color=line.get_color(), alpha=line.get_alpha(),
                        linewidth=line.get_linewidth(), zorder=line.get_zorder(),
                        label=line.get_label() if line.get_label() != "_nolegend_" else "")
        ax_all1.set_xlabel(r"$t$", fontsize=14)
        ax_all1.set_ylabel(r"$X_t$", fontsize=14)
        ax_all1.set_title("Neural Bridge", fontsize=14)
        ax_all1.legend(fontsize=12)
        
        # Guided Proposal subplot
        ax_all2 = fig_all.add_subplot(gs[0, 1])
        ax_all2.clear()
        for line in ax_guided_proposal.lines:
            ax_all2.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), alpha=line.get_alpha(),
                        linewidth=line.get_linewidth(), zorder=line.get_zorder(),
                        label=line.get_label() if line.get_label() != "_nolegend_" else "")
        ax_all2.set_xlabel(r"$t$", fontsize=14)
        ax_all2.set_ylabel(r"$X_t$", fontsize=14)
        ax_all2.set_title("Guided Proposal", fontsize=14)
        ax_all2.legend(fontsize=12)
        
        # Score Matching subplot
        ax_all3 = fig_all.add_subplot(gs[1, 0])
        ax_all3.clear()
        for line in ax_score_matching.lines:
            ax_all3.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), alpha=line.get_alpha(),
                        linewidth=line.get_linewidth(), zorder=line.get_zorder(),
                        label=line.get_label() if line.get_label() != "_nolegend_" else "")
        ax_all3.set_xlabel(r"$t$", fontsize=14)
        ax_all3.set_ylabel(r"$X_t$", fontsize=14)
        ax_all3.set_title("Score Matching", fontsize=14)
        ax_all3.legend(fontsize=12)
        
        # Adjoint Forward subplot
        ax_all4 = fig_all.add_subplot(gs[1, 1])
        ax_all4.clear()
        for line in ax_adjoint_forward.lines:
            ax_all4.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), alpha=line.get_alpha(),
                        linewidth=line.get_linewidth(), zorder=line.get_zorder(),
                        label=line.get_label() if line.get_label() != "_nolegend_" else "")
        ax_all4.set_xlabel(r"$t$", fontsize=14)
        ax_all4.set_ylabel(r"$X_t$", fontsize=14)
        ax_all4.set_title("Adjoint Forward", fontsize=14)
        ax_all4.legend(fontsize=12)
        
        fig_all.savefig(f"../assets/figures/cell_model/{args.conditional}_all_methods.pdf", dpi=300)
