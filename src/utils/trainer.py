import os
import functools
import time
import jax
import jax.numpy as jnp
import jax.random as jr

from flax import nnx
import optax
import orbax.checkpoint as ocp


@functools.partial(nnx.jit, static_argnums=(5))
def train_step(nb, optimizer, metrics, x0, ts, batch_size, *, rng_key):

    def loss_fn(nb):
        # rng_key = rngs.data()
        sub_keys = jr.split(rng_key, batch_size)
        _, aux = jax.vmap(
            nb.solve,
            in_axes=(None, None, 0)
        )(ts, x0, sub_keys) 
        
        vss, lls = aux # vss: (b, t-1, d), lls: (b, )
        dts = jnp.diff(ts) # dts: (t-1, )
        loss = 0.5 * jnp.sum(jnp.square(vss), axis=-1)  # (b, t-1, )
        loss = jnp.sum(loss * dts[None, :], axis=1) # (b, )
        loss = loss - lls # (b, )
        loss = jnp.mean(loss, axis=0) # scalar
        return loss
    
    loss, grads = nnx.value_and_grad(
        loss_fn,             
        argnums=0,
        has_aux=False
    )(nb)
    optimizer.update(grads)
    metrics.update(loss=loss)
    
    return loss
    
def train(nb, x0, ts, train_config, *, rngs):
    ckpt_dir = os.path.abspath(train_config["ckpt_dir"])
    best_dir = os.path.join(ckpt_dir, "best")
    
    # Create directories without erasing existing content
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    n_iters = train_config["n_iters"]
    batch_size = train_config["batch_size"]
    lr = train_config["lr"]
    patience = train_config.get("patience", None)
    
    # Create the optimizer
    optimizer = nnx.Optimizer(
        nb,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
            optax.ema(0.995)
        ),
    )
    
    metrics = nnx.metrics.Average("loss")
    
    nb.train()
    
    best_loss = float('inf')
    best_step = 0
    no_improvement_count = 0
    
    checkpointer = ocp.StandardCheckpointer()
    regular_options = ocp.CheckpointManagerOptions(
        max_to_keep=1000,
        create=True
    )
    best_options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        create=True
    )
    regular_manager = ocp.CheckpointManager(
        directory=ocp.test_utils.erase_and_create_empty(ckpt_dir),
        checkpointers={'state': checkpointer},
        options=regular_options
    )
    best_manager = ocp.CheckpointManager(
        directory=ocp.test_utils.erase_and_create_empty(best_dir),
        checkpointers={'state': checkpointer},
        options=best_options
    )
    
    loss_records = []
    data_key = rngs.data()
    
    start_time = time.perf_counter()
    for i in range(1, n_iters + 1):
        data_key = jr.fold_in(data_key, i)
        loss = train_step(nb, optimizer, metrics, x0, ts, batch_size, rng_key=data_key)
        loss_records.append(loss)
    
        if i % 100 == 0 or i == n_iters:
            current_loss = metrics.compute()
            elapsed_time = time.perf_counter() - start_time
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = i
                no_improvement_count = 0
                
                _, state = nnx.split(nb.nn)
                
                best_manager.save(
                    step=best_step,
                    items={'state': state},
                )
            else:
                no_improvement_count += 1
            
            if patience is not None and no_improvement_count >= patience:
                print(f"Early stopping triggered! No improvement for {patience} evaluations")
                print(f"Best loss: {best_loss:.6f} at step {best_step}")
                break
            
            print(f"Step [{i:>5} / {n_iters}]: loss = {current_loss:.6f}, Time elapsed: {elapsed_time:.3f}s")
            
            _, state = nnx.split(nb.nn)
            
            if i % 1000 == 0 or i == n_iters:
                regular_manager.save(
                    step=i,
                    items={'state': state},
                )
                
            metrics.reset()
    
    # Final message about the best model
    print(f"Training completed. Best model saved at step {best_step} with loss {best_loss:.6f}")
    print(f"Best model checkpoint is in: {best_dir}")
    print(f"Time elapsed: {time.perf_counter() - start_time:.4f}s")
    load(nb, best_dir, load_best=True)
    
    # Wait for any async checkpoint operations to complete
    regular_manager.wait_until_finished()
    best_manager.wait_until_finished()
    
    return jnp.array(loss_records)
    
def load(nb, ckpt_dir, load_best=False):
    ckpt_dir = os.path.abspath(ckpt_dir)
    best_dir = os.path.join(ckpt_dir, "best")
    
    # First check if we should load the best model and if it exists
    target_dir = best_dir if load_best and os.path.exists(best_dir) else ckpt_dir
    
    # Create the checkpointer
    checkpointer = ocp.StandardCheckpointer()
    
    # Get available steps
    try:
        steps = [int(step) for step in os.listdir(target_dir) if step.isdigit()]
        if not steps:
            print(f"No checkpoints found in {target_dir}")
            
        latest_step = max(steps)
    except (FileNotFoundError, NotADirectoryError):
        print(f"Directory not found: {target_dir}")
    
    # Prepare the abstract model for restoration
    abstract_model = nnx.eval_shape(lambda: nb.nn)
    graphdef, abstract_state = nnx.split(abstract_model)
    
    # Load the state directly using the checkpointer
    try:
        state_path = os.path.join(target_dir, str(latest_step), 'state')
        state_restored = checkpointer.restore(state_path, abstract_state)
        nb.nn = nnx.merge(graphdef, state_restored)
        
        model_type = "best" if target_dir == best_dir else "latest"
        print(f"Loaded {model_type} model from step {latest_step}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")