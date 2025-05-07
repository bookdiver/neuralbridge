import os
from functools import partial
import time
import jax
import jax.numpy as jnp
import jax.random as jr

from flax import nnx
import optax
import orbax.checkpoint as ocp

@partial(nnx.jit, static_argnums=(5))
def train_step(nb, optimizer, metrics, x0, ts, batch_size, *, rngs):

    def loss_fn(nb):
        rng_key = rngs.data()
        sub_keys = jr.split(rng_key, batch_size)
        _, aux = jax.vmap(
            nb.solve,
            in_axes=(None, None, 0)
        )(ts, x0, sub_keys) # vss: (b, t-1, d), lls: (b, )
        
        vss, lls = aux
        dts = jnp.diff(ts) # dts: (t-1, )
        loss = 0.5 * jnp.sum(jnp.square(vss), axis=-1) # (b, t-1, )
        loss = jnp.sum(loss * dts[None, :], axis=1) # (b, )
        loss -= lls # (b, )
        loss = jnp.mean(loss, axis=0) # scalar
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(nb)
    optimizer.update(grads)
    metrics.update(loss=loss)
    
def train(nb, x0, ts, train_config, *, rngs):
    ckpt_dir = os.path.abspath(train_config["ckpt_dir"])
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    n_iters = train_config["n_iters"]
    batch_size = train_config["batch_size"]
    lr = train_config["lr"]
    
    # Create a CheckpointManager instead of StandardCheckpointer
    checkpointer = ocp.StandardCheckpointer()
    manager = ocp.CheckpointManager(
        directory=ckpt_dir,
        checkpointers={'state': checkpointer}
    )
    
    optimizer = nnx.Optimizer(
        nb,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
            optax.ema(0.995),
        )
    )
    
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss")
    )
    
    nb.train()
    
    start_time = time.time()
    for i in range(n_iters):
        train_step(nb, optimizer, metrics, x0, ts, batch_size, rngs=rngs)
        
        if (i + 1) % 100 == 0:
            end_time = time.time()
            stage_time = end_time - start_time
            for metric, value in metrics.compute().items():
                print(f"Step [{i+1:>5} / {n_iters}]: {metric} = {value:.6f}, Time: {stage_time:.3f}s")
            metrics.reset()
            start_time = time.time()
    
    _, state = nnx.split(nb.nn)
    
    # Use the manager to save the checkpoint
    save_args = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        create=True
    )
    manager.save(
        step=0,  # or i+1 to track iterations
        items={'state': state},
        save_kwargs={'save_args': save_args}
    )
    
    # Wait for the checkpoint operation to complete
    manager.wait_until_finished()
    
def load(nb, ckpt_dir):
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpointer = ocp.StandardCheckpointer()
    
    abstract_model = nnx.eval_shape(lambda: nb.nn)
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = checkpointer.restore(os.path.join(ckpt_dir, 'state'), abstract_state)
    nb.nn = nnx.merge(graphdef, state_restored)