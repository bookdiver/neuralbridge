import jax
import jax.random as jr
import jax.numpy as jnp
import orbax
from flax.training import checkpoints
import time

from .data_loader import dataloader

seed = 1

def save(path, params, opt_state, batch_stats, sde, network, training, average_time=None):
    ckpt = {
        'params': params,
        'opt_state': opt_state,
        'batch_stats': batch_stats,
        'training': training
    }
    if average_time is not None:
        ckpt['average_time'] = average_time

    checkpoints.save_checkpoint(
        ckpt_dir=path,
        target=ckpt,
        step=0,  # Can be modified if step tracking is needed
        prefix="checkpoint_",
        keep=1,
        overwrite=True
    )


def save_training_data(traj, y, N, data_at_times=None): # reverse has shape (B, N*T, dim)
    times = [0.25, 0.5, 0.75]
    for i, t in enumerate(times):
        index = int(N*t)
        data_at_times[i].append(traj[:, index])
    return data_at_times


def train(key, training, data_fn, train_step, params, batch_stats, opt_state, sde, network, checkpoint_path):
    (data_key, dataloader_key, train_key) = jr.split(key, 3)
    data_key = jr.split(data_key, 1)

    batches_per_epoch = max(training["load_size"] // training["batch_size"], 1)

    print("Training")

    total_training_time = 0
    for load in range(training["num_reloads"]):
        # load data
        data_key = jr.split(data_key[0], training["load_size"])
        data = data_fn(data_key)
        infinite_dataloader = dataloader(data, training["batch_size"], loop=True, key=jr.split(dataloader_key, 1)[0])

        for epoch in range(training["epochs_per_load"]):
            total_loss = 0
            total_time = 0
            start = time.process_time()
            for batch, (ts, reverse, correction) in zip(range(batches_per_epoch), infinite_dataloader):
                params, batch_stats, opt_state, _loss = train_step(
                    params, batch_stats, opt_state, ts, reverse, correction
                )
                total_loss = total_loss + _loss
            epoch_loss = total_loss / batches_per_epoch
            end = time.process_time()
            epoch_time = end - start
            total_time += epoch_time
            total_training_time += epoch_time
            actual_epoch = load * training["epochs_per_load"] + epoch
            print(f"Epoch: {actual_epoch}, Loss: {epoch_loss}")

            last_epoch = load == training["num_reloads"] - 1 and epoch == training["epochs_per_load"] - 1
            if actual_epoch % 100 == 0 or last_epoch:
                average_time = total_time / (actual_epoch + 1)
                save(checkpoint_path, params, opt_state, batch_stats, sde, network, training, average_time)
                
    print(f"Training complete in {total_training_time:.3f} seconds")
    # Count and print the total number of parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total number of parameters: {param_count:,}")
