from flax import struct
from flax.training import train_state, checkpoints
import optax
from tqdm import tqdm

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.examples import SDEFactory
from neuralbridge.stochastic_processes.conds import ReversedBridgeProcess
from neuralbridge.networks.mlps import NetworkFactory
from neuralbridge.solvers.sde import WienerProcess, Euler
from neuralbridge.utils.t_grid import TimeGrid

@struct.dataclass
class TrainState(train_state.TrainState):
    batch_stats: dict = struct.field(pytree_node=True)
    rng_key: jax.Array = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats, rng_key, **kwargs):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            batch_stats=batch_stats,
            rng_key=rng_key,
            **kwargs,
        )

class ScoreMatchingReversedBridge:

    def __init__(self, config: OmegaConf):
        # SDE parameters
        self.X_unc, self.X_aux = SDEFactory(config).get_sde()
        self.u = jnp.array(config.sde.u)
        self.v = jnp.array(config.sde.v)
        
        self.tGrid = TimeGrid(
            T=config.sde.T,
            dt=config.sde.dt,
            t_scheme=config.sde.t_scheme
        )
        
        # Network parameters
        network_factory = NetworkFactory(config.network)
        self.neural_net = network_factory.create()

        # Training parameters
        self.save_name = config.training.save_name
        self.learning_rate = config.training.learning_rate
        self.batch_size = config.training.batch_size
        self.n_iters_per_epoch = config.training.n_iters_per_epoch
        self.n_epochs = config.training.n_epochs
        self.ema_decay = config.training.ema_decay
        self.optimizer_name = config.training.optimizer
        self.warmup_steps_ratio = config.training.warmup_steps_ratio
        self.clip_norm = config.training.clip_norm
        
        self.save_relative_dir = "../../assets/ckpts/score"
        self.W = None
        self.solver = None
        self._initialize_model()
        
    def _initialize_model(self):
        self.init_rng_key = DEFAULT_RNG_KEY
        dummy_t, dummy_x = jnp.zeros((1, 1)), jnp.zeros((1, self.X_unc.dim))
        variables = self.neural_net.init(
            rngs={
                "params": self.init_rng_key
            },
            t=dummy_t,
            x=dummy_x,
            training=False
        )
        
        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
        self.state = None 
        
        self.W = WienerProcess(dim=self.X_unc.dim)
        self.solver = Euler(
            X=self.X_unc,
            W=self.W,
            tGrid=self.tGrid
        )
        
    def _initialize_optimizer(self) -> None:
        total_steps = int(self.n_iters_per_epoch * self.n_epochs)
        if self.optimizer_name.lower() == "adam":
            opt_class = optax.adam
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=int(total_steps * self.warmup_steps_ratio),
                decay_steps=total_steps,
                end_value=0.01 * self.learning_rate
            )
        elif self.optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=self.learning_rate,
                boundaries_and_scales={
                    int(total_steps * 0.6): 0.1,
                    int(total_steps * 0.85): 0.1,
                }
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_norm) if self.clip_norm is not None else optax.identity(),
            opt_class(lr_schedule),
            optax.ema(self.ema_decay) if self.ema_decay is not None else optax.identity()
        )
        
        if self.state is None:
            self.state = TrainState.create(
                apply_fn=self.neural_net.apply,
                params=self.init_params,
                tx=optimizer,
                batch_stats=self.init_batch_stats,
                rng_key=self.init_rng_key,
            )
        else:
            self.state = self.state.replace(
                tx=optimizer,
                opt_state=optimizer.init(self.state.params)
            )
        
    def _sample_batch(self, rng_key: jax.Array) -> Tuple[tTYPE, xTYPE]:
        path = self.solver.solve(
            x0=self.u,
            rng_key=rng_key,
            batch_size=self.batch_size
        )
        ts, xs = path.ts, path.xs
        ts = repeat(ts, "t -> b t", b=self.batch_size)
        ts = rearrange(ts, "b t -> b t 1")
        return ts, xs
    
    def _get_grads_and_Sigmas(self, ts: tTYPE, xs: xTYPE) -> Tuple[xTYPE, xTYPE]:
        euler = xs[1:] - xs[:-1] - jax.vmap(
                lambda t, x: self.X_unc.f(t, x))(ts[:-1], xs[:-1]) * self.tGrid.dt
        Sigmas = jax.vmap(
            lambda t, x: self.X_unc.Sigma(t, x)
        )(ts[1:], xs[1:])
        inv_Sigmas = jax.vmap(
            lambda t, x: self.X_unc.inv_Sigma(t, x)
        )(ts[:-1], xs[:-1])
        grads = - jnp.einsum('t i j, t j -> t i', inv_Sigmas, euler) / self.tGrid.dt
        return grads, Sigmas

    def _compute_loss(self, params, batch_stats, batch: Tuple[tTYPE, xTYPE]) -> Tuple[jnp.ndarray, dict]:
        ts, xs = batch
        grads, Sigmas = jax.vmap(self._get_grads_and_Sigmas)(ts, xs)
        
        ts_flatten = rearrange(ts[:, 1:, :], "b t 1 -> (b t) 1")
        xs_flatten = rearrange(xs[:, 1:, :], "b t d -> (b t) d")
        
        scores, updated_batch_stats = self.neural_net.apply(
            variables={
                "params": params,
                "batch_stats": batch_stats
            },
            rngs={
                "dropout": jax.random.split(self.state.rng_key)[0]
            },
            t=ts_flatten,
            x=xs_flatten,
            training=True,
            mutable=["batch_stats"]
        )
        scores = rearrange(scores, "(b t) d -> b t d", b=self.batch_size)
        
        loss = scores - grads
        loss = jnp.einsum('b t i, b t i j, b t j -> b t', loss, Sigmas, loss)
        loss = jnp.sum(loss, axis=1) * self.tGrid.dt
        loss = 0.5 * jnp.mean(loss, axis=0)
        return loss, updated_batch_stats

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state, batch: Tuple[tTYPE, xTYPE]) -> Tuple[TrainState, jnp.ndarray]:
        
        def loss_fn(params):
            loss, updated_batch_stats = self._compute_loss(params, state.batch_stats, batch)
            return loss, updated_batch_stats
        
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=aux["batch_stats"])
        state = state.replace(rng_key=jax.random.split(state.rng_key)[0])
        return state, loss
    
    def train(self, mode: str = "train", load_relative_dir: Optional[str] = None) -> Optional[jnp.ndarray]:
        if mode == "train":
            self._initialize_optimizer()
            start_epoch = 1
        elif mode == "pretrained":
            if load_relative_dir is None:
                load_relative_dir = self.save_relative_dir
            self._load_checkpoint(load_relative_dir)
            logging.info(f"Loading pretrained model from the last epoch")
            losses = jnp.load(os.path.join(load_relative_dir, f"{self.save_name}", "losses.npy"))
            return losses
        else:
            raise ValueError(f"Mode {mode} not supported")
        
        losses = []
        for epoch in range(start_epoch, start_epoch + self.n_epochs):
            epoch_losses = []
            
            iter_bar = tqdm(range(self.n_iters_per_epoch), desc=f"Epoch {epoch}", unit="iter", leave=False)
            for _ in iter_bar:
                
                batch = self._sample_batch(self.state.rng_key)
                self.state, loss = self._train_step(self.state, batch)
                
                losses.append(loss)
                epoch_losses.append(loss)
                iter_bar.set_postfix({"loss": f"{loss:>7.5f}"})
                
            epoch_avg_loss = jnp.mean(jnp.array(epoch_losses))
            logging.info(f"Epoch {epoch} average loss: {epoch_avg_loss:>7.5f}")
            
            self._save_checkpoint(epoch)
        
        losses = jnp.array(losses)
        jnp.save(os.path.join(self.save_relative_dir, f"{self.save_name}", "losses.npy"), losses)
        return losses
    
    def _save_checkpoint(self, epoch: int) -> None:
        save_dir = os.path.join(self.save_relative_dir, f"{self.save_name}")
        absolute_save_dir = os.path.abspath(save_dir)
        if not os.path.exists(absolute_save_dir):
            os.makedirs(absolute_save_dir)
        try:
            ckpt = checkpoints.save_checkpoint(
                ckpt_dir=absolute_save_dir,
                target={
                    "params": self.state.params,
                    "batch_stats": self.state.batch_stats,
                    "opt_state": self.state.opt_state,
                    "rng_key": self.state.rng_key,
                    "epoch": epoch,
                    "step": self.state.step 
                },
                step=epoch,
                prefix=f"epoch_",
                keep=1,
                overwrite=True
            )
            logging.info(f"Checkpoint saved to {ckpt}")
            return ckpt
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            return e
        
    def _load_checkpoint(self, load_relative_dir: str) -> None:
        save_dir = os.path.join(load_relative_dir, f"{self.save_name}")
        absolute_save_dir = os.path.abspath(save_dir)
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=absolute_save_dir,
            target=None,
            step=None,
            prefix=f"epoch_"
        )
        if self.state is None:
            self._initialize_optimizer()  
        
        self.state = TrainState(
            apply_fn=self.neural_net.apply,
            params=ckpt["params"],
            batch_stats=ckpt["batch_stats"],
            tx=self.state.tx,
            opt_state=ckpt["opt_state"],
            rng_key=ckpt["rng_key"],
            step=0
        )
        logging.info(f"Checkpoint loaded from {absolute_save_dir}")
        return None
    
    def build_reversed_bridge(self) -> ReversedBridgeProcess:
        return ReversedBridgeProcess(
            X_unc=self.X_unc,
            T=self.tGrid.T,
            neural_net=self.neural_net,
            nn_variables={
                "params": self.state.params,
                "batch_stats": self.state.batch_stats
            }
        )
