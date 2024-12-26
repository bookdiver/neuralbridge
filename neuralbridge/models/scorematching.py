import flax.linen as nn
from flax import struct
from flax.training import train_state, checkpoints
import optax
from tqdm.auto import tqdm

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.unconds import ContinuousTimeProcess
from neuralbridge.stochastic_processes.conds import ReversedBridgeProcess
from neuralbridge.solvers.sde import WienerProcess, Euler

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
    X: ContinuousTimeProcess
    neural_net: nn.Module
    config: dict

    def __init__(self, 
                 X: ContinuousTimeProcess, 
                 u: jnp.ndarray,
                 v: jnp.ndarray,
                 neural_net: nn.Module, 
                 config: dict):
        self.X = X
        
        self.u = u
        self.v_ = v
        self.v = jnp.where(~jnp.isnan(v), v, self.v_)
        
        self.neural_net = neural_net
        
        # Training parameters
        self.save_name = config.get("save_name", "default_model")
        self.seed = config.get("seed", DEFAULT_SEED)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 128)
        self.n_iters = config.get("n_iters", 1000)
        self.n_epochs = config.get("n_epochs", 10)
        self.ema_decay = config.get("ema_decay", None)
        self.optimizer_name = config.get("optimizer", "adam")
        self.warmup_steps = config.get("warmup_steps", 0)
        self.clip_norm = config.get("clip_norm", None)
        
        self.solver = None
        self._initialize_model()
        
    def _initialize_model(self):
        self.init_rng_key = jax.random.PRNGKey(self.seed)
        dummy_t, dummy_x = jnp.zeros((1, 1)), jnp.zeros((1, self.X.dim))
        variables = self.neural_net.init(
            self.init_rng_key,
            dummy_t,
            dummy_x,
            training=False
        )
        
        self.init_params = variables["params"]
        self.init_batch_stats = variables["batch_stats"] if "batch_stats" in variables else {}
        self.state = None 
        
    def _initialize_optimizer(self, learning_rate: Optional[float] = None) -> None:
        if self.optimizer_name == "adam":
            opt_class = optax.adam
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate if learning_rate is not None else self.learning_rate,
                warmup_steps=self.warmup_steps,
                decay_steps=int(self.n_iters * self.n_epochs),
                end_value=0.01 * (learning_rate if learning_rate is not None else self.learning_rate)
            )
        elif self.optimizer_name == "sgd":
            opt_class = optax.sgd
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=learning_rate if learning_rate is not None else self.learning_rate,
                boundaries_and_scales={
                    int(self.n_iters * self.n_epochs * 0.6): 0.1,
                    int(self.n_iters * self.n_epochs * 0.85): 0.1,
                }
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_norm) if self.clip_norm is not None else optax.identity(),
            opt_class(lr_schedule),
            optax.ema(self.ema_decay) if self.ema_decay is not None else optax.identity()
        )
        
        self.state = TrainState.create(
            apply_fn=self.neural_net.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            rng_key=self.init_rng_key if self.state is None else self.state.rng_key,
        )

    def initialize_solver(self, W: WienerProcess) -> None:
        self.solver = Euler(
            sde=self.X,
            W=W
        )
        
    def _sample_batch(self, rng_key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        path = self.solver.solve(
            x0=self.u,
            rng_key=rng_key,
            batch_size=self.batch_size,
            enforce_endpoint=None,
        )
        ts, xs = path.ts, path.xs
        ts = repeat(ts, "t -> b t", b=self.batch_size)
        ts = rearrange(ts, "b t -> b t 1")
        return ts, xs
    
    def _get_grads_and_Sigmas(self, ts: jnp.ndarray, xs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        euler = xs[1:] - xs[:-1] - jax.vmap(
                lambda t, x: self.X.f(t, x))(ts[:-1], xs[:-1]) * self.solver.dt
        Sigmas = jax.vmap(
            lambda t, x: self.X.Sigma(t, x)
        )(ts[1:], xs[1:])
        inv_Sigmas = jax.vmap(
            lambda t, x: self.X.inv_Sigma(t, x)
        )(ts[:-1], xs[:-1])
        grads = - jnp.einsum('t i j, t j -> t i', inv_Sigmas, euler) / self.solver.dt
        return grads, Sigmas

    def _compute_loss(self, params: jnp.ndarray, batch_stats: dict, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, dict]:
        variables = {"params": params, "batch_stats": batch_stats}
        ts, xs = batch
        grads, Sigmas = jax.vmap(self._get_grads_and_Sigmas)(ts, xs)
        
        ts_flatten = rearrange(ts[:, 1:, :], "b t 1 -> (b t) 1")
        xs_flatten = rearrange(xs[:, 1:, :], "b t d -> (b t) d")
        
        scores, updated_batch_stats = self.neural_net.apply(
            variables,
            t=ts_flatten,
            x=xs_flatten,
            training=True,
            mutable=["batch_stats"]
        )
        scores = rearrange(scores, "(b t) d -> b t d", b=self.batch_size)
        
        loss = scores - grads
        loss = jnp.einsum('b t i, b t i j, b t j -> b t', loss, Sigmas, loss)
        loss = jnp.sum(loss, axis=1) * self.solver.dt
        loss = 0.5 * jnp.mean(loss, axis=0)
        return loss, updated_batch_stats

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state, batch) -> Tuple[TrainState, jnp.ndarray]:
        
        def loss_fn(params):
            loss, updated_batch_stats = self._compute_loss(params, state.batch_stats, batch)
            return loss, updated_batch_stats
        
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=aux["batch_stats"])
        state = state.replace(rng_key=jax.random.split(state.rng_key)[0])
        return state, loss
    
    def train(self, mode: str = "train", epoch: Optional[int] = None) -> Optional[jnp.ndarray]:
        if mode == "train":
            assert self.solver is not None, "path solver not initialized, initialize it using .initialize_path_solver(wiener_proc)"
            self._initialize_optimizer()
            start_epoch = 1
        elif mode == "pretrained":
            self._load_checkpoint(epoch)
            logging.info(f"Loading pretrained model from the last epoch")
            return None
        else:
            raise ValueError(f"Mode {mode} not supported")
        
        losses = []
        epoch_bar = tqdm(range(start_epoch, start_epoch + self.n_epochs), desc="Training epochs", unit="epoch")
        for epoch in epoch_bar:
            epoch_losses = []
            
            iter_bar = tqdm(range(self.n_iters), desc=f"Epoch {epoch}", unit="iter", leave=False)
            for _ in iter_bar:
                
                batch = self._sample_batch(self.state.rng_key)
                self.state, loss = self._train_step(self.state, batch)
                
                losses.append(loss)
                epoch_losses.append(loss)
                iter_bar.set_postfix({"loss": f"{loss:>7.5f}"})
                
            epoch_avg_loss = jnp.mean(jnp.array(epoch_losses))
            epoch_bar.set_postfix({"avg_loss": f"{epoch_avg_loss:>7.5f}"})
            logging.info(f"Epoch {epoch} average loss: {epoch_avg_loss:>7.5f}")
            
            self._save_checkpoint(epoch)
        
        losses = jnp.array(losses)
        jnp.save(os.path.join("../assets/ckpts/score_matching", f"{self.save_name}", "losses.npy"), losses)
        return losses
    
    def _save_checkpoint(self, epoch: int, relative_dir: Optional[str] = "../assets/ckpts/score_matching") -> None:
        save_dir = os.path.join(relative_dir, f"{self.save_name}")
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
                keep=1000,
                overwrite=True
            )
            logging.info(f"Checkpoint saved to {ckpt}")
            return ckpt
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")
            return e
        
    def _load_checkpoint(self, epoch: int, relative_dir: Optional[str] = "../assets/ckpts/score_matching") -> None:
        save_dir = os.path.join(relative_dir, f"{self.save_name}")
        absolute_save_dir = os.path.abspath(save_dir)
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=absolute_save_dir,
            target=None,
            step=epoch,
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
    
    def inference(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        t = rearrange(jnp.array([t]), "t -> 1 t")
        x = rearrange(x, "d -> 1 d")
        output = self.state.apply_fn(
            variables={
                "params": self.state.params,
                "batch_stats": self.state.batch_stats
            },
            t=t,
            x=x,
            training=False,
            mutable=False
        )
        output = rearrange(output, "1 d -> d")
        return output
    
    def build_reversed_bridge(self) -> ReversedBridgeProcess:
        score_fn = lambda t, x: self.inference(t, x)
        return ReversedBridgeProcess(
            X=self.X,
            score_fn=score_fn
        )
        

        
    