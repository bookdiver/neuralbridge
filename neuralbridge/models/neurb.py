import flax.linen as nn
from flax import struct
from flax.training import train_state, checkpoints
import optax
from tqdm.auto import tqdm

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.conds import (
    GuidedBridgeProcess, NeuralBridgeProcess
)
from neuralbridge.networks.mlps import NetworkFactory
from neuralbridge.solvers.sde import WienerProcess, Euler
from neuralbridge.utils.sample_path import SamplePath

@struct.dataclass
class TrainState(train_state.TrainState):
    rng_key: jax.Array = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, rng_key, **kwargs):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            rng_key=rng_key,
            **kwargs,
        )
        
class NeuralBridge:
    
    def __init__(self, X_circ: GuidedBridgeProcess, config: dict):
        self.X_circ = X_circ
        
        # Network parameters
        network_config = config.get("network", {})
        network_factory = NetworkFactory(network_config)
        self.neural_net = network_factory.create()
        self.X_diamond = NeuralBridgeProcess(X_circ, self.neural_net)

        # Training parameters
        training_config = config.get("training", {})
        self.save_name = training_config.get("save_name", "default_model")
        self.seed = training_config.get("seed", DEFAULT_SEED)
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.batch_size = training_config.get("batch_size", 128)
        self.n_iters = training_config.get("n_iters", 1000)
        self.n_epochs = training_config.get("n_epochs", 10)
        self.ema_decay = training_config.get("ema_decay", None)
        self.optimizer_name = training_config.get("optimizer", "adam")
        self.warmup_steps = training_config.get("warmup_steps", 0)
        self.clip_norm = training_config.get("clip_norm", None)
        
        self.save_relative_dir = "../../assets/ckpts/neurb"
        self.W = None
        self.solver = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        self.init_rng_key = jax.random.PRNGKey(self.seed)
        dummy_t, dummy_x = jnp.zeros((1, 1)), jnp.zeros((1, self.X_circ.dim))
        variables = self.neural_net.init(
            self.init_rng_key,
            dummy_t,
            dummy_x,
        )
        
        self.init_params = variables["params"]
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
            rng_key=self.init_rng_key if self.state is None else self.state.rng_key,
        )
    
    def initialize_solver(self, W: WienerProcess) -> None:
        self.W = W
        self.solver = Euler(
            sde=self.X_diamond,
            W=W
        )
    
    @partial(jax.jit, static_argnums=(0))
    def _compute_loss(self, params, dWs: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        path = self.solver.solve_with_variables(
            x0=self.X_circ.u,
            variables={
                "params": params,
            },
            dWs=dWs,
            batch_size=self.batch_size,
            enforce_endpoint=self.X_circ.v_,
        )
        log_ll, nus = path.log_ll, path.nus
        loss = jnp.sum(0.5 * jnp.sum(nus ** 2, axis=-1), axis=1) * self.solver.dt - log_ll # (b, )
        loss = jnp.mean(loss, axis=0)
        return loss
    
    @partial(jax.jit, static_argnums=(0))
    def _train_step(self, state, dWs: jnp.ndarray) -> Tuple[TrainState, jnp.ndarray]:
        
        def loss_fn(params):
            loss = self._compute_loss(params, dWs)
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
        
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng_key=jax.random.split(state.rng_key)[0])
        return state, loss
    
    def train(self, mode: str = "train", load_relative_dir: Optional[str] = None) -> jnp.ndarray:
        if mode == "train":
            assert self.solver is not None, "path solver not initialized, initialize it using .initialize_path_solver(W)"
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
            
            iter_bar = tqdm(range(self.n_iters), desc=f"Epoch {epoch}", unit="iter", leave=False)
            for _ in iter_bar:
                dWs = self.W.sample(rng_key=self.state.rng_key, batch_size=self.batch_size)
                self.state, loss = self._train_step(self.state, dWs)
                
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

    def _load_checkpoint(self, relative_dir: str) -> None:
        save_dir = os.path.join(relative_dir, f"{self.save_name}")
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
            tx=self.state.tx,
            opt_state=ckpt["opt_state"],
            rng_key=ckpt["rng_key"],
            step=0
        )
        logging.info(f"Checkpoint loaded from {absolute_save_dir}")
        return None
    
    def solve(self, x0: jnp.ndarray, rng_key: jax.Array, batch_size: int, enforce_endpoint: Optional[jnp.ndarray] = None) -> SamplePath:
        dWs = self.W.sample(rng_key, batch_size)
        path = self.solver.solve_with_variables(
            x0=x0,
            variables={
                "params": self.state.params,
            },
            dWs=dWs,
            batch_size=batch_size,
            enforce_endpoint=enforce_endpoint,
        )
        return path