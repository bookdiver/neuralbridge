from flax import struct
from flax.training import train_state, checkpoints
import optax
import time

from neuralbridge.setups import *
from neuralbridge.stochastic_processes.examples import SDEFactory
from neuralbridge.stochastic_processes.conds import (
    GuidedBridgeProcess, NeuralBridgeProcess
)
from neuralbridge.networks.mlps import NetworkFactory
from neuralbridge.solvers.sde import WienerProcess, Euler
from neuralbridge.utils.t_grid import TimeGrid

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
    
    def __init__(self, config: OmegaConf):
        # SDE parameters
        self.X_unc, self.X_aux = SDEFactory(config).get_sde()
        self.u = config["sde"]["u"]
        self.v = config["sde"]["v"]
        
        if config["sde"]["name"].lower() == "landmark":
            self.X_aux.initialize_g(self.v)
        
        self.tGrid = TimeGrid(
            T=config["sde"]["T"],
            dt=config["sde"]["dt"],
            t_scheme=config["sde"]["t_scheme"]
        )
        self.X_gui = GuidedBridgeProcess(
            X_unc=self.X_unc,
            X_aux=self.X_aux,
            u=self.u,
            v=self.v,
            eps=config["sde"]["eps"],
            ts=self.tGrid.ts
        )
        
        self.W = WienerProcess(dim=config["sde"]["W_dim"])
        
        # Network parameters
        network_factory = NetworkFactory(config["network"])
        self.neural_net = network_factory.create()

        # Training parameters
        self.save_name = config["training"]["save_name"]
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.n_iters_per_epoch = config["training"]["n_iters_per_epoch"]
        self.n_epochs = config["training"]["n_epochs"]
        self.ema_decay = config["training"]["ema_decay"]
        self.optimizer_name = config["training"]["optimizer"]
        self.warmup_steps_ratio = config["training"]["warmup_steps_ratio"]
        self.clip_norm = config["training"]["clip_norm"]
        
        self.save_relative_dir = "../../assets/ckpts/neurb"
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        self.init_rng_key = DEFAULT_RNG_KEY
        dummy_t, dummy_x = jnp.zeros((1, 1)), jnp.zeros((1, self.X_gui.dim))
        variables = self.neural_net.init(
            rngs={
                "params": self.init_rng_key
            },
            t=dummy_t,
            x=dummy_x,
            training=False
        )
        
        self.init_params = variables["params"]
        
        self.X_neu = NeuralBridgeProcess(
            X_gui=self.X_gui,
            neural_net=self.neural_net
        )
        self.state = None 
        
        self.solver = Euler(
            X=self.X_neu,
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
        elif self.optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=int(total_steps * self.warmup_steps_ratio),
                decay_steps=total_steps,
                end_value=0.01 * self.learning_rate
            )
        elif self.optimizer_name.lower() == "rmsprop":
            opt_class = optax.rmsprop
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
                rng_key=self.init_rng_key,
            )
        else:
            self.state = self.state.replace(
                tx=optimizer,
                opt_state=optimizer.init(self.state.params)
            )
    
    def _compute_loss(self, params, dWs: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        path = self.solver.solve(
            x0=self.X_gui.u,
            dWs=dWs,
            batch_size=self.batch_size,
            enforced_endpoint=self.v,
            compute_log_likelihood_ratio=True,
            correct_log_likelihood_ratio=False,
            nn_variables=FrozenDict({
                "params": params,
            }),
            training=True,
        )
        log_likelihood_ratio, nn_vals = path.log_likelihood_ratio, path.nn_vals
        loss = jnp.sum(0.5 * jnp.sum(jnp.square(nn_vals) * self.tGrid.dts[jnp.newaxis, :, jnp.newaxis], axis=-1), axis=1) - log_likelihood_ratio # (b, )
        loss = jnp.mean(loss, axis=0)
        return loss
    
    @partial(jax.jit, static_argnums=(0))
    def _train_step(self, state, dWs: jnp.ndarray) -> Tuple[TrainState, jnp.ndarray]:
        
        def loss_fn(params):
            loss = self._compute_loss(params, dWs)
            return loss
        
        loss, grads = jax.value_and_grad(
            loss_fn, 
            argnums=0,
            has_aux=False
        )(state.params)
        
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng_key=jax.random.split(state.rng_key)[0])
        
        return state, loss
    
    def train(self, mode: str = "train", load_relative_dir: Optional[str] = None) -> jnp.ndarray:
        if mode == "train":
            self._initialize_optimizer()
            start_epoch = 1
        elif mode == "pretrained":
            if load_relative_dir is None:
                load_relative_dir = self.save_relative_dir
            self._load_checkpoint(load_relative_dir)
            print(f"Loading pretrained model from the last epoch")
            losses = jnp.load(os.path.join(load_relative_dir, f"{self.save_name}", "losses.npy"))
            return losses
        else:
            raise ValueError(f"Mode {mode} not supported")
        
        losses = []
        total_training_time = 0
        for epoch in range(start_epoch, start_epoch + self.n_epochs):
            epoch_losses = 0.0
            
            start = time.process_time()
            for _ in range(self.n_iters_per_epoch):
                dWs = self.W.sample(rng_key=self.state.rng_key, dts=self.tGrid.dts, batch_size=self.batch_size)
                self.state, loss = self._train_step(self.state, dWs)
                
                losses.append(loss)
                epoch_losses += loss
            end = time.process_time()
            epoch_time = end - start
            total_training_time += epoch_time
            epoch_avg_loss = epoch_losses / self.n_iters_per_epoch
            print(f"Epoch {epoch}, Loss: {epoch_avg_loss}")

        self._save_checkpoint(epoch) 
            
        losses = jnp.array(losses)
        jnp.save(os.path.join(self.save_relative_dir, f"{self.save_name}", "losses.npy"), losses)
        
        print(f"Training complete in {total_training_time:.3f} seconds")
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params))
        print(f"Total number of parameters: {param_count:,}")
        
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
        
        self.state = self.state.replace(
            params=ckpt["params"],
            opt_state=ckpt["opt_state"],
            rng_key=ckpt["rng_key"],
            step=ckpt["step"]
        )
        logging.info(f"Checkpoint loaded from {absolute_save_dir}")
        return None
    
    def build_neural_bridge(self) -> NeuralBridgeProcess:
        return NeuralBridgeProcess(
            X_gui=self.X_gui,
            neural_net=self.neural_net,
            nn_variables=FrozenDict({
                "params": self.state.params,
            })
        )
