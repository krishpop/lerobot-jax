import collections
import os
import pickle
from functools import partial
from typing import Dict, Optional, Tuple, Union, List, Any, Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import omegaconf
import optax
import tqdm
import wandb
from absl import app, flags
from flax import struct

from flax.core import FrozenDict, frozen_dict
from flax.training import checkpoints
from jaxrl_m.common import target_update
from jaxrl_m.evaluation import evaluate
from jaxrl_m.networks import Critic, ensemblize
from jaxrl_m.typing import InfoDict, PRNGKey, Batch, Array, Data
from jaxrl_m.vision.preprocess import PreprocessEncoder
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from ml_collections import ConfigDict, config_flags

from lerobot_jax.utils.model_utils import (
    MLP,
    SimNorm,
    TDMPC2SimpleConv,
    TrainState,
    WithMappedEncoders,
)
from lerobot_jax.utils.norm_utils import normalize_inputs, unnormalize_outputs

# -------------
# Config class
# -------------


@jdc.pytree_dataclass
class TDMPC2Config:
    """
    Configuration for TDMPC2. Adjust or extend as necessary.
    """

    # Training
    lr: float = 3e-4
    enc_lr_scale: float = 1.0
    max_std: float = 2.0
    min_std: float = 0.05
    horizon: int = 3
    num_samples: int = 512
    num_elites: int = 64
    iterations: int = 6
    num_pi_trajs: int = 24
    temperature: float = 0.5
    grad_clip_norm: float = 20.0
    batch_size: int = 256
    discount_min: float = 0.95
    discount_max: float = 0.995
    discount_denom: float = 5
    tau: float = 0.01
    # Model architecture / latent dims
    latent_dim: int = 64
    action_dim: int = 6
    mlp_dim: int = 128
    num_channels: int = 32
    input_shapes: Optional[Dict[str, Any]] = None
    image_keys: Tuple[str, ...] = ()
    output_shapes: Optional[Dict[str, Any]] = None
    dropout: float = 0.0  # 0.01
    simnorm_dim: int = 8
    # obs: str = "state"  # Observation type (state or rgb)
    # Loss coefficients
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    # actor
    log_std_min: float = -10
    log_std_max: float = 2
    entropy_coef: float = 1e-4
    # Critic
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10
    vmax: float = 10
    # For dynamic tasks
    mpc: bool = True
    episode_length: int = 200
    episode_lengths: Tuple[int, ...] = ()
    # Discount factor for weighting Q-values over the horizon
    rho: float = 0.5
    # For large action dims
    action_mask: bool = False
    # Multitask specific
    multitask: bool = False
    tasks: Tuple[str, ...] = ()
    task_dim: int = 32
    action_dims: Tuple[int, ...] = ()  # Action dimensions for each task
    n_action_steps: int = 5
    n_action_repeats: int = 1

    @property
    def bin_size(self) -> float:
        return (self.vmax - self.vmin) / (self.num_bins - 1)

    @property
    def discount_values(
        self,
    ) -> jnp.ndarray:  # Or a method get_discount(task_idx=None)
        if self.multitask and len(self.episode_lengths) > 0:
            discounts = []
            for ep_len in self.episode_lengths:
                base = (
                    (ep_len - self.discount_denom) / ep_len
                    if ep_len > 0
                    else self.discount_max
                )  # Avoid div by zero
                discounts.append(
                    jnp.clip(base, self.discount_min, self.discount_max)
                )
            return jnp.array(discounts)
        else:
            base = (
                (self.episode_length - self.discount_denom)
                / self.episode_length
                if self.episode_length > 0
                else self.discount_max
            )
            return jnp.array(
                [jnp.clip(base, self.discount_min, self.discount_max)]
            )  # Return as array for consistency


# ----------------------------------------------------------------------------------------
# WorldModel, reward model, Q-function
# ----------------------------------------------------------------------------------------
class WorldModel(nn.Module):
    """
    JAX version of TDMPC2's world model with task embedding support.
        TD-MPC2 implicit world model architecture.
        Can be used for both single-task and multi-task experiments.
    """

    cfg: TDMPC2Config

    def setup(self):
        # Create task embedding and action masks if using multitask.
        if self.cfg.multitask:
            self.task_emb = nn.Embed(
                num_embeddings=len(self.cfg.tasks),
                features=self.cfg.task_dim,
                embedding_init=nn.initializers.uniform(scale=1.0),
            )
            if self.cfg.action_mask:
                masks = jnp.zeros((len(self.cfg.tasks), self.cfg.action_dim))
                for i in range(len(self.cfg.tasks)):
                    masks = masks.at[i, : self.cfg.action_dims[i]].set(1.0)
                self._action_masks = masks
            else:
                self._action_masks = None

        # Set up the encoder based on observation type.
        if len(self.cfg.image_keys) > 0:
            encoder_def = TDMPC2SimpleConv(
                num_channels=self.cfg.num_channels,
                apply_shift_aug=True,
                act=SimNorm(self.cfg.simnorm_dim),
            )
            img_encoder = PreprocessEncoder(
                encoder_def,
                normalize=True,
                resize=True,
                resize_shape=(224, 224),
            )
        # Check if inputs are provided as a dict with multiple keys.
        if (
            isinstance(self.cfg.input_shapes, dict)
            and len(self.cfg.input_shapes) > 1
        ):
            encoder_defs = {}
            for key in self.cfg.input_shapes:
                # Use a simple MLP for state inputs.
                if key.endswith("state"):
                    encoder_defs[key] = MLP(
                        (128, 128),
                        output_dim=self.cfg.latent_dim,
                        use_sim_norm=True,
                        sim_norm_dim=self.cfg.simnorm_dim,
                    )
                else:
                    encoder_defs[key] = img_encoder
            self.encoder = WithMappedEncoders(
                encoders=encoder_defs,
                network=nn.Sequential(
                    [nn.Dense(self.cfg.latent_dim), nn.relu]
                ),
                concatenate_keys=tuple(self.cfg.input_shapes.keys()),
            )
        else:
            if len(self.cfg.image_keys) > 0:
                self.encoder = img_encoder
            else:
                self.encoder = MLP((128, 128), output_dim=self.cfg.latent_dim)

        # Instantiate the rest of the submodules.
        self.dynamics = MLP((128, 128), output_dim=self.cfg.latent_dim)
        self.reward = MLP(
            (128, 128), output_dim=self.cfg.num_bins
        )  # 5 categories for discrete reward.
        self.pi_base = MLP((128, 128), output_dim=2 * self.cfg.action_dim)

    def __call__(
        self,
        batch_obs: Data,
        task: Optional[Union[int, jnp.ndarray]] = None,
    ):
        """
        Forward pass.

        Args:
            x: observations or a dict of observations.
            task: Task ID or array of task IDs (optional).

        Returns:
            next_z: Next latent state predicted by the dynamics network.
        """

        x = {}
        if self.cfg.input_shapes is not None:
            for k in self.cfg.input_shapes:
                if k.startswith("observation"):
                    # Safely access the observation without using tuple slicing
                    if k in batch_obs:
                        # Get the observation array
                        obs_array = batch_obs[k]
                        # Select all elements except the last one in the time dimension
                        # This avoids using the problematic [:, :-1] syntax
                        if obs_array.ndim > 1 and obs_array.shape[1] > 1:
                            # Get indices for all but the last element
                            time_indices = jnp.arange(obs_array.shape[1] - 1)
                            # Reshape for broadcasting - add extra dimensions to match obs_array
                            if obs_array.ndim == 3:  # batch, time, features
                                # Expand time_indices to match 3D array
                                # [1, time] -> [batch, time, 1] to match [batch, time, features]
                                batch_size = obs_array.shape[0]
                                feature_dim = obs_array.shape[2]
                                expanded_indices = time_indices.reshape(1, -1, 1)
                                expanded_indices = jnp.broadcast_to(expanded_indices, (batch_size, time_indices.shape[1], 1))
                                
                                # Need to create indices for all three dimensions
                                batch_indices = jnp.arange(batch_size).reshape(batch_size, 1, 1)
                                batch_indices = jnp.broadcast_to(batch_indices, (batch_size, time_indices.shape[1], 1))
                                
                                # Just select the elements along time dimension
                                x[k] = obs_array[:, :-1]
                            else:
                                # Original approach for 2D tensors
                                x[k] = jnp.take_along_axis(obs_array, time_indices, axis=1)
                        else:
                            # Just copy if no time dimension to slice
                            x[k] = obs_array
        a = batch_obs["action"]

        # Apply task conditioning if using multitask.
        if self.cfg.multitask and task is not None:
            # The get_task_embedding method (unchanged) combines x with an
            # embedded task representation.
            x = self.get_task_embedding(x, task)

        # Encode the input.
        z_enc = self.encoder(x)
        if a is not None:
            z = jnp.concatenate([z_enc, a], axis=-1)

        # Compute the next latent state.
        next_z = self.dynamics(z)

        # Computes and initializes reward and policy weights
        rewards = self.reward(jnp.concatenate([next_z, a], axis=-1))
        rng = self.make_rng("sample_actions")
        actions = self.pi(next_z, rng, task)

        return next_z

    def get_task_embedding(
        self, x: jnp.ndarray, task: Union[int, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Get task embedding and concatenate with input x.
        Args:
            x: Input tensor
            task: Task ID or array of task IDs
        Returns:
            Concatenated tensor of x and task embedding
        """
        if not self.cfg.multitask:
            return x

        # Convert task to array if needed
        if isinstance(task, int):
            task = jnp.array([task])

        # Get embedding
        emb = self.task_emb(task)

        # Handle different input dimensions
        if x.ndim == 5:  # [T, B, C, H, W] format
            emb = jnp.reshape(emb, (1, emb.shape[0], 1, emb.shape[1], 1))
            emb = jnp.tile(emb, (x.shape[0], 1, 1, 1, x.shape[-1]))
        elif x.ndim == 4:  # [B, C, H, W] format
            emb = jnp.reshape(emb, (emb.shape[0], 1, emb.shape[1], 1))
            emb = jnp.tile(emb, (1, 1, 1, x.shape[-1]))
        elif x.ndim == 3:  # [B, T, D] format
            emb = jnp.expand_dims(emb, 0)
            emb = jnp.tile(emb, (x.shape[0], 1, 1))
        elif emb.shape[0] == 1:  # Single task case
            emb = jnp.tile(emb, (x.shape[0], 1))

        return jnp.concatenate([x, emb], axis=-1)

    def encode(
        self, obs: jnp.ndarray, task: Optional[Union[int, jnp.ndarray]] = None
    ) -> jnp.ndarray:
        """
        Encode observations with optional task embedding.
        """
        if self.cfg.multitask and task is not None:
            obs = self.get_task_embedding(obs, task)

        return self.encoder(obs)

    def next(
        self,
        z: jnp.ndarray,
        action: jnp.ndarray,
        task: Optional[Union[int, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """
        Predict next latent state with optional task conditioning.
        """
        if self.cfg.multitask and task is not None:
            z = self.get_task_embedding(z, task)

        z_action = jnp.concatenate([z, action], axis=-1)
        return self.dynamics(z_action)

    def pi(
        self,
        z: jnp.ndarray,
        key: PRNGKey,
        task: Optional[Union[int, jnp.ndarray]] = None,
        eval_mode: bool = False,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Sample action from policy with optional task conditioning.
        """

        if self.cfg.multitask and task is not None:
            z = self.get_task_embedding(z, task)

        # Get mean and log_std
        base_out = self.pi_base(z)
        mean, log_std = jnp.split(base_out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.cfg.log_std_min, self.cfg.log_std_max)

        noise = jax.random.normal(rng, mean.shape)
        if eval_mode:
            action = mean
        else:
            # Sample with reparameterization
            action = mean + jnp.exp(log_std) * noise

        # Apply action masks for multitask case
        if self.cfg.multitask and task is not None:
            action_mask = self._action_masks[task]
            mean = mean * action_mask
            log_std = log_std * action_mask
            action = action * action_mask

        mean, action, log_prob = squash(mean, action, log_std)
        log_prob = gaussian_logprob(noise, log_std)

        # Scale log probability by action dimensions
        size = action.shape[-1]
        scaled_log_prob = log_prob * size

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def reward_fn(self, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.reward(jnp.concatenate([z, action], axis=-1))


class TDMPC2Critic(Critic):
    cfg: TDMPC2Config = TDMPC2Config()

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, action: jnp.ndarray, training: bool = True
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([z, action], -1)
        q = MLP(
            (self.cfg.mlp_dim, self.cfg.mlp_dim),
            output_dim=self.cfg.num_bins,
            activations=jax.nn.mish,
            use_normed_linear=True,
            use_sim_norm=False,
            dropout_rate=self.cfg.dropout,
        )(inputs, training=training)
        return q  # shape [batch, num_bins]


@jdc.pytree_dataclass
class AgentEvalState:
    observation_queue: collections.deque
    action_queue: collections.deque

    @classmethod
    def create(cls, obs_maxlen=1, act_maxlen=5):
        return cls(
            observation_queue=collections.deque(maxlen=obs_maxlen),
            action_queue=collections.deque(maxlen=act_maxlen),
        )

    def replace(self, **kwargs):
        return struct.replace(self, **kwargs)


# ----------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------
def mse_loss(val: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((val - target) ** 2)


def soft_ce(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int,
    vmin: float,
    vmax: float,
    bin_size: float,
) -> jnp.ndarray:
    """
    For the discrete classification approach. Expects 'labels' as a one-hot or scalar continuous values.
    """
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    target = two_hot(
        labels, num_bins, vmin, vmax, bin_size
    )  # shape [batch, num_bins]
    return -(target * log_prob).sum(-1, keepdims=True)


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric logarithmic function."""
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric exponential function."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    squashed_pi = jnp.log(jnp.maximum(1 - pi**2, 1e-6))
    log_pi = log_pi - squashed_pi.sum(-1, keepdims=True)
    return mu, pi, log_pi


def gaussian_logprob(eps, log_std):
    """Compute Gaussian log probability."""
    residual = -0.5 * jnp.power(eps, 2) - log_std
    log_prob = residual - 0.9189385175704956
    return log_prob.sum(-1, keepdims=True)


def two_hot_old(
    x: jnp.ndarray, num_bins: int, vmin: float, vmax: float, bin_size: float
) -> jnp.ndarray:
    """
    Convert a scalar to a two-hot encoded vector, JAX-optimized.
    """
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)

    # bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    x = jnp.clip(symlog(x), vmin, vmax)
    bin_idx = jnp.floor((x - vmin) / bin_size)
    bin_offset = (x - vmin) / bin_size - bin_idx

    soft_two_hot = jnp.zeros((x.shape[0], num_bins))
    bin_idx = bin_idx.astype(jnp.int32)
    soft_two_hot = soft_two_hot.at[:, bin_idx].set(1 - bin_offset)
    soft_two_hot = soft_two_hot.at[:, (bin_idx + 1) % num_bins].set(bin_offset)
    return soft_two_hot


def two_hot(
    x_input: jnp.ndarray,
    num_bins: int,
    vmin: float,
    vmax: float,
    bin_size: float,
) -> jnp.ndarray:
    """
    Converts a tensor of scalars to soft two-hot encoded targets for discrete regression.
    Output shape will be (*x_input.shape, num_bins).
    Matches the logic from the PyTorch version in common/math.py.
    """
    if num_bins == 0:
        return x_input  # No encoding
    elif num_bins == 1:
        return symlog(x_input)  # Symlog encode only

    original_shape = x_input.shape
    original_dtype = x_input.dtype  # Preserve original dtype for the output

    # Flatten all but the last dimension if x_input is not just a batch of scalars
    # Or treat the input as a flat batch of scalars to be processed
    x_flat = x_input.reshape(-1)

    # Perform calculations in float32 for precision, then cast back if needed
    _x = symlog(x_flat.astype(jnp.float32))
    _x = jnp.clip(_x, vmin, vmax)

    # Calculate bin indices and offsets
    bin_idx_float = (_x - vmin) / bin_size
    bin_idx_int = jnp.floor(bin_idx_float).astype(jnp.int32)

    # Clip indices to be within valid range [0, num_bins-1]
    # This is crucial for safety, especially if x can be exactly vmin or vmax.
    bin_idx_int = jnp.clip(bin_idx_int, 0, num_bins - 1)

    bin_offset_float = bin_idx_float - bin_idx_int.astype(jnp.float32)

    # Initialize the output tensor
    soft_two_hot_flat = jnp.zeros((_x.shape[0], num_bins), dtype=jnp.float32)
    flat_arange = jnp.arange(
        _x.shape[0]
    )  # Indices for the flattened dimension

    # Set the (1 - offset) part for the current bin
    current_bin_indices = bin_idx_int
    soft_two_hot_flat = soft_two_hot_flat.at[
        flat_arange, current_bin_indices
    ].set(1.0 - bin_offset_float)

    # Set the (offset) part for the next bin, with wraparound
    next_bin_indices = (bin_idx_int + 1) % num_bins
    # The PyTorch scatter operations overwrite. JAX .set() also overwrites.
    # Since current_bin_indices and next_bin_indices are distinct for num_bins > 1,
    # the second .set() correctly places the offset without interfering with the first.
    soft_two_hot_flat = soft_two_hot_flat.at[
        flat_arange, next_bin_indices
    ].set(bin_offset_float)

    # Reshape to (*original_shape, num_bins) and cast to original dtype
    return soft_two_hot_flat.reshape(*original_shape, num_bins).astype(
        original_dtype
    )


def two_hot_inv(logits: jnp.ndarray, cfg: TDMPC2Config) -> jnp.ndarray:
    """
    Example of converting discrete reward distribution back to a real value
    (like TDMPC2 does with 'math.two_hot_inv').
    For simplicity, we define categories = [-2, -1, 0, 1, 2].
    """
    if cfg.num_bins == 0:
        return logits
    elif cfg.num_bins == 1:
        return symexp(logits)

    dreg_bins = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_bins)
    probs = jax.nn.softmax(logits, axis=-1)
    x = jnp.sum(dreg_bins * probs, axis=-1, keepdims=True)
    return symexp(x)


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            "{}/{}".format(prefix, key): value
            for key, value in collected.items()
        }
    return collected


# ----------------------------------------------------------------------------------------
# TDMPC2 main logic
# ----------------------------------------------------------------------------------------
class TDMPC2Agent(struct.PyTreeNode):
    model: TrainState
    critic: TrainState
    target_critic: TrainState
    cfg: TDMPC2Config
    normalization_stats: Dict[str, Dict[str, np.ndarray]] = struct.field(
        default_factory=dict
    )
    normalization_modes: Dict[str, str] = struct.field(default_factory=dict)
    eval_state: Optional[AgentEvalState] = None
    _use_image: bool = False
    _use_env_state: bool = False
    input_image_key: Optional[str] = None
    rng: PRNGKey = struct.field(default_factory=lambda: jax.random.PRNGKey(0))

    def init_eval_state(self) -> "TDMPC2Agent":
        """Initialize evaluation state with empty queues."""
        return self.replace(
            eval_state=AgentEvalState.create(
                obs_maxlen=1,
                act_maxlen=max(
                    self.cfg.n_action_steps, self.cfg.n_action_repeats
                ),
            )
        )

    def update_eval_state(
        self,
        observations: Optional[Dict[str, jnp.ndarray]] = None,
        actions: Optional[jnp.ndarray] = None,
    ) -> "TDMPC2Agent":
        """Update observation and action queues in eval state."""
        if self.eval_state is None:
            self = self.init_eval_state()

        new_state = self.eval_state

        if observations is not None:
            queue = collections.deque(
                new_state.observation_queue,
                maxlen=new_state.observation_queue.maxlen,
            )
            queue.append(observations)
            new_state = new_state.replace(observation_queue=queue)

        if actions is not None:
            queue = collections.deque(
                new_state.action_queue, maxlen=new_state.action_queue.maxlen
            )
            if isinstance(actions, list):
                for action in actions:
                    queue.append(action)
            else:
                queue.append(actions)
            new_state = new_state.replace(action_queue=queue)

        return self.replace(eval_state=new_state)

    def Qs(
        self,
        z: jnp.ndarray,
        action: jnp.ndarray,
        task: Optional[Union[int, jnp.ndarray]] = None,
        return_type: str = "min",  # can be "min", "avg", or "all"
        target: bool = False,
        detach: bool = False,
    ) -> jnp.ndarray:
        """
        Compute Q-values with optional task conditioning.

        Args:
            z: latent state.
            action: action tensor.
            task: Optional task (for multitask conditioning).
            return_type: 'min' | 'avg' | 'all'
            target: if True, use target Q-network parameters.
            detach: if True, stop gradients from flowing through Qs.

        Returns:
            If return_type='all', returns the full stack [num_q, batch, 5].
            Otherwise, applies two-hot inverse and returns a (min or average) over Q heads.
        """
        if self.cfg.multitask and task is not None:
            z = self.get_task_embedding(z, task)

        if target:
            rngs = {"dropout": self.target_critic.key}
            all_qs = self.target_critic(z, action, rngs=rngs)
        elif detach:
            # Stop gradients if needed.
            rngs = {"dropout": self.critic.key}
            all_qs = jax.lax.stop_gradient(self.critic(z, action, rngs=rngs))
        else:
            rngs = {"dropout": self.critic.key}
            all_qs = self.critic(z, action, rngs=rngs)

        # Stack fields from the ensemble: shape [num_q, batch, 5]
        q_stack = jnp.stack(all_qs, axis=0)

        if return_type == "all":
            return q_stack

        # Convert the discrete Q distribution output into a scalar value using two_hot_inv.
        q_idx = jax.random.randint(rngs["dropout"], (2,), 0, self.cfg.num_q)
        q_values = two_hot_inv(q_stack[q_idx], self.cfg)  # shape [2, batch, 1]

        if return_type == "min":
            return jnp.min(q_values, axis=0)  # shape [batch, 1]
        elif return_type == "avg":
            return jnp.mean(q_values, axis=0)  # shape [batch, 1]
        else:
            raise ValueError(
                "Unsupported return_type. Expected 'min', 'avg', or 'all'."
            )

    def populate_queues(
        self,
        queues: Dict[str, collections.deque],
        batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, collections.deque]:
        """Populate the queues with new observations."""
        new_queues = dict(queues)
        for key in batch:
            if key in new_queues:
                new_queues[key].append(batch[key])
        return new_queues

    @partial(jax.jit, static_argnames=("agent",))
    def sample_actions(
        agent,
        observations: Dict[str, jnp.ndarray],
        *,
        seed: PRNGKey = None,
        task: Optional[int] = None,
    ) -> jnp.ndarray:
        """Select a single action given environment observations.

        Args:
            observations: Dictionary containing observations.
            seed: Optional random seed for reproducibility.
            task: Optional task index for multi-task settings.
        Returns:
            Selected action as a JAX array.
        """
        rng = seed if seed is not None else agent.rng

        # Define a local function to avoid passing frozen dictionaries as arguments
        def _normalize_with_agent_stats(obs):
            return normalize_inputs(
                obs, agent.normalization_stats, agent.normalization_modes
            )

        # Normalize inputs using the local function
        batch = _normalize_with_agent_stats(observations)
        if agent._use_image:
            batch = dict(batch)  # shallow copy
            batch["observation.image"] = batch[agent.input_image_key]

        # Update queues with new observations
        agent = agent.update_eval_state(observations=batch)

        # When action queue is depleted, populate it by querying policy
        if len(agent.eval_state.action_queue) == 0:
            # Stack queue contents
            batch = {
                key: jnp.stack(
                    list(agent.eval_state.observation_queue[key]), axis=1
                )
                for key in batch
            }

            # Remove time dimensions as it's not handled yet
            for key in batch:
                # Assert shape[1] == 1 for time dimension
                batch_val = batch[key]
                assert batch_val.shape[1] == 1
                batch[key] = batch_val[:, 0]

            # Encode observations in correct order
            encode_keys = []
            if agent._use_image:
                encode_keys.append("observation.image")
            if agent._use_env_state:
                encode_keys.append("observation.environment_state")
            encode_keys.append("observation.state")

            # Encode current observation
            z = agent.model.apply_fn.encode({k: batch[k] for k in encode_keys})

            # Get actions either through planning or direct policy
            if agent.cfg.mpc:
                actions = agent.plan(z, rng)  # (horizon, batch, action_dim)
            else:
                # Use policy directly - returns one action
                actions = jnp.expand_dims(agent.pi(z, rng)[0], 0)

            # Clip actions to [-1, 1]
            actions = jnp.clip(actions, -1.0, 1.0)

            # Unnormalize actions using local function to avoid passing FrozenDict directly
            def _unnormalize_with_agent_stats(action_data):
                return unnormalize_outputs(
                    action_data,
                    agent.normalization_stats,
                    agent.normalization_modes,
                )

            actions = _unnormalize_with_agent_stats({"action": actions})[
                "action"
            ]

            # Handle action repeats
            if agent.cfg.n_action_repeats > 1:
                # Ensure we have a valid action to repeat
                action_to_repeat = actions[0] if actions.ndim > 1 else actions
                agent = agent.update_eval_state(
                    actions=[action_to_repeat] * agent.cfg.n_action_repeats
                )

            else:
                # Extend action queue with planned actions
                agent = agent.update_eval_state(
                    actions=[a for a in actions[: agent.cfg.n_action_steps]]
                )

            # Return next action from queue
            action = jnp.array(agent.eval_state.action_queue.popleft())
            return action

    def encode(
        self,
        obs: Dict[str, jnp.ndarray],
        task: Optional[Union[int, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        return self.model.apply_fn.encode(obs, task)

    def next(
        self,
        z: jnp.ndarray,
        action: jnp.ndarray,
        task: Optional[Union[int, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        return self.model.apply_fn.next(z, action, task)

    def reward_fn(
        self,
        z: jnp.ndarray,
        action: jnp.ndarray,
        task: Optional[Union[int, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        return self.model.apply_fn.reward_fn(z, action, task)

    def pi(
        self,
        z: jnp.ndarray,
        rng: Optional[jax.random.PRNGKey] = None,
        eval_mode: bool = False,
    ):
        if rng is None:
            rng = self.model.key
        rngs = {"sample_actions": rng}
        return self.model(z, rng, eval_mode, method="pi", rngs=rngs)

    def plan(
        self,
        obs: jnp.ndarray,
        rng: Optional[PRNGKey] = None,
        eval_mode: bool = False,
        task: Optional[jnp.ndarray] = None,
    ):
        """
        Implementation of MPPI planning algorithm in JAX.
        Follows the TDMPC2 planning process from the PyTorch implementation.
        """
        if rng is None:
            rng = self.rng

        # Encode observation
        z = self.encode(obs)
        horizon = self.cfg.horizon
        num_samples = self.cfg.num_samples
        num_elites = self.cfg.num_elites
        batch_size = z.shape[0]

        # Initialize planning parameters
        mean = jnp.zeros((horizon, self.cfg.action_dim))
        std = jnp.full((horizon, self.cfg.action_dim), self.cfg.max_std)

        # For reproducibility
        rng, key = jax.random.split(rng)

        # Prepare latent state for planning
        z = jnp.repeat(z, num_samples, axis=0)

        # Initialize actions array
        actions = jnp.zeros((horizon, num_samples, self.cfg.action_dim))

        # MPPI iterations
        for _ in range(self.cfg.iterations):
            # Sample actions with noise
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(
                key, (horizon, num_samples, self.cfg.action_dim)
            )
            actions_noisy = (
                mean.reshape(horizon, 1, self.cfg.action_dim)
                + std.reshape(horizon, 1, self.cfg.action_dim) * noise
            )
            actions = jnp.clip(actions_noisy, -1.0, 1.0)

            # Estimate trajectory value
            def estimate_value(z0, actions):
                total_reward = 0.0
                discount = 1.0
                z_t = z0

                for t in range(horizon):
                    # Get reward for current state-action
                    reward_logits = self.model.apply_fn.reward_fn(
                        z_t, actions[t], task
                    )
                    reward = two_hot_inv(reward_logits, self.cfg)

                    # Update total reward
                    total_reward += discount * reward

                    # Transition to next state
                    z_t = self.model.apply_fn.next(z_t, actions[t], task)

                    # Update discount
                    discount *= self.cfg.discount

                # Add terminal value estimate
                final_action, _ = self.model.apply_fn.pi(z_t, rng, task)
                final_q = self.Qs(z_t, final_action, task, return_type="avg")

                return total_reward + discount * final_q

            # Compute trajectory values
            values = estimate_value(z, actions)

            # Select elite trajectories
            elite_idx = jnp.argsort(values.squeeze(), axis=0)[-num_elites:]
            elite_actions = jnp.take(actions, elite_idx, axis=1)
            elite_values = jnp.take(values, elite_idx, axis=0)

            # Compute weights for elite trajectories
            max_value = jnp.max(elite_values)
            weights = jnp.exp(
                self.cfg.temperature * (elite_values - max_value)
            )
            weights = weights / (jnp.sum(weights) + 1e-9)

            # Update mean and std from elite trajectories
            mean = jnp.sum(weights.reshape(1, -1, 1) * elite_actions, axis=1)
            std = jnp.sqrt(
                jnp.sum(
                    weights.reshape(1, -1, 1)
                    * (elite_actions - mean.reshape(horizon, 1, -1)) ** 2,
                    axis=1,
                )
            )
            std = jnp.clip(std, self.cfg.min_std, self.cfg.max_std)

        # Select best action
        best_idx = jnp.argmax(values.squeeze())
        best_action = actions[0, best_idx]

        # Add exploration noise if not in eval mode
        if not eval_mode:
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, best_action.shape) * std[0]
            best_action = jnp.clip(best_action + noise, -1.0, 1.0)

        return best_action

    def act(
        self,
        obs: jnp.ndarray,
        rng: Optional[jax.random.PRNGKey] = None,
        eval_mode: bool = False,
    ) -> jnp.ndarray:
        """
        Decide on an action for the environment.
        """
        if self.cfg.mpc:
            return self.plan(obs, rng, eval_mode)
        z = self.encode(obs)
        action, _info = self.pi(z, rng, eval_mode)
        return action[0]

    def update(
        agent, batch: Batch, pmap_axis: Optional[str] = None
    ) -> InfoDict:
        """
        A complete TDMPC2-style update, implementing the core TDMPC2 learning algorithm.
        This follows the approach from the PyTorch implementation with:
          1) A latent consistency rollout.
          2) Reward prediction.
          3) Q-function target for TD update.
          4) Policy update.
          5) Soft update of target Qs.
          6) An InfoDict of losses/statistics.

        Args:
            agent: An agent object with model, Qs, policy, etc.
            batch: A dictionary of JAX arrays including:
                'observations', 'actions', 'rewards', (optionally) 'task'.

        Returns:
            info: An InfoDict of training metrics.
        """

        # ----------------------------------------------------
        # 1) Parse the batch. If tasks are included, extract them.
        # ----------------------------------------------------
        obs = {}
        if agent.cfg.input_shapes is not None:
            for k in agent.cfg.input_shapes:
                if k.startswith("observation") and k in batch:
                    # Get the observation array
                    batch_data = batch[k]
                    # Select all elements except the last one in the time dimension
                    if batch_data.ndim > 1 and batch_data.shape[1] > 1:
                        # Get indices for all but the last element
                        time_indices = jnp.arange(batch_data.shape[1] - 1)
                        # Reshape for broadcasting
                        time_indices = time_indices.reshape(1, -1)
                        # Select using take_along_axis which is safer
                        obs[k] = jnp.take_along_axis(
                            batch_data, time_indices, axis=1
                        )
                    else:
                        # Just copy if no time dimension to slice
                        obs[k] = batch_data

        next_obs = {}
        if agent.cfg.input_shapes is not None:
            for k in agent.cfg.input_shapes:
                if k.startswith("observation") and k in batch:
                    # Get the observation array
                    batch_data = batch[k]
                    # Select all elements except the first one in the time dimension
                    if batch_data.ndim > 1 and batch_data.shape[1] > 1:
                        # Get indices for all elements starting from the second
                        time_indices = jnp.arange(1, batch_data.shape[1])
                        # Reshape for broadcasting
                        time_indices = time_indices.reshape(1, -1)
                        # Select using take_along_axis which is safer
                        next_obs[k] = jnp.take_along_axis(
                            batch_data, time_indices, axis=1
                        )
                    else:
                        # Just copy if no time dimension to slice
                        next_obs[k] = batch_data
        actions = batch["action"]
        rewards = batch["next.reward"]

        task = batch.get("task", None)  # optional

        # ----------------------------------------------------
        # 2) Compute next_z for the TD target (no gradient).
        # ----------------------------------------------------
        def td_target(next_z, rew):
            """
            Compute one-step TD target:
                T = rew + discount * Q(next_z, pi(next_z)), taking a minimum or average Q.
            Here we use a discrete representation for Q, so we do a two-hot inverse
            for both reward and Q-values, as in TDMPC2 practice.
            """
            # Get next action from the policy
            # Suppose agent.pi(...) returns (actions, info). Use a separate rng for sampling.
            rng, rng_sample = jax.random.split(agent.model.key, 2)
            next_act, _ = agent.model.pi(
                next_z, rng_sample, task=task, eval_mode=True
            )
            # Q of (s', a')
            q_next_logits = agent.Qs(
                next_z, next_act, task=task, target=True, return_type="min"
            )
            v_next = two_hot_inv(q_next_logits, agent.cfg).squeeze(
                -1
            )  # shape [B]
            # Compute two-hot inverse for the reward prediction if stored as discrete
            rew_val = rew
            # discount can be adjusted by agent.cfg if needed, here we keep it constant
            discount = agent.cfg.discount
            return rew_val + discount * v_next

        def policy_loss(zs, task):
            """
            Update policy using the latent states. Typically, TDMPC2 might do a
            model-based approach or a direct actor update. This version, for demonstration,
            uses discrete Q distribution methods from TDMPC2 logic.
            """
            # Sample actions from current policy
            pi_batched = jax.vmap(agent.model.pi, in_axes=(0, None, 0))
            sampled_actions, info = pi_batched(zs, agent.model.key, task)
            # print("zs.shape", zs.shape)
            # print("sampled_actions.shape", sampled_actions.shape)

            # Extract entropy from the policy output info
            entropy = info.get("scaled_entropy", 0.0)
            # print("entropy.shape", entropy.shape)

            # Evaluate Q
            q_vals = agent.Qs(
                zs, sampled_actions, task=task, return_type="avg", detach=True
            )  # [B, 1]

            # Calculate policy loss with entropy term
            rho = jnp.power(
                agent.cfg.rho,
                jnp.arange(q_vals.shape[1], dtype=jnp.float32)[None],
            )
            pi_loss = -jnp.mean(
                agent.cfg.entropy_coef * entropy + q_vals * rho
            )

            return pi_loss

        # ----------------------------------------------------
        # 3) Perform a multi-step latent rollout for consistency and gather states.
        #    We do "unroll" each step: z_{t+1} = model.next(z_t, action_t).
        # ----------------------------------------------------

        zt_buffer = []

        def td_error_loss(qs, td_targets, horizon_factor):
            """
            Computes the value loss across time and Q-ensemble.
            qs: (T, num_q, batch_size, num_classes)
            td_targets: (T, batch_size, num_classes)
            horizon_factor: float, scaling factor for the loss
            """
            value_loss = 0.0
            value_loss = jnp.sum(
                jax.vmap(
                    lambda q, td: soft_ce(
                        q,
                        td,
                        num_bins=agent.cfg.num_bins,
                        vmin=agent.cfg.vmin,
                        vmax=agent.cfg.vmax,
                        bin_size=agent.cfg.bin_size,
                    ),
                    in_axes=(0, 0),
                )(qs, td_targets)
            )
            return value_loss * horizon_factor

        def update_critic(critic_params):
            """
            Compute the critic loss using JAX-compatible vectorized operations.
            """
            # Clamp target values
            zt_stacked = jax.lax.stop_gradient(
                jnp.stack(zt_buffer, axis=1)
            )  # shape [B, T, D]
            qs_logits = agent.Qs(
                zt_stacked, actions, task=task, return_type="avg"
            )
            # qs_batched = jax.vmap(agent.Qs, in_axes=(0, 0))
            # qs_logits = jax.vmap(qs_batched, in_axes=(0, 0))(zt_stacked, actions, return_type="avg")
            # print("qs_logits", qs_logits.shape)
            td_targets = jax.vmap(td_target, in_axes=(0, 0))(
                zt_stacked, rewards
            )
            # print("td_targets", td_targets.shape)
            q_loss = td_error_loss(
                qs_logits, td_targets, 1 / (rewards.shape[1] - 1)
            )
            # print("q_loss", q_loss)
            info = {
                "td_targets": td_targets,
                "qs_logits": qs_logits,
                "q_loss": q_loss,
            }
            return q_loss, info

        def update_model(params):
            """
            Rollout the latent model for each step to compute:
              - Consistency loss between predicted latent and actual next latent.
              - Reward loss for the predicted reward distribution.
              - Q-value loss from a TD perspective.
              - Policy update.
            """
            # Step-by-step consistency
            consistency_loss = 0.0
            # If we have horizon steps in obs, each dimension should be [T, B, D]
            # We'll do a single-step or multi-step approach based on shapes.
            T = rewards.shape[1]

            # Collect partial TD targets
            # td_targets_all = []
            # We'll collect Q logits as well for each step
            # q_logits_all = []
            reward_loss = 0.0

            # For each step t in [0..T-1], do a forward rollout
            zt_next = None
            for t in range(T - 1):
                # Current z
                obs_t = {}
                if agent.cfg.input_shapes is not None:
                    for k in agent.cfg.input_shapes:
                        if k.startswith("observation") and k in obs:
                            # Get a specific timestep from observation sequence
                            # Using JAX's indexing to avoid tuple slicing issues
                            obs_array = obs[k]
                            # Create a single element index for the time dimension
                            time_idx = jnp.array([[t]])
                            # Extract the t-th timestep safely
                            obs_t[k] = jnp.squeeze(
                                jnp.take_along_axis(
                                    obs_array, time_idx, axis=1
                                ),
                                axis=1,
                            )

                obs_t_next = {}
                if agent.cfg.input_shapes is not None:
                    for k in agent.cfg.input_shapes:
                        if k.startswith("observation") and k in next_obs:
                            # Get a specific timestep from next_observation sequence
                            next_obs_array = next_obs[k]
                            # Create a single element index for the time dimension
                            time_idx = jnp.array([[t]])
                            # Extract the t-th timestep safely
                            obs_t_next[k] = jnp.squeeze(
                                jnp.take_along_axis(
                                    next_obs_array, time_idx, axis=1
                                ),
                                axis=1,
                            )
                zt = (
                    agent.model.encode(obs_t, task)
                    if zt_next is None
                    else zt_next
                )
                zt_buffer.append(zt)
                # Next z from model
                # Get actions for the current timestep using JAX's indexing
                time_idx = jnp.array([[t]])
                actions_t = jnp.squeeze(
                    jnp.take_along_axis(actions, time_idx, axis=1), axis=1
                )
                zt_next_pred = agent.model.next(zt, actions_t, task=task)
                # Next z from data
                zt_next = agent.model.encode(obs_t_next, task=task)
                # Consistency
                consistency_loss += jnp.mean((zt_next_pred - zt_next) ** 2)

                # Reward prediction
                # Get actions for current timestep
                time_idx = jnp.array([[t]])
                actions_t = jnp.squeeze(
                    jnp.take_along_axis(actions, time_idx, axis=1), axis=1
                )

                reward_pred_logits = agent.reward_fn(
                    zt, actions_t
                )  # shape [B, 5]

                # Extract rewards for current timestep
                rewards_t = jnp.squeeze(
                    jnp.take_along_axis(rewards, time_idx, axis=1), axis=1
                )

                true_rew = jax.nn.one_hot(
                    rewards_t.astype(jnp.int32), reward_pred_logits.shape[-1]
                )
                rew_loss = soft_ce(
                    reward_pred_logits,
                    true_rew,
                    agent.cfg.num_bins,
                    agent.cfg.vmin,
                    agent.cfg.vmax,
                    agent.cfg.bin_size,
                )
                reward_loss += rew_loss

                # Q distribution for Q(s,a)
                # qs_logits = agent.Qs(zt, actions[:, t], task=task)  # shape [num_q, B, 5]
                # q_logits_all.append(qs_logits)

                # Next z for TD
                # We'll do a 1-step TD target
                # td_val = td_target(zt_next, rewards[:, t])
                # td_targets_all.append(td_val)

            zt_buffer.append(zt_next)

            # Average consistency and reward loss
            # to mirror the TDMPC2 idea of weighting by horizon
            horizon_factor = 1 / float(T - 1) if T > 1 else 1.0
            consistency_loss = consistency_loss * horizon_factor
            reward_loss = reward_loss * horizon_factor

            # Compute Q loss using the new critic_loss_fn
            # q_loss = td_error_loss(
            #     q_logits_all, td_targets_all, horizon_factor / agent.cfg.num_q
            # )

            # Now, for a policy update (like in TDMPC2) we sample a sequence of latent states
            # again. For simplicity, just use the first T states:
            # We combine them along batch dimension. Then compute the policy loss.
            # In TDMPC2, we often do a separate loop or call pi_loss.
            pi_loss = 0.0
            zt_stacked = jnp.stack(zt_buffer, axis=1)  # shape [B, T, D]
            pi_loss += policy_loss(zt_stacked, task)
            pi_loss = pi_loss / float(T)

            # Weighted sum, or just add them as recommended by TDMPC2
            total_loss_with_pi = (
                consistency_loss + reward_loss + pi_loss
            )  # q_loss is not included

            # Return losses to keep track
            losses_dict = {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "pi_loss": pi_loss,
                "total_loss": total_loss_with_pi,
                # "q_loss": q_loss,
            }
            return total_loss_with_pi.mean(), losses_dict

        # Update world model, policy, and critic parameters
        print("about to update model")
        new_model, info_dict = agent.model.apply_loss_fn(
            loss_fn=update_model, has_aux=True, pmap_axis=pmap_axis
        )
        print("about to update critic")
        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=update_critic, has_aux=True, pmap_axis=pmap_axis
        )
        print("info_dict keys", info_dict.keys())
        print("critic_info keys", critic_info.keys())
        info = {}
        for k in info_dict.keys():
            if isinstance(info_dict[k], jnp.ndarray):
                if info_dict[k].size == 1:
                    info[k] = info_dict[k]
                else:
                    info[k] = info_dict[k].mean(axis=0)
            else:
                info[k] = info_dict[k]
        for k in critic_info.keys():
            if isinstance(critic_info[k], jnp.ndarray):
                if critic_info[k].size == 1:
                    info[k] = critic_info[k]
                else:
                    info[k] = critic_info[k].mean(axis=0)

        # Soft update target critic
        new_target_critic = target_update(
            agent.critic, agent.target_critic, agent.cfg.tau
        )

        # Update agent's rng so it changes after each update.
        rng_updated = jax.random.split(agent.rng, 2)[0]

        # Return updated agent and the info
        updated_agent = agent.replace(
            rng=rng_updated,
            model=new_model,
            critic=new_critic,
            target_critic=new_target_critic,
        )
        print("updated agent")
        return updated_agent, info


def create_q_ensemble(cfg):
    """
    Build an ensemble of Q networks externally.

    Args:
        cfg: A config object containing at least cfg.num_q and cfg.critic_lr.

    Returns:
        A TrainState containing the Q ensemble.
    """
    # Define the ensemble using ensemblize along with any additional settings.
    q_def = ensemblize(TDMPC2Critic, cfg.num_q, out_axes=0)
    # Instantiate the ensemble by passing the hidden dims.
    q_net = q_def(cfg)
    return q_net


def create_tdmpc2_learner(
    config: TDMPC2Config,
    rng: jax.random.PRNGKey,
    normalization_stats: Dict = None,
    normalization_modes: Dict = None,
    shape_meta: Dict = None,
    **kwargs,
) -> TDMPC2Agent:
    """Create a TDMPC2 agent with proper initialization."""

    # Debug the input parameters
    print(f"DEBUG create_tdmpc2_learner: normalization_stats type: {type(normalization_stats)}")
    print(f"DEBUG create_tdmpc2_learner: normalization_modes type: {type(normalization_modes)}")
    print(f"DEBUG create_tdmpc2_learner: shape_meta type: {type(shape_meta)}")
    
    # Ensure normalization_modes is a dictionary before freezing
    if normalization_modes is None:
        print("DEBUG create_tdmpc2_learner: normalization_modes is None, creating empty dict")
        normalization_modes = {}
    
    # jax.tree_util.register_dataclass(TDMPC2Config)
    normalization_stats = frozen_dict.freeze(normalization_stats)
    normalization_modes = frozen_dict.freeze(normalization_modes)
    print(f"DEBUG create_tdmpc2_learner: After freezing - normalization_modes: {normalization_modes}")
    input_shapes = shape_meta["input_shapes"]

    # Create zero tensors for inputs - ensure consistent dimensions
    batch_obs = {}
    print(f"DEBUG create_tdmpc2_learner: input_shapes = {input_shapes}")
    
    for key, shape in input_shapes.items():
        # Ensure each observation has at least 3 dimensions (batch, time, features)
        if len(shape) == 2:  # If only (batch, features)
            # Add a time dimension
            batch_obs[key] = jnp.zeros((shape[0], 2, shape[1]))  # Add time dim with 2 timesteps
            print(f"DEBUG create_tdmpc2_learner: Expanded {key} from {shape} to {batch_obs[key].shape}")
        else:
            batch_obs[key] = jnp.zeros(shape)
    
    print(f"DEBUG create_tdmpc2_learner: batch_obs shapes after input creation:")
    for k, v in batch_obs.items():
        print(f"DEBUG create_tdmpc2_learner:   {k}: shape={v.shape}, ndim={v.ndim}")
    
    # Handle output shapes - ensure action has correct dimensions too
    for k in shape_meta["output_shape"]:
        out_shape = shape_meta["output_shape"][k]
        if len(out_shape) == 2:  # If only (batch, action_dim)
            # Add a time dimension for actions too
            batch_obs[k] = jnp.zeros((out_shape[0], 2, out_shape[1]))
            print(f"DEBUG create_tdmpc2_learner: Expanded {k} from {out_shape} to {batch_obs[k].shape}")
        else:
            batch_obs[k] = jnp.zeros(out_shape)
    
    print(f"DEBUG create_tdmpc2_learner: batch_obs shapes after output update:")
    for k, v in batch_obs.items():
        print(f"DEBUG create_tdmpc2_learner:   {k}: shape={v.shape}, ndim={v.ndim}")

    # Initialize model
    rng, dropout_key1, dropout_key2 = jax.random.split(rng, 3)
    critic_def = create_q_ensemble(config)
    critic_params = critic_def.init(
        rng,
        jnp.zeros((1, config.latent_dim)),
        jnp.zeros((1, config.action_dim)),
        training=False,
    )["params"]
    critic = TrainState.create(
        critic_def,
        critic_params,
        tx=optax.adam(learning_rate=config.lr),
        key=dropout_key1,
    )
    target_critic = TrainState.create(
        critic_def,
        critic_params,
        tx=optax.adam(learning_rate=config.lr),
        key=dropout_key2,
    )

    rng, model_rng, sample_rng = jax.random.split(rng, 3)
    model_def = WorldModel(config)
    rngs = {"params": model_rng, "sample_actions": sample_rng}
    # Initialize the model without try/catch now that we've fixed the dimensions
    print(f"DEBUG create_tdmpc2_learner: About to initialize model with batch_obs shapes:")
    for k, v in batch_obs.items():
        print(f"DEBUG create_tdmpc2_learner:   {k}: shape={v.shape}, ndim={v.ndim}")
    params = model_def.init(rngs, batch_obs)["params"]
    model = TrainState.create(
        model_def,
        params,
        tx=optax.adam(learning_rate=config.lr),
        key=sample_rng,
    )

    # Determine observation types
    use_image = config.obs == "rgb"
    use_env_state = "observation.environment_state" in config.input_shapes  # type: ignore
    input_image_key = None
    if use_image and config.input_shapes is not None:
        image_keys = [
            k for k in config.input_shapes if k.startswith("observation.image")
        ]
        assert len(image_keys) == 1, (
            "Expected exactly one image observation key"
        )
        input_image_key = image_keys[0]

    # Create agent
    agent = TDMPC2Agent(
        rng=rng,
        model=model,
        critic=critic,
        target_critic=target_critic,
        cfg=config,
        _use_image=use_image,
        _use_env_state=use_env_state,
        input_image_key=input_image_key,
        normalization_stats=normalization_stats,
        normalization_modes=normalization_modes,
    )

    # Initialize queues
    agent = agent.init_eval_state()

    return agent


def main(_):
    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.envs.factory import make_env

    # Create wandb logger
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    # Setup save directory if needed
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    # Load environment config
    dataset_cfg = omegaconf.DictConfig(
        {
            "dataset_repo_id": "lerobot/d3il_sorting_example",
            "env": {"name": "sorting"},
            "training": {"image_transforms": {"enable": False}},
            "dataset_root": None,
            "device": "cpu",  # Keep dataset on CPU for JAX transfer
        }
    )

    # Create environment and dataset
    env = make_env(dataset_cfg)
    dataset = make_dataset(dataset_cfg, split="train")
    print(f"Loaded dataset of length {len(dataset)} on CPU")

    # Create numpy dataloader
    from jax.tree_util import tree_map
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True
    )
    train_iter = iter(train_loader)

    # Initialize agent
    rng = jax.random.PRNGKey(FLAGS.seed)
    agent = create_tdmpc2_learner(FLAGS.config, rng, dataset)

    # Training loop
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        # Get next batch (with restart if needed)
        try:
            batch = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Convert relevant parts of batch to jnp arrays
        obs = jnp.array(batch["observation.state"])
        next_obs = jnp.array(batch["next.observation.state"])
        actions = jnp.array(batch["action"])
        rewards = jnp.array(batch["next.reward"])

        # Update step
        metrics = agent.train_step(
            obs_batch=obs,
            action_batch=actions,
            reward_batch=rewards,
            next_obs_batch=next_obs,
            rng=rng,
        )

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in metrics.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            # Create a policy function that uses agent.plan
            def policy_fn(obs, rng):
                return agent.plan(obs, rng, eval_mode=True)

            eval_info = evaluate(
                policy_fn, env, num_episodes=FLAGS.eval_episodes
            )
            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(
                FLAGS.save_dir,
                agent.model.params,  # Save just the parameters
                step=i,
            )


if __name__ == "__main__":
    # Define flags similar to run_d4rl_iql.py
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "save_dir", None, "Logging dir (if not None, save params)."
    )
    flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
    flags.DEFINE_integer(
        "eval_episodes", 10, "Number of episodes used for evaluation."
    )
    flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
    flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
    flags.DEFINE_integer("save_interval", 25000, "Save interval.")
    flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
    flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
    flags.DEFINE_string(
        "env_config",
        "../lerobot/lerobot/configs/env/d3il_sorting.yaml",
        "Path to the environment config.",
    )

    # Add wandb config
    wandb_config = default_wandb_config()
    wandb_config.update(
        {
            "project": "lerobot_tdmpc2",
            "group": "tdmpc2_test",
            "name": "tdmpc2_{env_config}",
        }
    )
    config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)

    # Convert TDMPC2Config to a dictionary and then to a ConfigDict
    config_dict = ConfigDict(vars(TDMPC2Config()))
    config_flags.DEFINE_config_dict("config", config_dict, lock_config=False)
    app.run(main)
