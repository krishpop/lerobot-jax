import collections
import os
import pickle
from functools import partial
from typing import Dict

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
from jaxrl_m.typing import *
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
from lerobot_jax.utils.norm_utils import normalize_transform

# -------------
# Config class
# -------------


@jdc.pytree_dataclass
class TDMPC2Config:
    """
    Configuration for TDMPC2. Adjust or extend as necessary.
    """

    normalization_stats: jdc.Static[FrozenDict[str, Dict[str, jnp.ndarray]]] = None
    normalization_modes: jdc.Static[FrozenDict[str, str]] = None

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
    input_shapes: Dict[str, str] = None
    image_keys: Tuple[str, ...] = ()
    output_shapes: Dict[str, str] = None
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
    mpc: bool = False
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
    # TD-M(PC)^2 specific
    use_policy_constraint: bool = False  # Flag to enable TD-M(PC)^2 policy regularization
    prior_constraint_coef: float = 1.0  # Beta: Strength of the policy regularization term
    adaptive_regularization: bool = False # Flag to enable adaptive curriculum for beta
    scale_threshold: float = 2.0  # s_threshold: Q-function percentile threshold for adaptive beta


    @property
    def bin_size(self) -> float:
        return (self.vmax - self.vmin) / (self.num_bins - 1)

    @property
    def discount(self) -> float:
        """
        Compute the discount factor based on episode_length, discount_denom,
        discount_min, and discount_max.
        Calculation: discount = (episode_length - discount_denom) / episode_length,
        then clamped between discount_min and discount_max.
        """
        base = (self.episode_length - self.discount_denom) / self.episode_length
        return min(max(base, self.discount_min), self.discount_max)


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
                act=SimNorm(self.cfg),
            )
            img_encoder = PreprocessEncoder(
                encoder_def, normalize=True, resize=True, resize_shape=(224, 224)
            )
        # Check if inputs are provided as a dict with multiple keys.
        if isinstance(self.cfg.input_shapes, dict) and len(self.cfg.input_shapes) > 1:
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
                encoders=frozen_dict.freeze(encoder_defs),
                network=nn.Sequential([nn.Dense(self.cfg.latent_dim), nn.relu]),
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

        x = {
            k: batch_obs[k][:, :-1]
            for k in self.cfg.input_shapes
            if k.startswith("observation")
        }
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

    def get_task_embedding(self, x: Data, task: Union[int, jnp.ndarray]) -> Data:
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
            task = jax.nn.one_hot(task, len(self.cfg.tasks))

        # Get embedding
        emb = self.task_emb(task)

        # Handle different input dimensions
        if isinstance(x, dict):
            for key, value in x.items():
                if value.ndim == 3:
                    emb = jnp.expand_dims(emb, 0)
                    emb = jnp.tile(emb, (value.shape[0], 1, 1))
                elif emb.shape[0] == 1:  # Single task case
                    emb = jnp.tile(emb, (value.shape[0], 1))
                x[key] = jnp.concatenate([value, emb], axis=-1)
        else:
            if x.ndim == 3:  # [B, T, D] format
                emb = jnp.expand_dims(emb, 0)
                emb = jnp.tile(emb, (x.shape[0], 1, 1))
            elif emb.shape[0] == 1:  # Single task case
                emb = jnp.tile(emb, (x.shape[0], 1))
            x = jnp.concatenate([x, emb], axis=-1)
        return x

    def encode(
        self, obs: Data, task: Optional[Union[int, jnp.ndarray]] = None
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
        rng: jax.random.PRNGKey,
        task: Optional[Union[int, jnp.ndarray]] = None,
        eval_mode: bool = False,
    ):
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
    """State maintained during evaluation."""

    from collections import deque

    observation_queue: deque = struct.field(default_factory=lambda: deque(maxlen=2))
    action_queue: deque = struct.field(default_factory=lambda: deque(maxlen=5))


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
    For the discrete classification approach. Expects 'labels' as a one-hot or integer class.
    """
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    target = two_hot(labels, num_bins, vmin, vmax, bin_size)
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


def two_hot(
    x: jnp.ndarray, num_bins: int, vmin: float, vmax: float, bin_size: float
) -> jnp.ndarray:
    """
    Convert a scalar to a two-hot encoded vector.
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
    soft_two_hat = soft_two_hot.at[:, bin_idx].set(1 - bin_offset)
    soft_two_hat = soft_two_hot.at[:, (bin_idx + 1) % num_bins].set(bin_offset)
    return soft_two_hat


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
            "{}/{}".format(prefix, key): value for key, value in collected.items()
        }
    return collected


# ----------------------------------------------------------------------------------------
# TDMPC2 main logic
# ----------------------------------------------------------------------------------------
class TDMPC2Agent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    model: TrainState
    critic: TrainState
    target_critic: TrainState
    cfg: TDMPC2Config
    # normalization_stats: FrozenDict[str, Dict[str, jnp.ndarray]] = None
    # normalization_modes: FrozenDict[str, str] = None
    eval_state: Optional[AgentEvalState] = None
    _use_image: bool = False
    _use_env_state: bool = False
    input_image_key: Optional[str] = None

    def init_eval_state(self) -> "TDMPC2Agent":
        """Initialize evaluation state with empty queues."""
        queues = {
            "observation.state": collections.deque(maxlen=1),
            "action": collections.deque(
                maxlen=max(self.cfg.n_action_steps, self.cfg.n_action_repeats)
            ),
        }
        if self._use_image:
            queues["observation.image"] = collections.deque(maxlen=1)
        if self._use_env_state:
            queues["observation.environment_state"] = collections.deque(maxlen=1)

        return self.replace(
            eval_state=AgentEvalState(
                observation_queue=queues["observation.state"],
                action_queue=queues["action"],
            )
        )

    def update_eval_state(
        self,
        observations: Optional[Dict[str, np.ndarray]] = None,
        actions: Optional[np.ndarray] = None,
    ) -> "TDMPC2Agent":
        """Update observation and action queues in eval state."""
        if self.eval_state is None:
            self = self.init_eval_state()

        new_state = self.eval_state

        if observations is not None:
            queue = new_state.observation_queue
            queue.append(observations)

            # Pad queue with first observation if not full
            while len(queue) < queue.maxlen:
                queue.append(observations)

            new_state = self.eval_state.replace(observation_queue=queue)

        if actions is not None:
            queue = new_state.action_queue
            queue.extend(actions)

            # Pad queue with first action if not full
            if len(queue) < queue.maxlen:
                first_action = queue[-1]
                while len(queue) < queue.maxlen:
                    queue.extend(first_action)

            new_state = self.eval_state.replace(action_queue=queue)
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
        self, queues: Dict[str, collections.deque], batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, collections.deque]:
        """Populate the queues with new observations."""
        new_queues = dict(queues)
        for key in batch:
            if key in new_queues:
                new_queues[key].append(batch[key])
        return new_queues

    def sample_actions(
        agent,
        observations: Dict[str, jnp.ndarray],
        *,
        seed: PRNGKey = None,  # noqa: F405
    ) -> jnp.ndarray:
        """Select a single action given environment observations.

        Args:
            observations: Dictionary containing observations.
            seed: Optional random seed for reproducibility.
        Returns:
            Selected action as a JAX array.
        """
        rng = jax.random.PRNGKey(seed) if seed is not None else agent.model.key
        # Normalize inputs
        # We need to avoid passing JAX arrays as dict keys which causes unhashable errors
        batch = dict(observations)  # Shallow copy to avoid modifying the original batch.
        
        # Convert normalization stats to Python primitives where needed
        if agent.cfg.normalization_stats is not None and agent.cfg.normalization_modes is not None:
            for key, value in observations.items():
                # Skip keys not in our normalization stats
                if key not in agent.cfg.normalization_stats or key not in agent.cfg.normalization_modes:
                    continue
                
                # Get stats and mode as Python values, not JAX arrays
                stats = dict(agent.cfg.normalization_stats[key])
                mode = agent.cfg.normalization_modes[key]
                
                # Apply normalization directly with the normalize_transform function
                if mode == "mean_std":
                    mean = stats["mean"]
                    std = stats["std"]
                    batch[key] = (value - mean) / (std + 1e-8)
                elif mode == "min_max":
                    breakpoint()
                    min_val = stats["min"]
                    max_val = stats["max"]
                    normalized = (value - min_val) / (max_val - min_val + 1e-8)
                    batch[key] = normalized * 2.0 - 1.0

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
                    [obs[key] for obs in agent.eval_state.observation_queue], axis=1
                )
                for key in batch
            }

            # Remove time dimensions as it's not handled yet
            for key in batch:
                # Assert shape[1] == 1 for time dimension
                assert batch[key].shape[1] == 1
                batch[key] = batch[key][:, 0]

            # Encode observations in correct order
            encode_keys = []
            if agent._use_image:
                encode_keys.append("observation.image")
            if agent._use_env_state:
                encode_keys.append("observation.environment_state")
            encode_keys.append("observation.state")

            # Encode current observation
            z = agent.encode({k: batch[k] for k in encode_keys})

            # Get actions either through planning or direct policy
            if agent.cfg.mpc:
                actions = agent.plan(z, rng)  # (horizon, batch, action_dim)
            else:
                # Use policy directly - returns one action
                actions = agent.pi(z, rng)[0][None]

            # Clip actions to [-1, 1]
            actions = jnp.clip(actions, -1.0, 1.0)

            # Unnormalize actions
            actions = normalize_transform(actions, agent.cfg.normalization_stats["action"], agent.cfg.normalization_modes["action"], unnormalize=True)

            # Handle action repeats
            if agent.cfg.n_action_repeats > 1:
                agent = agent.update_eval_state(
                    actions=[actions[0]] * agent.cfg.n_action_repeats
                )
            else:
                # Extend action queue with planned actions
                for act in actions[: agent.cfg.n_action_steps]:
                    agent.eval_state.action_queue.append(act)

        # Return next action from queue
        action = agent.eval_state.action_queue.popleft()
        breakpoint()
        return action

    def encode(self, obs: Data) -> jnp.ndarray:
        return self.model.encode(obs)

    def next(self, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.model.next(z, action)

    def reward_fn(self, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.model.reward_fn(z, action)

    def pi(
        self,
        z: jnp.ndarray,
        rng: Optional[jax.random.PRNGKey] = None,  # noqa: F405
        eval_mode: bool = False,
    ):
        if rng is None:
            rng = self.model.key
        rngs = {"sample_actions": rng}
        return self.model(z, rng, eval_mode, method="pi", rngs=rngs)

    def plan(
        self,
        z: jnp.ndarray,
        rng: Optional[jax.random.PRNGKey] = None,
        eval_mode: bool = False,
    ):
        """
        Example of MPPI or CEM-based plan in JAX.
        For demonstration, we do a trivial approach here that picks best from random sampling.
        """
        if rng is None:
            rng = self.model.key

        # Encode obs
        horizon = self.cfg.horizon
        samples = self.cfg.num_samples

        # We'll do random sampling
        if rng is None:
            rng = self.rng
        rng, rand_action_rng = jax.random.split(rng)
        rand_actions = jax.random.uniform(
            rand_action_rng,
            shape=(horizon, samples, self.cfg.action_dim),
            minval=-1,
            maxval=1,
        )

        def scan_trajectory(z0):
            # z0 shape [latent_dim], we broadcast to [samples, latent_dim]
            z_samp = jnp.expand_dims(z0, 0).repeat(samples, axis=0)
            rews = []
            for tstep in range(horizon):
                rew_logits = self.reward_fn(z_samp, rand_actions[tstep])
                rew = two_hot_inv(rew_logits, self.cfg)
                rews.append(rew)
                z_samp = self.next(z_samp, rand_actions[tstep])
            # approximate value by final Q
            # We'll take standard approach: pick Q_min from our two Qs
            # Just for demonstration, we do Q-avg
            final_q_logits = self.Qs(
                z_samp, jnp.zeros_like(rand_actions[0])
            )  # zero act
            # shape [num_q, batch, 5]
            final_q_values = jnp.mean(final_q_logits, axis=0)
            final_val = two_hot_inv(final_q_values, self.cfg)
            # sum up rewards
            total_rew = jnp.sum(jnp.concatenate(rews, axis=-1), axis=-1, keepdims=True)
            return total_rew + final_val

        returns = scan_trajectory(z[0])
        best_idx = jnp.argmax(returns[:, 0])  # returns shape [samples, 1]
        best_action = rand_actions[0, best_idx]
        # Optionally add exploration if not eval
        if not eval_mode:
            noise = jax.random.normal(rng, best_action.shape) * 0.1
            best_action = jnp.clip(best_action + noise, -1, 1)
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

    def update(agent, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:  # noqa: F405
        """
        A complete TDMPC2-style update, incorporating logic similar to tdmpc2.py.
        It includes:
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
        obs = {
            k: batch[k][:, :-1]
            for k in agent.cfg.input_shapes
            if k.startswith("observation")
        }
        next_obs = {
            k: batch[k][:, 1:]
            for k in agent.cfg.input_shapes
            if k.startswith("observation")
        }
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
            next_act, _ = agent.model.pi(next_z, rng_sample, task=task, eval_mode=True)
            # Q of (s', a')
            q_next_logits = agent.Qs(
                next_z, next_act, task=task, target=True, return_type="min"
            )
            v_next = two_hot_inv(q_next_logits, agent.cfg).squeeze(-1)  # shape [B]
            # Compute two-hot inverse for the reward prediction if stored as discrete
            rew_val = rew
            # discount can be adjusted by agent.cfg if needed, here we keep it constant
            discount = agent.cfg.discount
            return rew_val + discount * v_next

        def policy_loss(params, zs, actions_from_buffer, behavior_mean, behavior_log_std, task): # Added behavior_mean, behavior_log_std
            # Sample actions a_pi ~ π(.|zs)
            sampled_actions, pi_info = agent.model.apply_fn(
                {'params': params}, zs, agent.model.key, task=task, eval_mode=False, method=WorldModel.pi
            )
            
            log_prob_pi = -pi_info.get("entropy") # entropy is -log_prob.sum(), so -entropy is log_prob.sum()

            # Q-values for these actions
            q_values_for_sampled_actions = agent.Qs(
                zs, sampled_actions, task=task, return_type="avg", detach=True # Detach Qs from policy gradient
            )
            q_values_for_sampled_actions = q_values_for_sampled_actions.squeeze(-1)

            sampled_actions_pre_squash = jnp.arctanh(jnp.clip(sampled_actions, -1+1e-6, 1-1e-6))
            
            # Noise term for μ
            noise_for_mu = (sampled_actions_pre_squash - behavior_mean) / (jnp.exp(behavior_log_std) + 1e-6)
            
            # Log prob of raw actions under μ's Gaussian
            log_prob_mu_raw = gaussian_logprob(noise_for_mu, behavior_log_std) # This is sum over action dim
            
            # Squashing correction for μ (using sampled_actions which are already squashed)
            squash_correction_mu = jnp.log(jnp.maximum(1 - sampled_actions**2, 1e-6)).sum(axis=-1, keepdims=True)
            log_prob_mu = log_prob_mu_raw - squash_correction_mu

            # Adaptive beta
            current_beta = agent.cfg.prior_constraint_coef
            if agent.cfg.adaptive_regularization:
                # This needs access to Q-value statistics, placeholder for now
                # q_percentile = ... (compute moving percentile of Q values from critic updates)
                # if q_percentile < agent.cfg.scale_threshold:
                #     current_beta = 0.0
                # This logic might be better placed in the main update loop and beta passed to policy_loss
                pass


            # TD-M(PC)^2 Policy Loss (Equation 10 from paper [cite: 119])
            # L = E [ Q(s,a) - alpha * log pi(a|s) + beta * log mu(a|s) ]
            # We want to MAXIMIZE this, so for gradient ASCENT, the loss is -L
            # Or for gradient DESCENT, the loss is L_pi = E [ -Q + alpha*log pi - beta*log mu ]
            
            pi_loss = - (
                q_values_for_sampled_actions # Q(s, a_pi)
                - agent.cfg.entropy_coef * log_prob_pi.squeeze(-1) # alpha * log pi(a_pi|s)
                + current_beta * log_prob_mu.squeeze(-1) # beta * log mu(a_pi|s)
            )
            pi_loss = jnp.mean(pi_loss)

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
            qs_logits = agent.Qs(zt_stacked, actions, task=task, return_type="avg")
            # qs_batched = jax.vmap(agent.Qs, in_axes=(0, 0))
            # qs_logits = jax.vmap(qs_batched, in_axes=(0, 0))(zt_stacked, actions, return_type="avg")
            # print("qs_logits", qs_logits.shape)
            td_targets = jax.vmap(td_target, in_axes=(0, 0))(zt_stacked, rewards)
            # print("td_targets", td_targets.shape)
            q_loss = td_error_loss(qs_logits, td_targets, 1 / (rewards.shape[1] - 1))
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
                obs_t = {
                    k: obs[k][:, t]
                    for k in agent.cfg.input_shapes
                    if k.startswith("observation")
                }
                obs_t_next = {
                    k: next_obs[k][:, t]
                    for k in agent.cfg.input_shapes
                    if k.startswith("observation")
                }
                zt = agent.model.encode(obs_t, task) if zt_next is None else zt_next
                zt_buffer.append(zt)
                # Next z from model
                zt_next_pred = agent.model.next(zt, actions[:, t], task=task)
                # Next z from data
                zt_next = agent.model.encode(obs_t_next, task=task)
                # Consistency
                consistency_loss += jnp.mean((zt_next_pred - zt_next) ** 2)

                # Reward prediction
                reward_pred_logits = agent.reward_fn(zt, actions[:, t])  # shape [B, 5]
                true_rew = jax.nn.one_hot(
                    rewards[:, t].astype(jnp.int32), reward_pred_logits.shape[-1]
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
        # print("about to update model")
        new_model, info_dict = agent.model.apply_loss_fn(
            loss_fn=update_model, has_aux=True, pmap_axis=pmap_axis
        )
        # print("about to update critic")
        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=update_critic, has_aux=True, pmap_axis=pmap_axis
        )
        # print("info_dict keys", info_dict.keys())
        # print("critic_info keys", critic_info.keys())
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
    shape_meta: Dict = None,
    **kwargs,
) -> TDMPC2Agent:
    """Create a TDMPC2 agent with proper initialization."""

    # jax.tree_util.register_dataclass(TDMPC2Config)
    input_shapes = shape_meta["input_shapes"]

    # Create zero tensors for inputs
    batch_obs = {key: jnp.zeros(shape) for key, shape in input_shapes.items()}
    batch_obs.update(
        {
            k: jnp.zeros(shape_meta["output_shape"][k])
            for k in shape_meta["output_shape"]
        }
    )

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
    params = model_def.init(rngs, batch_obs)["params"]
    model = TrainState.create(
        model_def, params, tx=optax.adam(learning_rate=config.lr), key=sample_rng
    )

    # Determine observation types
    use_image = len(config.image_keys) > 0
    use_env_state = "observation.environment_state" in config.input_shapes
    input_image_key = None
    if use_image:
        image_keys = [
            k for k in config.input_shapes if k.startswith("observation.image")
        ]
        assert len(image_keys) == 1, "Expected exactly one image observation key"
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
        # eval_state=AgentEvalState.init_eval_state(config, use_image, use_env_state),
    )

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
    agent = create_tdmpc2_learner(FLAGS.config, rng, shape_meta=dataset.shape_meta)

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

            eval_info = evaluate(policy_fn, env, num_episodes=FLAGS.eval_episodes)
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
    flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")
    flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
    flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
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
