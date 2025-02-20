import collections
import os
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import optax
import tqdm
import wandb
from absl import app, flags
from flax import struct
from flax.core import frozen_dict, FrozenDict
from flax.training import checkpoints
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.evaluation import evaluate
from jaxrl_m.networks import ensemblize
from jaxrl_m.typing import *
from jaxrl_m.vision.preprocess import PreprocessEncoder
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from lerobot_jax.model_utils import TDMPC2SimpleConv, WithMappedEncoders
from ml_collections import ConfigDict, config_flags

from .model_utils import MLP

# -------------
# Config class 
# -------------

@dataclass
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
    num_samples: int = 64
    num_elites: int = 6
    iterations: int = 10
    num_pi_trajs: int = 4
    temperature: float = 0.5
    grad_clip_norm: float = 100.0
    batch_size: int = 256
    discount_min: float = 0.99
    discount_max: float = 0.99
    discount_denom: float = 200
    # Model architecture / latent dims
    latent_dim: int = 64
    action_dim: int = 6
    num_channels: int = 32
    input_shapes: Dict[str, str] = None
    output_shapes: Dict[str, str] = None
    obs: str = "state"  # Observation type (state or rgb)
    # Loss coefficients
    consistency_coef: float = 1.0
    reward_coef: float = 1.0
    value_coef: float = 1.0
    num_q: int = 2
    # For dynamic tasks
    mpc: bool = True
    episode_length: int = 200
    episode_lengths: Tuple[int, ...] = ()
    # Temperature for Gumbel-Softmax sampling, etc.
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
                    masks = masks.at[i, :self.cfg.action_dims[i]].set(1.0)
                self._action_masks = masks
            else:
                self._action_masks = None

        # Set up the encoder based on observation type.
        if self.cfg.obs == "rgb":
            encoder_def = TDMPC2SimpleConv(
                num_channels=self.cfg.num_channels, apply_shift_aug=True
            )
            img_encoder = PreprocessEncoder(
                encoder_def, normalize=True, resize=True, resize_shape=(224, 224)
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
                    encoder_defs[key] = MLP((128, 128), self.cfg.latent_dim)
                else:
                    encoder_defs[key] = img_encoder
            self.encoder = WithMappedEncoders(
                encoders=encoder_defs,
                network=nn.Sequential([nn.Dense(self.cfg.latent_dim), nn.relu]),
                concatenate_keys=list(self.cfg.input_shapes.keys()),
            )
        else:
            if self.cfg.obs == "rgb":
                self.encoder = img_encoder
            else:
                self.encoder = MLP((128, 128), self.cfg.latent_dim)

        # Instantiate the rest of the submodules.
        self.dynamics = MLP((128, 128), self.cfg.latent_dim)
        self.reward = MLP((128, 128), 5)  # 5 categories for discrete reward.
        self.Q_list = ensemblize(MLP, self.cfg.num_q)(hidden_dims=(128, 128), output_dim=5)
        self.pi_base = MLP((128, 128), 2 * self.cfg.action_dim)

    def __call__(
        self, batch_obs: Data, task: Optional[Union[int, jnp.ndarray]] = None,
    ):
        """
        Forward pass.

        Args:
            x: observations or a dict of observations.
            task: Task ID or array of task IDs (optional).

        Returns:
            next_z: Next latent state predicted by the dynamics network.
        """

        x = {k: batch_obs[k][:, :-1] for k in self.cfg.input_shapes if k.startswith('observation')}
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
        qs = self.Qs(z_enc, a)

        rewards = self.reward(jnp.concatenate([next_z, a], axis=-1))
        actions = self.pi(next_z, jax.random.PRNGKey(0))

        return next_z, qs

    def get_task_embedding(self, x: jnp.ndarray, task: Union[int, jnp.ndarray]) -> jnp.ndarray:
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

    def encode(self, obs: jnp.ndarray, task: Optional[Union[int, jnp.ndarray]] = None) -> jnp.ndarray:
        """
        Encode observations with optional task embedding.
        """
        if self.cfg.multitask and task is not None:
            if isinstance(obs, dict):
                obs = obs[self.cfg.obs]
            obs = self.get_task_embedding(obs, task)
            
        return self.encoder(obs)

    def next(self, z: jnp.ndarray, action: jnp.ndarray, task: Optional[Union[int, jnp.ndarray]] = None) -> jnp.ndarray:
        """
        Predict next latent state with optional task conditioning.
        """
        if self.cfg.multitask and task is not None:
            z = self.get_task_embedding(z, task)
        
        z_action = jnp.concatenate([z, action], axis=-1)
        return self.dynamics(z_action)

    def pi(self, z: jnp.ndarray, rng: jax.random.PRNGKey, task: Optional[Union[int, jnp.ndarray]] = None, eval_mode: bool = False):
        """
        Sample action from policy with optional task conditioning.
        """
        if self.cfg.multitask and task is not None:
            z = self.get_task_embedding(z, task)

        # Get mean and log_std
        base_out = self.pi_base(z)
        mean, log_std = jnp.split(base_out, 2, axis=-1)
        log_std = jnp.clip(log_std, -10.0, 2.0)

        if eval_mode:
            action = mean
        else:
            # Sample with reparameterization
            noise = jax.random.normal(rng, mean.shape)
            action = mean + jnp.exp(log_std) * noise

        # Apply action masks for multitask case
        if self.cfg.multitask and task is not None:
            action_mask = self._action_masks[task]
            mean = mean * action_mask
            log_std = log_std * action_mask
            action = action * action_mask

        return jnp.tanh(action), (mean, log_std)

    def Qs(self, z: jnp.ndarray, action: jnp.ndarray, task: Optional[Union[int, jnp.ndarray]] = None) -> jnp.ndarray:
        """
        Compute Q-values with optional task conditioning.
        """
        if self.cfg.multitask and task is not None:
            z = self.get_task_embedding(z, task)
            
        q_input = jnp.concatenate([z, action], axis=-1)
        all_qs = self.Q_list(q_input)
        return jnp.stack(all_qs, axis=0)  # shape [num_q, batch, 5]

    def reward_fn(self, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.reward(jnp.concatenate([z, action], axis=-1))


# ----------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------
def mse_loss(val: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((val - target) ** 2)


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    For the discrete classification approach. Expects 'labels' as a one-hot or integer class.
    """
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(labels * log_prob, axis=-1))


def two_hot_inv(logits: jnp.ndarray) -> jnp.ndarray:
    """
    Example of converting discrete reward distribution back to a real value
    (like TDMPC2 does with 'math.two_hot_inv').
    For simplicity, we define categories = [-2, -1, 0, 1, 2].
    """
    categories = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    probs = nn.softmax(logits, axis=-1)
    return jnp.sum(categories * probs, axis=-1, keepdims=True)


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



@dataclass
class AgentEvalState:
    observation_queue: collections.deque = struct.field(default_factory=lambda: collections.deque(maxlen=1))
    action_queue: collections.deque = struct.field(default_factory=lambda: collections.deque(maxlen=5))


# ----------------------------------------------------------------------------------------
# TDMPC2 main logic
# ----------------------------------------------------------------------------------------
class TDMPC2Agent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    model: TrainState
    cfg: TDMPC2Config
    normalization_stats: FrozenDict[str, Dict[str, jnp.ndarray]] = None
    normalization_modes: FrozenDict[str, str] = None
    eval_state: Optional[AgentEvalState] = None
    _use_image: bool = False
    _use_env_state: bool = False
    input_image_key: Optional[str] = None

    def init_eval_state(self) -> 'TDMPC2Agent':
        """Initialize evaluation state with empty queues."""
        queues = {
            "observation.state": collections.deque(maxlen=1),
            "action": collections.deque(maxlen=max(
                self.cfg.n_action_steps, self.cfg.n_action_repeats
            )),
        }
        if self._use_image:
            queues["observation.image"] = collections.deque(maxlen=1)
        if self._use_env_state:
            queues["observation.environment_state"] = collections.deque(maxlen=1)
            
        return self.replace(
            eval_state=AgentEvalState(
                observation_queue=queues["observation.state"],
                action_queue=queues["action"]
            )
        )

    def update_eval_state(
        self,
        observations: Optional[Dict[str, jnp.ndarray]] = None,
        actions: Optional[jnp.ndarray] = None,
    ) -> 'TDMPC2Agent':
        """Update observation and action queues in eval state."""
        if self.eval_state is None:
            self = self.init_eval_state()
        
        new_state = self.eval_state
        
        if observations is not None:
            queue = new_state.observation_queue
            queue.append(observations)
            new_state = self.eval_state.replace(observation_queue=queue)
            
        if actions is not None:
            queue = new_state.action_queue
            if isinstance(actions, list):
                queue.extend(actions)
            else:
                queue.append(actions)
            new_state = self.eval_state.replace(action_queue=queue)
            
        return self.replace(eval_state=new_state)

    def normalize_inputs(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Normalize inputs based on dataset statistics.
        
        Args:
            batch: Dictionary of input arrays to normalize.
            
        Returns:
            Normalized batch dictionary.
        """
        batch = dict(batch)  # Shallow copy to avoid mutating input
        
        # Normalize each key based on its mode
        for key in batch:
            if key not in self.cfg.normalization_stats:
                continue
            
            stats = self.cfg.normalization_stats[key]
            mode = self.cfg.normalization_modes[key]
            
            if mode == "mean_std":
                mean = stats["mean"]
                std = stats["std"]
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif mode == "min_max":
                min_val = stats["min"] 
                max_val = stats["max"]
                # First normalize to [0,1]
                batch[key] = (batch[key] - min_val) / (max_val - min_val + 1e-8)
                # Then to [-1, 1]
                batch[key] = batch[key] * 2.0 - 1.0
            
        return batch

    def unnormalize_outputs(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Unnormalize outputs back to original scale.
        
        Args:
            batch: Dictionary of output arrays to unnormalize.
            
        Returns:
            Unnormalized batch dictionary.
        """
        batch = dict(batch)  # Shallow copy to avoid mutating input
        
        # Unnormalize each key based on its mode
        for key in batch:
            if key not in self.cfg.normalization_stats:
                continue
            
            stats = self.cfg.normalization_stats[key]
            mode = self.cfg.normalization_modes[key]
            
            if mode == "mean_std":
                mean = stats["mean"]
                std = stats["std"]
                batch[key] = batch[key] * std + mean
            elif mode == "min_max":
                min_val = stats["min"]
                max_val = stats["max"]
                # First from [-1, 1] to [0, 1]
                batch[key] = (batch[key] + 1.0) / 2.0
                # Then to original range
                batch[key] = batch[key] * (max_val - min_val) + min_val
            
        return batch

    def populate_queues(self, queues: Dict[str, collections.deque], 
                       batch: Dict[str, jnp.ndarray]) -> Dict[str, collections.deque]:
        """Populate the queues with new observations."""
        new_queues = dict(queues)
        for key in batch:
            if key in new_queues:
                new_queues[key].append(batch[key])
        return new_queues

    @partial(jax.jit, static_argnums=(0,))
    def sample_actions(
        agent, 
        observations: Dict[str, jnp.ndarray], 
        *, 
        seed: PRNGKey = None
    ) -> jnp.ndarray:
        """Select a single action given environment observations.
        
        Args:
            observations: Dictionary containing observations.
            seed: Optional random seed for reproducibility.
        Returns:
            Selected action as a JAX array.
        """
        rng = jax.random.PRNGKey(seed) if seed is not None else agent.rng
        bsz = observations[list(agent.cfg.input_shapes.keys())[0]].shape[0]
        
        # Normalize inputs
        batch = agent.normalize_inputs(observations)
        if agent._use_image:
            batch = dict(batch)  # shallow copy
            batch["observation.image"] = batch[agent.input_image_key]

        # Update queues with new observations
        agent = agent.update_eval_state(observations=batch)

        # When action queue is depleted, populate it by querying policy
        if len(agent.eval_state.action_queue) == 0:
            # Stack queue contents
            batch = {
                key: jnp.stack(list(agent.eval_state.observation_queue[key]), axis=1) 
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
            z = agent.model.encode({k: batch[k] for k in encode_keys})

            # Get actions either through planning or direct policy
            if agent.cfg.mpc:
                actions = agent.plan(z, rng)  # (horizon, batch, action_dim)
            else:
                # Use policy directly - returns one action
                actions = agent.model.pi(z, rng)[0].unsqueeze(0)

            # Clip actions to [-1, 1]
            actions = jnp.clip(actions, -1.0, 1.0)

            # Unnormalize actions
            actions = agent.unnormalize_outputs({"action": actions})["action"]

            # Handle action repeats
            if agent.cfg.n_action_repeats > 1:
                agent = agent.update_eval_state(actions=[actions[0]] * agent.cfg.n_action_repeats)

            else:
                # Extend action queue with planned actions
                for act in actions[:agent.cfg.n_action_steps]:
                    agent.eval_state.action_queue.append(act)

        # Return next action from queue
        return agent.eval_state.action_queue.popleft()

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.model(obs, method=WorldModel.encode)

    def next(self, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.model(jnp.concatenate([z, action], axis=-1), method=WorldModel.next)

    def reward_fn(self, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.model.reward_fn(z, action)

    def pi(self, z: jnp.ndarray, rng: jax.random.PRNGKey, eval_mode: bool = False):
        return self.model(z, rng, eval_mode, method=WorldModel.pi)

    def Qs(self, z: jnp.ndarray, action: jnp.ndarray):
        """
        Return shape [num_q, batch, 5].
        """
        return self.model(jnp.concatenate([z, action], axis=-1), method=WorldModel.Qs)

    def plan(self, obs: jnp.ndarray, rng: jax.random.PRNGKey, eval_mode: bool = False):
        """
        Example of MPPI or CEM-based plan in JAX. 
        For demonstration, we do a trivial approach here that picks best from random sampling.
        """
        # Encode obs
        z = self.encode(obs)
        horizon = self.cfg.horizon
        samples = self.cfg.num_samples
        # We'll do random sampling
        rngs = jax.random.split(rng, horizon)
        shape = (horizon, samples, self.cfg.action_dim)
        rand_actions = [jax.random.uniform(r, shape=(samples, self.cfg.action_dim),
                                           minval=-1, maxval=1) for r in rngs]
        rand_actions = jnp.stack(rand_actions, axis=0)  # [horizon, samples, act_dim]

        # Evaluate
        # We'll do a simple step-by-step rollout in latent space
        def rollout_body(carry, t):
            z_t = carry
            # we have the index t, we want to retrieve the actions for time t
            act_t = rand_actions[t]
            z_next = self.next(
                jnp.expand_dims(z_t, 0).repeat(samples, axis=0), act_t
            )
            # compute rewards
            rew_logits = self.reward_fn(
                jnp.expand_dims(z_t, 0).repeat(samples, axis=0), act_t
            )
            rew = two_hot_inv(rew_logits).squeeze(-1)
            return (z_next), rew

        def scan_trajectory(z0):
            # z0 shape [latent_dim], we broadcast to [samples, latent_dim]
            z_samp = jnp.expand_dims(z0, 0).repeat(samples, axis=0)
            rews = []
            for tstep in range(horizon):
                rew_logits = self.reward_fn(z_samp, rand_actions[tstep])
                rew = two_hot_inv(rew_logits)
                rews.append(rew)
                z_samp = self.next(z_samp, rand_actions[tstep])
            # approximate value by final Q
            # We'll take standard approach: pick Q_min from our two Qs
            # Just for demonstration, we do Q-avg
            final_q_logits = self.Qs(z_samp, jnp.zeros_like(rand_actions[0]))  # zero act
            # shape [num_q, batch, 5]
            final_q_values = jnp.mean(final_q_logits, axis=0)
            final_val = two_hot_inv(final_q_values)
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

    def act(self, obs: jnp.ndarray, rng: jax.random.PRNGKey, eval_mode: bool = False) -> jnp.ndarray:
        """
        Decide on an action for the environment.
        """
        if self.cfg.mpc:
            return self.plan(obs, rng, eval_mode)
        z = self.encode(obs)
        action, _info = self.pi(z, rng, eval_mode)
        return action[0]

    def update(agent, batch: Batch, target: jnp.ndarray=None) -> InfoDict:  # noqa: F405
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
        obs = {k: batch[k][:, :-1] for k in agent.cfg.input_shapes if k.startswith('observation')}
        next_obs = {k: batch[k][:, 1:] 
                    for k in agent.cfg.input_shapes if k.startswith('observation')}
        actions = batch['action']
        rewards = batch['next.reward']

        task = batch.get('task', None)  # optional

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
            rng, rng_sample = jax.random.split(agent.rng, 2)
            next_act, _ = agent.model.pi(next_z, rng_sample, task=task, eval_mode=True)
            # Q of (s', a')
            q_next_logits = agent.model.Qs(next_z, next_act, task=task)  # shape [num_q, B, 5]
            q_next_mean = jnp.mean(q_next_logits, axis=0)                # shape [B, 5]
            v_next = two_hot_inv(q_next_mean).squeeze(-1)               # shape [B]
            # Compute two-hot inverse for the reward prediction if stored as discrete
            rew_val = rew
            # discount can be adjusted by agent.cfg if needed, here we keep it constant
            discount = agent.config['discount']
            return rew_val + discount * v_next

        # We need a separate function to do the policy update
        def update_pi(zs, task):
            """
            Update policy using the latent states. Typically, TDMPC2 might do a
            model-based approach or a direct actor update. This version, for demonstration,
            uses discrete Q distribution methods from TDMPC2 logic.
            """
            # Sample actions from current policy
            rng_pi, new_rng = jax.random.split(agent.rng, 2)
            agent_rng_repl = agent.replace(rng=new_rng)
            sampled_actions, _info = agent_rng_repl.model.pi(zs, rng_pi, task=task)
            # Evaluate Q
            q_logits = agent_rng_repl.model.Qs(zs, sampled_actions, task=task)  # [num_q, B, 5]
            q_avg_logits = jnp.mean(q_logits, axis=0)                           # [B, 5]
            # Convert discrete Q to a real value
            q_vals = two_hot_inv(q_avg_logits)
            # Negative of Q-values is the basic policy loss
            pi_loss = -jnp.mean(q_vals)
            return pi_loss

        # ----------------------------------------------------
        # 3) Perform a multi-step latent rollout for consistency and gather states.
        #    We do "unroll" each step: z_{t+1} = model.next(z_t, action_t).
        # ----------------------------------------------------
        def rollout_loss(params):
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
            td_targets_all = []
            # We'll collect Q logits as well for each step
            q_logits_all = []
            reward_loss = 0.0

            # For each step t in [0..T-1], do a forward rollout
            zt_next = None
            for t in range(T - 1):
                # Current z 
                zt = agent.model.encode({k: obs[k][:, t] for k in agent.cfg.input_shapes if k.startswith('observation')}, task) if zt_next is None else zt_next
                # Next z from model
                zt_next_pred = agent.model.next(zt, actions[:, t], task=task)
                # Next z from data
                zt_next = agent.model.encode({k: next_obs[k][:, t] for k in agent.cfg.input_shapes if k.startswith('observation')}, task=task)
                # Consistency
                consistency_loss += jnp.mean((zt_next_pred - zt_next) ** 2)

                # Reward prediction
                reward_pred_logits = agent.reward_fn(zt, actions[:, t])  # shape [B, 5]
                true_rew = jax.nn.one_hot(rewards[:,t].astype(jnp.int32), reward_pred_logits.shape[-1])
                rew_loss = cross_entropy_loss(reward_pred_logits, true_rew)
                reward_loss += rew_loss

                # Q distribution for Q(s,a)
                qs_logits = agent.model.Qs(zt, actions[:, t], task=task)  # shape [num_q, B, 5]
                q_logits_all.append(qs_logits)

                # Next z for TD 
                # We'll do a 1-step TD target
                td_val = td_target(zt_next, rewards[:,t])
                td_targets_all.append(td_val)

            # Average consistency and reward loss
            # to mirror the TDMPC2 idea of weighting by horizon
            horizon_factor = float(T - 1) if T > 1 else 1.0
            consistency_loss = consistency_loss / horizon_factor
            reward_loss = reward_loss / horizon_factor

            # Q loss from cross-entropy with TD target
            # Convert TD targets to discrete bins [-2, -1, 0, 1, 2]
            # Then CE loss comparing Q(s,a) distribution with that discrete bin
            q_loss_sum = 0.0
            cat_vals = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
            for qs_logits, td_val in zip(q_logits_all, td_targets_all):
                # clamp the target to [-2,2]
                td_clamped = jnp.clip(td_val, -2.0, 2.0)
                # find closest index
                def find_nearest_idx(val):
                    abs_diff = jnp.abs(cat_vals - val)
                    return jnp.argmin(abs_diff)

                target_idx = jax.vmap(find_nearest_idx)(td_clamped)
                target_dist = jax.nn.one_hot(target_idx, 5)
                # sum across all Q heads
                per_step_loss = 0.0
                for i in range(agent.config['num_q']):
                    per_step_loss += cross_entropy_loss(qs_logits[i], target_dist)
                per_step_loss = per_step_loss / float(agent.config['num_q'])
                q_loss_sum += per_step_loss

            q_loss = q_loss_sum / horizon_factor

            # Now, for a policy update (like in TDMPC2) we sample a sequence of latent states
            # again. For simplicity, just use the first T states:
            # We combine them along batch dimension. Then compute the policy loss.
            # In TDMPC2, we often do a separate loop or call update_pi. 
            pi_loss = 0.0
            for t in range(T):
                zt_for_pi = agent.model.encode(obs[t], task=task)
                pi_loss += update_pi(zt_for_pi, task)
            pi_loss = pi_loss / float(T)

            # Weighted sum, or just add them as recommended by TDMPC2
            total_loss_with_pi = total_loss + pi_loss

            # Return losses to keep track
            losses_dict = {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "q_loss": q_loss,
                "pi_loss": pi_loss,
                "total_loss": total_loss_with_pi,
            }
            return total_loss_with_pi, losses_dict

        new_model, info_dict = agent.model.apply_loss_fn(loss_fn=rollout_loss, has_aux=True)
        loss_val = info_dict.pop('loss_val', None)

        # Update critic parameters
        new_model = new_model.apply_gradients(grads=grads)

        # Soft update target critic
        new_target_critic = agent.soft_update(agent.target_critic, new_critic, agent.config['tau'])

        # Update agent's rng so it changes after each update.
        rng_updated = jax.random.split(agent.rng, 2)[0]

        # Prepare final info dict with numeric indicators
        info = {}
        for k, v in info_dict.items():
            info[k] = v

        # Return updated agent and the info
        updated_agent = agent.replace(
            rng=rng_updated,
            model=new_model,
        )
        return updated_agent, info


def create_q_ensemble(rng, cfg, example_q_input):
    """
    Build an ensemble of Q networks externally.
    
    Args:
        rng: JAX PRNGKey.
        cfg: A config object containing at least cfg.num_q and cfg.critic_lr.
        example_q_input: An example input (e.g. concatenated [z, a]) used for parameter initialization.
    
    Returns:
        A TrainState containing the Q ensemble.
    """
    # Define the ensemble using ensemblize along with any additional settings.
    q_def = ensemblize(MLP, cfg.num_q, out_axes=0)
    # Instantiate the ensemble by passing the hidden dims.
    q_net = q_def(hidden_dims=(128, 128, 5))
    return q_net


def create_tdmpc2_learner(
    config: TDMPC2Config,
    rng: jax.random.PRNGKey,
    normalization_stats: Dict = None,
    normalization_modes: Dict = None,
    shape_meta: Dict = None,
    **kwargs
) -> TDMPC2Agent:
    """Create a TDMPC2 agent with proper initialization."""
    
    normalization_stats = frozen_dict.freeze(normalization_stats)
    normalization_modes = frozen_dict.freeze(normalization_modes)
    input_shapes = shape_meta["input_shapes"]

    # Create zero tensors for inputs
    batch_obs = {key: jnp.zeros(shape) for key, shape in input_shapes.items()}
    batch_obs.update({k: jnp.zeros(shape_meta["output_shape"][k]) for k in shape_meta["output_shape"]})

    # Initialize model
    # critic_def = create_q_ensemble(rng, config, batch_obs)
    model_def = WorldModel(config)
    params = model_def.init(rng, batch_obs)['params']
    model = TrainState.create(model_def, params, tx=optax.adam(learning_rate=config.lr))

    # Determine observation types
    use_image = config.obs == "rgb"
    use_env_state = "observation.environment_state" in config.input_shapes
    input_image_key = None
    if use_image:
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        assert len(image_keys) == 1, "Expected exactly one image observation key"
        input_image_key = image_keys[0]

    # Create agent
    agent = TDMPC2Agent(
        rng=rng,
        model=model,
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
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, 
                                    wandb.config.exp_prefix, wandb.config.experiment_id)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f'Saving config to {FLAGS.save_dir}/config.pkl')
        with open(os.path.join(FLAGS.save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(get_flag_dict(), f)

    # Load environment config
    dataset_cfg = omegaconf.DictConfig({
        "dataset_repo_id": "lerobot/d3il_sorting_example",
        "env": {"name": "sorting"},
        "training": {"image_transforms": {"enable": False}},
        "dataset_root": None,
        "device": "cpu"  # Keep dataset on CPU for JAX transfer
    })

    # Create environment and dataset
    env = make_env(dataset_cfg)
    dataset = make_dataset(dataset_cfg, split="train")
    print(f"Loaded dataset of length {len(dataset)} on CPU")

    # Create numpy dataloader
    from jax.tree_util import tree_map
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True
    )
    train_iter = iter(train_loader)

    # Initialize agent
    rng = jax.random.PRNGKey(FLAGS.seed)
    agent = create_tdmpc2_learner(FLAGS.config, rng, dataset)

    # Training loop
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        
        # Get next batch (with restart if needed)
        try:
            batch = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Convert relevant parts of batch to jnp arrays
        obs = jnp.array(batch['observation.state'])
        next_obs = jnp.array(batch['next.observation.state'])
        actions = jnp.array(batch['action'])
        rewards = jnp.array(batch['next.reward'])

        # Update step
        metrics = agent.train_step(
            obs_batch=obs,
            action_batch=actions,
            reward_batch=rewards,
            next_obs_batch=next_obs,
            rng=rng
        )

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in metrics.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            # Create a policy function that uses agent.plan
            def policy_fn(obs, rng):
                return agent.plan(obs, rng, eval_mode=True)
            
            eval_info = evaluate(policy_fn, env, 
                               num_episodes=FLAGS.eval_episodes)
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(
                FLAGS.save_dir, 
                agent._state.params,  # Save just the parameters
                step=i
            )


if __name__ == '__main__':
    # Define flags similar to run_d4rl_iql.py
    FLAGS = flags.FLAGS
    flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
    flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
    flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
    flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
    flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
    flags.DEFINE_integer('save_interval', 25000, 'Save interval.')
    flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
    flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
    flags.DEFINE_string('env_config', '../lerobot/lerobot/configs/env/d3il_sorting.yaml',
                        'Path to the environment config.')

    # Add wandb config
    wandb_config = default_wandb_config()
    wandb_config.update({
        'project': 'lerobot_tdmpc2',
        'group': 'tdmpc2_test',
        'name': 'tdmpc2_{env_config}'
    })
    config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)

    # Convert TDMPC2Config to a dictionary and then to a ConfigDict
    config_dict = ConfigDict(vars(TDMPC2Config()))
    config_flags.DEFINE_config_dict('config', config_dict, lock_config=False)
    app.run(main)