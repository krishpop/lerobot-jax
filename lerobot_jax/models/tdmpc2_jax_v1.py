import collections
import os
import pickle
from functools import partial
from typing import Dict, Optional, Tuple, Union, Any, Callable # Added Any, Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np # Added for AgentEvalState
# import omegaconf # Not used in this file
import optax
# import tqdm # Not used in this file
# import wandb # Not used in this file
# from absl import app, flags # Not used in this file
from flax import struct
from flax.core import FrozenDict, frozen_dict
from flax.training import checkpoints
from jaxrl_m.common import target_update # TrainState is defined in model_utils
from jaxrl_m.evaluation import evaluate
# Critic is defined here, but TDMPC2Critic is a specific subclass
from jaxrl_m.networks import Critic, ensemblize # type: ignore
from jaxrl_m.typing import Array, Batch, InfoDict, PRNGKey # Data was removed, using Dict[str, jnp.ndarray]
from jaxrl_m.vision.preprocess import PreprocessEncoder
# from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb # Not used
# from ml_collections import ConfigDict, config_flags # Not used

from lerobot_jax.utils.model_utils import (
    MLP,
    SimNorm,
    TDMPC2SimpleConv,
    TrainState, # Using this TrainState
    WithMappedEncoders,
)
from lerobot_jax.utils.norm_utils import normalize_inputs, unnormalize_outputs


# -------------
# Config class
# -------------


@jdc.pytree_dataclass
class TDMPC2Config:
    """
    Configuration for TDMPC2.
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
    discount_denom: float = 5.0
    tau: float = 0.01
    # Model architecture / latent dims
    latent_dim: int = 64
    action_dim: int = 6
    mlp_dim: int = 128
    num_channels: int = 32
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    image_keys: Tuple[str, ...] = ()
    output_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    dropout: float = 0.0
    simnorm_dim: int = 8
    # Loss coefficients
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    # actor
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1e-4 # Alpha for policy entropy
    # Critic
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    # For dynamic tasks
    mpc: bool = True
    episode_length: int = 200
    # Discount factor for weighting Q-values over the horizon in policy loss
    rho: float = 0.5
    # For large action dims
    action_mask: bool = False
    # Multitask specific
    multitask: bool = False
    tasks: Tuple[str, ...] = ()
    task_dim: int = 32
    action_dims: Tuple[int, ...] = ()

    # TD-M(PC)^2 specific parameters
    use_policy_constraint: bool = False
    prior_constraint_coef: float = 1.0  # Beta for TD-M(PC)^2
    adaptive_regularization: bool = False
    scale_threshold: float = 2.0

    # Eval action queue params (from original tdmpc2_jax_v1.py)
    n_action_steps: int = 5
    n_action_repeats: int = 1


    @property
    def bin_size(self) -> float:
        if self.num_bins <= 1:
            return 0.0
        return (self.vmax - self.vmin) / (self.num_bins - 1)

    @property
    def discount(self) -> float:
        if self.episode_length <= 0:
            effective_ep_len = 200
        else:
            effective_ep_len = self.episode_length
        base = (effective_ep_len - self.discount_denom) / effective_ep_len
        return jnp.clip(base, self.discount_min, self.discount_max)


# ----------------------------------------------------------------------------------------
# WorldModel, reward model, Q-function
# ----------------------------------------------------------------------------------------
class WorldModel(nn.Module):
    cfg: TDMPC2Config

    def setup(self):
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
                self._action_masks = masks # type: ignore
            else:
                self._action_masks = None
        else:
            self.task_emb = None # type: ignore
            self._action_masks = None

        img_encoder = None
        if len(self.cfg.image_keys) > 0:
            encoder_def = TDMPC2SimpleConv(
                num_channels=self.cfg.num_channels,
                apply_shift_aug=True,
                act=SimNorm(self.cfg.simnorm_dim),
            )
            img_encoder = PreprocessEncoder( # type: ignore
                encoder_def, normalize=True, resize=True, resize_shape=(224, 224)
            )

        if isinstance(self.cfg.input_shapes, dict) and len(self.cfg.input_shapes) > 1:
            encoder_defs = {}
            if self.cfg.input_shapes is not None:
                for key in self.cfg.input_shapes:
                    if "state" in key: # More robust check for state keys
                        encoder_defs[key] = MLP(
                            (128, 128),
                            output_dim=self.cfg.latent_dim,
                            use_sim_norm=True,
                            sim_norm_dim=self.cfg.simnorm_dim,
                        )
                    elif img_encoder is not None and key in self.cfg.image_keys:
                        encoder_defs[key] = img_encoder
            self.encoder = WithMappedEncoders( # type: ignore
                encoders=frozen_dict.freeze(encoder_defs),
                network=nn.Sequential([nn.Dense(self.cfg.latent_dim), nn.relu]), # type: ignore
                concatenate_keys=tuple(self.cfg.input_shapes.keys()) if self.cfg.input_shapes else tuple(),
            )
        else: # Single input type
            if len(self.cfg.image_keys) > 0 and img_encoder is not None:
                self.encoder = img_encoder
            elif self.cfg.input_shapes and "state" in list(self.cfg.input_shapes.keys())[0]:
                 self.encoder = MLP( # type: ignore
                    (128,128), output_dim=self.cfg.latent_dim, use_sim_norm=True, sim_norm_dim=self.cfg.simnorm_dim
                )
            else: # Fallback for single, non-image, non-"state" key, or if input_shapes is None
                self.encoder = MLP((128,128), output_dim=self.cfg.latent_dim) # type: ignore


        self.dynamics = MLP((128, 128), output_dim=self.cfg.latent_dim) # type: ignore
        self.reward = MLP((128, 128), output_dim=self.cfg.num_bins) # type: ignore
        self.pi_base = MLP((128, 128), output_dim=2 * self.cfg.action_dim) # type: ignore

    def __call__(
        self,
        batch_obs: Dict[str, jnp.ndarray],
        actions: jnp.ndarray,
        task: Optional[Union[int, jnp.ndarray]] = None,
        rng_key: Optional[PRNGKey] = None,
        training: bool = False
    ):
        z_enc = self.encode(batch_obs, task, training=training)
        z_combined = jnp.concatenate([z_enc, actions], axis=-1)
        next_z = self.dynamics(z_combined, training=training)
        rewards_logits = self.reward_fn(next_z, actions, training=training)

        if rng_key is None:
            rng_key = self.make_rng("sample") if self.is_mutable_collection("rngs") else jax.random.PRNGKey(0) # type: ignore

        sampled_actions, _ = self.pi(next_z, rng_key, task, training=training)
        return next_z, rewards_logits, sampled_actions

    def get_task_embedding_for_input(self, x_dict: Dict[str,jnp.ndarray], task_arr: jnp.ndarray) -> Dict[str,jnp.ndarray]:
        """ Helper to apply task embedding to a dictionary of input tensors. """
        if not self.cfg.multitask or self.task_emb is None:
            return x_dict

        emb = self.task_emb(task_arr) # task_arr is (B,)
        
        x_conditioned = {}
        for key, value in x_dict.items(): # value is (B, ...)
            current_emb = emb
            if emb.shape[0] == 1 and value.shape[0] > 1:
                current_emb = jnp.repeat(emb, value.shape[0], axis=0) # (B, task_dim)

            # Make current_emb (B, 1, ..., 1, task_dim) to match value's rank for concat
            while current_emb.ndim < value.ndim :
                current_emb = jnp.expand_dims(current_emb, axis=-2) # Add dims before last
            
            # Tile current_emb to match spatial/temporal dims of value, if any
            # Example: value (B, T, H, W, C_obs), current_emb (B, 1, 1, 1, C_task)
            tile_dims = [1] * current_emb.ndim
            for i in range(1, value.ndim -1): # Iterate over non-batch, non-channel dims
                if current_emb.shape[i] == 1 and value.shape[i] > 1:
                    tile_dims[i] = value.shape[i]
            current_emb = jnp.tile(current_emb, tile_dims)
            x_conditioned[key] = jnp.concatenate([value, current_emb], axis=-1)
        return x_conditioned

    def get_task_embedding_for_latent(self, z_latent: jnp.ndarray, task_arr: jnp.ndarray) -> jnp.ndarray:
        """ Helper to apply task embedding to a flat latent tensor. """
        if not self.cfg.multitask or self.task_emb is None:
            return z_latent
        
        emb = self.task_emb(task_arr) # task_arr is (B,)
        current_emb = emb
        if emb.shape[0] == 1 and z_latent.shape[0] > 1:
             current_emb = jnp.repeat(emb, z_latent.shape[0], axis=0) # (B, task_dim)
        return jnp.concatenate([z_latent, current_emb], axis=-1)


    def encode(
        self, obs: Dict[str, jnp.ndarray], task: Optional[Union[int, jnp.ndarray]] = None, training: bool = False
    ) -> jnp.ndarray:
        obs_input = obs
        if self.cfg.multitask and task is not None and self.task_emb is not None:
            task_arr = jnp.array([task]) if isinstance(task, int) else task
            obs_input = self.get_task_embedding_for_input(obs, task_arr)
        return self.encoder(obs_input, training=training)

    def next(
        self, z: jnp.ndarray, action: jnp.ndarray, task: Optional[Union[int, jnp.ndarray]] = None, training: bool = False
    ) -> jnp.ndarray:
        z_input = z
        if self.cfg.multitask and task is not None and self.task_emb is not None:
            task_arr = jnp.array([task]) if isinstance(task, int) else task
            z_input = self.get_task_embedding_for_latent(z, task_arr)
        z_action = jnp.concatenate([z_input, action], axis=-1)
        return self.dynamics(z_action, training=training)

    def pi(
        self, z: jnp.ndarray, key: PRNGKey, task: Optional[Union[int, jnp.ndarray]] = None,
        eval_mode: bool = False, training: bool = False
    ):
        z_input = z
        if self.cfg.multitask and task is not None and self.task_emb is not None:
            task_arr = jnp.array([task]) if isinstance(task, int) else task
            z_input = self.get_task_embedding_for_latent(z, task_arr)

        base_out = self.pi_base(z_input, training=training)
        mean, log_std = jnp.split(base_out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.cfg.log_std_min, self.cfg.log_std_max)

        if eval_mode:
            action_raw = mean
            noise = jnp.zeros_like(mean)
        else:
            noise = jax.random.normal(key, mean.shape)
            action_raw = mean + jnp.exp(log_std) * noise

        if self.cfg.multitask and task is not None and self._action_masks is not None:
            task_idx = task[0] if isinstance(task, jnp.ndarray) else task # Assuming task is scalar index or (1,)
            action_mask_val = self._action_masks[task_idx]
            if mean.ndim > action_mask_val.ndim:
                action_mask_val = jnp.expand_dims(action_mask_val, axis=0)
            mean = mean * action_mask_val
            action_raw = action_raw * action_mask_val
            log_std = log_std * action_mask_val

        log_prob_raw_gaussian = gaussian_logprob(noise, log_std)
        mean_processed, action_processed, log_prob_processed = squash(mean, action_raw, log_prob_raw_gaussian)

        info = {
            "mean": mean_processed, "log_std": log_std, "raw_action": action_raw,
            "final_action": action_processed, "log_prob_pi": log_prob_processed,
            "entropy": -log_prob_processed.squeeze(-1) if log_prob_processed.ndim > 1 and log_prob_processed.shape[-1] == 1 else -log_prob_processed
        }
        return action_processed, info

    def reward_fn(self, z: jnp.ndarray, action: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Task conditioning could be added here if reward depends on task and z directly
        return self.reward(jnp.concatenate([z, action], axis=-1), training=training)


class TDMPC2Critic(Critic): # type: ignore
    cfg: TDMPC2Config
    @nn.compact
    def __call__(self, z: jnp.ndarray, action: jnp.ndarray, training: bool = True ) -> jnp.ndarray:
        inputs = jnp.concatenate([z, action], -1)
        q = MLP( # type: ignore
            (self.cfg.mlp_dim, self.cfg.mlp_dim), output_dim=self.cfg.num_bins,
            activations=jax.nn.mish, use_normed_linear=True, use_sim_norm=False,
            dropout_rate=self.cfg.dropout,
        )(inputs, training=training)
        return q

@jdc.pytree_dataclass
class AgentEvalState:
    observation_queue: collections.deque
    action_queue: collections.deque
    @classmethod
    def create(cls, obs_maxlen: int=1, act_maxlen: int=5):
        return cls( observation_queue=collections.deque(maxlen=obs_maxlen),
                    action_queue=collections.deque(maxlen=act_maxlen))

# ----------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------
def mse_loss(val: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(val - target))

def soft_ce( logits: jnp.ndarray, labels: jnp.ndarray, num_bins: int,
             vmin: float, vmax: float, bin_size: float) -> jnp.ndarray:
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    target_two_hot = two_hot(labels, num_bins, vmin, vmax, bin_size)
    return -jnp.sum(target_two_hot * log_prob, axis=-1, keepdims=True)

def symlog(x: jnp.ndarray) -> jnp.ndarray: return jnp.sign(x) * jnp.log(1 + jnp.abs(x))
def symexp(x: jnp.ndarray) -> jnp.ndarray: return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

def squash(mu_raw: jnp.ndarray, pi_raw: jnp.ndarray, log_prob_pi_raw: jnp.ndarray):
    mu_squashed = jnp.tanh(mu_raw)
    pi_squashed = jnp.tanh(pi_raw)
    squash_correction = jnp.sum(jnp.log(jnp.maximum(1 - pi_squashed**2, 1e-6)), axis=-1, keepdims=True)
    log_prob_pi_squashed = log_prob_pi_raw - squash_correction
    return mu_squashed, pi_squashed, log_prob_pi_squashed

def gaussian_logprob(noise: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    residual = -0.5 * jnp.square(noise) * jnp.exp(-2 * log_std) - log_std
    log_prob = residual - 0.5 * jnp.log(2 * jnp.pi)
    return jnp.sum(log_prob, axis=-1, keepdims=True)

def two_hot(x_input: jnp.ndarray, num_bins: int, vmin: float, vmax: float, bin_size: float) -> jnp.ndarray:
    if num_bins == 0: return x_input
    if num_bins == 1: return symlog(x_input)
    original_shape = x_input.shape
    x_flat = symlog(x_input.reshape(-1).astype(jnp.float32))
    x_clipped = jnp.clip(x_flat, vmin, vmax)
    if abs(bin_size) < 1e-6: # Avoid division by zero
        bin_idx_int = jnp.round((x_clipped - vmin) / (vmax - vmin + 1e-6) * (num_bins -1)).astype(jnp.int32)
        bin_idx_int = jnp.clip(bin_idx_int, 0, num_bins -1)
        soft_two_hot_flat = jax.nn.one_hot(bin_idx_int, num_bins, dtype=jnp.float32)
    else:
        bin_idx_float = (x_clipped - vmin) / bin_size
        bin_idx_int = jnp.floor(bin_idx_float).astype(jnp.int32)
        bin_offset_float = bin_idx_float - bin_idx_int.astype(jnp.float32)
        current_bin_indices = jnp.clip(bin_idx_int, 0, num_bins - 1)
        next_bin_indices = jnp.clip(bin_idx_int + 1, 0, num_bins - 1)
        soft_two_hot_flat = jnp.zeros((x_flat.shape[0], num_bins), dtype=jnp.float32)
        flat_arange = jnp.arange(x_flat.shape[0])
        soft_two_hot_flat = soft_two_hot_flat.at[flat_arange, current_bin_indices].set(1.0 - bin_offset_float)
        soft_two_hot_flat = soft_two_hot_flat.at[flat_arange, next_bin_indices].add(bin_offset_float)
        soft_two_hot_flat = soft_two_hot_flat / (jnp.sum(soft_two_hot_flat, axis=-1, keepdims=True) + 1e-8)
    return soft_two_hot_flat.reshape(*original_shape, num_bins)

def two_hot_inv(logits: jnp.ndarray, cfg: TDMPC2Config) -> jnp.ndarray:
    if cfg.num_bins == 0: return logits
    if cfg.num_bins == 1: return symexp(logits)
    dreg_bins = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_bins)
    probs = jax.nn.softmax(logits, axis=-1)
    x = jnp.sum(dreg_bins * probs, axis=-1, keepdims=True)
    return symexp(x)

# ----------------------------------------------------------------------------------------
# TDMPC2 Agent
# ----------------------------------------------------------------------------------------
@struct.dataclass
class TDMPC2Agent:
    model: TrainState
    critic: TrainState
    target_critic: TrainState
    cfg: TDMPC2Config
    rng: PRNGKey
    moving_q_percentile: Optional[Array] = None
    normalization_stats: FrozenDict[str, Dict[str, Array]] = struct.field(default_factory=lambda: frozen_dict.freeze({}))
    normalization_modes: FrozenDict[str, str] = struct.field(default_factory=lambda: frozen_dict.freeze({}))
    _use_image: bool = struct.field(pytree_node=False, default=False)
    _use_env_state: bool = struct.field(pytree_node=False, default=False)
    input_image_key: Optional[str] = struct.field(pytree_node=False, default=None)
    eval_state: Optional[AgentEvalState] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create( cls, config: TDMPC2Config, rng: PRNGKey, shape_meta: Dict,
                normalization_stats: Dict, normalization_modes: Dict ) -> "TDMPC2Agent":
        rng, model_key, critic_key, target_critic_key, pi_key = jax.random.split(rng, 5)
        input_shapes_cfg = config.input_shapes or shape_meta["input_shapes"]
        action_dim_cfg = config.action_dim or shape_meta["output_shape"]["action"][-1]

        dummy_obs = {k: jnp.zeros((config.batch_size, *s)) for k,s in input_shapes_cfg.items()} # type: ignore
        dummy_actions = jnp.zeros((config.batch_size, action_dim_cfg ))
        model_def = WorldModel(config)
        model_params = model_def.init(model_key, dummy_obs, dummy_actions, rng_key=pi_key, training=False)['params'] # Add training=False
        model_train_state = TrainState.create( apply_fn=model_def.apply, params=model_params,
                                            tx=optax.adam(learning_rate=config.lr), key=pi_key)
        dummy_latent_z = jnp.zeros((config.batch_size, config.latent_dim))
        critic_def = ensemblize(TDMPC2Critic, num_qs=config.num_q, out_axes=0)(config) # type: ignore
        critic_params = critic_def.init(critic_key, dummy_latent_z, dummy_actions, training=False)['params'] # Add training=False
        critic_train_state = TrainState.create(apply_fn=critic_def.apply, params=critic_params,
                                               tx=optax.adam(learning_rate=config.lr), key=jax.random.split(critic_key,1)[0])
        target_critic_train_state = TrainState.create(apply_fn=critic_def.apply, params=critic_params,
                                                      tx=optax.adam(learning_rate=config.lr), key=jax.random.split(target_critic_key,1)[0])
        use_image = len(config.image_keys) > 0
        use_env_state = "observation.environment_state" in (config.input_shapes if config.input_shapes else {})
        input_image_key = config.image_keys[0] if use_image else None
        q_percentile_init = jnp.array(0.0) if config.adaptive_regularization else None

        return cls(model=model_train_state, critic=critic_train_state, target_critic=target_critic_train_state,
                   cfg=config, rng=rng, moving_q_percentile=q_percentile_init,
                   normalization_stats=frozen_dict.freeze(normalization_stats),
                   normalization_modes=frozen_dict.freeze(normalization_modes),
                   _use_image=use_image, _use_env_state=use_env_state, input_image_key=input_image_key,
                   eval_state=AgentEvalState.create(act_maxlen=max(config.n_action_steps, config.n_action_repeats)))

    def Qs( self, z: jnp.ndarray, action: jnp.ndarray, task: Optional[Union[int, jnp.ndarray]] = None,
            target: bool = False, training: bool = True, return_type: str = "min") -> jnp.ndarray:
        critic_state = self.target_critic if target else self.critic
        all_qs_logits = critic_state.apply_fn( {'params': critic_state.params}, z, action, training=training,
                                             rngs={'dropout': critic_state.key})
        if return_type == "all": return all_qs_logits
        q_values_scalar = jax.vmap(two_hot_inv, in_axes=(0, None), out_axes=0)(all_qs_logits, self.cfg)
        if return_type == "min":
            # It's generally better to use a fixed key or split from a passed-in key for dropout consistency across ensemble members if desired
            # Using self.rng here makes Qs stateful if not careful with JIT.
            # For TD3 min, it's usually over two randomly chosen Qs from the ensemble.
            # If num_q is small (e.g. 2), just take min over all. If larger, sample.
            if self.cfg.num_q > 2:
                 rng_for_q_sample, _ = jax.random.split(self.rng) # Placeholder for better rng handling
                 idx = jax.random.permutation(rng_for_q_sample, self.cfg.num_q)[:2]
                 q_values_subset = q_values_scalar[idx, ...]
                 return_val = jnp.min(q_values_subset, axis=0)
            else: # If num_q is 1 or 2, min over all is fine.
                 return_val = jnp.min(q_values_scalar, axis=0)

        elif return_type == "avg": return_val = jnp.mean(q_values_scalar, axis=0)
        else: raise ValueError(f"Unsupported Qs return_type: {return_type}")
        return return_val

    def train_step(self, batch: Batch, pmap_axis: Optional[str] = None) -> Tuple["TDMPC2Agent", InfoDict]:
        rng, key_model_dropout, key_model_sample, key_critic_dropout = jax.random.split(self.rng, 4)

        batch_normalized = normalize_inputs(batch, self.normalization_stats, self.normalization_modes)
        
        # Ensure input_shapes is not None before trying to iterate over it
        obs_keys = list(self.cfg.input_shapes.keys()) if self.cfg.input_shapes else []

        obs_norm = {k: batch_normalized[k][:, :-1] for k in obs_keys if k.startswith("observation") and k in batch_normalized}
        next_obs_norm = {k: batch_normalized[k][:, 1:] for k in obs_keys if k.startswith("observation") and k in batch_normalized}
        actions_norm = batch_normalized["action"]
        rewards_continuous = batch["next.reward"]
        task = batch.get("task")

        behavior_mean_norm = batch_normalized.get("behavior_mean")
        behavior_log_std_norm = batch_normalized.get("behavior_log_std")


        def world_model_policy_loss_fn(model_params):
            zt_buffer_list = []
            consistency_loss_total = 0.0
            reward_loss_total = 0.0
            
            # Prepare initial obs for encoding: select the first time step for all keys
            initial_obs_t0_norm = {k: v[:, 0] for k, v in obs_norm.items()}
            current_z = self.model.apply_fn( {'params': model_params}, initial_obs_t0_norm, task, # task for t=0
                                            method=WorldModel.encode, training=True, rngs={'dropout': key_model_dropout})
            zt_buffer_list.append(current_z)

            T = actions_norm.shape[1]
            for t in range(T):
                action_t_buffer = actions_norm[:, t]
                task_t = task[:,t] if task is not None and task.ndim == 2 and task.shape[1] == T else task

                z_next_pred = self.model.apply_fn( {'params': model_params}, current_z, action_t_buffer, task_t,
                                                 method=WorldModel.next, training=True, rngs={'dropout': key_model_dropout})
                next_obs_t_norm = {k: v[:, t] for k, v in next_obs_norm.items()}
                z_next_true = self.model.apply_fn( {'params': model_params}, next_obs_t_norm, task_t, # task for t+1
                                                 method=WorldModel.encode, training=True, rngs={'dropout': key_model_dropout})
                z_next_true_no_grad = jax.lax.stop_gradient(z_next_true)
                consistency_loss_total += mse_loss(z_next_pred, z_next_true_no_grad)

                reward_logits_pred = self.model.apply_fn( {'params': model_params}, current_z, action_t_buffer,
                                                        method=WorldModel.reward_fn, training=True, rngs={'dropout': key_model_dropout})
                reward_loss_total += soft_ce( reward_logits_pred, rewards_continuous[:, t], self.cfg.num_bins,
                                             self.cfg.vmin, self.cfg.vmax, self.cfg.bin_size).mean()
                current_z = z_next_true
                zt_buffer_list.append(current_z)

            consistency_loss_avg = consistency_loss_total / T
            reward_loss_avg = reward_loss_total / T
            
            zt_for_policy = jnp.stack(zt_buffer_list[:-1], axis=1)
            B, T_policy, _ = zt_for_policy.shape
            zt_for_policy_flat = zt_for_policy.reshape(B * T_policy, -1)
            
            task_for_policy_flat = task
            if task is not None:
                if task.ndim == 1 and T_policy > 1: task_for_policy_flat = jnp.repeat(task, T_policy, axis=0)
                elif task.ndim == 2 and task.shape[1] == T : task_for_policy_flat = task[:, :T_policy].reshape(B*T_policy, -1)


            key_pi_sample, _ = jax.random.split(key_model_sample) # Different key for policy sampling per update
            sampled_actions_pi_flat, pi_info_flat = self.model.apply_fn(
                {'params': model_params}, zt_for_policy_flat, key_pi_sample, task_for_policy_flat,
                method=WorldModel.pi, training=True, eval_mode=False, rngs={'dropout': key_model_dropout} # Pass sample key to pi
            )
            log_prob_pi_flat = pi_info_flat["log_prob_pi"].squeeze(-1) # (B*T,)

            q_values_for_pi_flat = self.Qs( zt_for_policy_flat, sampled_actions_pi_flat, task_for_policy_flat,
                                           return_type="avg", training=True, target=False).squeeze(-1) # (B*T,)

            current_beta_policy = self.cfg.prior_constraint_coef
            if self.cfg.use_policy_constraint and self.cfg.adaptive_regularization:
                if self.moving_q_percentile is not None and self.moving_q_percentile < self.cfg.scale_threshold:
                    current_beta_policy = 0.0
            
            if self.cfg.use_policy_constraint:
                if behavior_mean_norm is None or behavior_log_std_norm is None:
                    raise ValueError("Behavior policy parameters (mean, log_std) must be in batch for policy constraint.")
                
                behavior_mean_flat = behavior_mean_norm[:, :T_policy].reshape(B*T_policy, -1)
                behavior_log_std_flat = behavior_log_std_norm[:, :T_policy].reshape(B*T_policy, -1)
                
                # Denormalize sampled_actions_pi_flat before atanh IF actions were normalized for policy output
                # Assuming policy outputs actions in [-1,1] and they are NOT further normalized by policy itself.
                # The `squash` function already produces actions in [-1,1] range via tanh.
                
                # Calculate log_prob_mu(sampled_actions_pi_flat | zt_for_policy_flat)
                # sampled_actions_pi_flat are already tanh-squashed by WorldModel.pi
                sampled_actions_pre_squash = jnp.arctanh(jnp.clip(sampled_actions_pi_flat, -1 + 1e-6, 1 - 1e-6))
                noise_for_mu = (sampled_actions_pre_squash - behavior_mean_flat) / (jnp.exp(behavior_log_std_flat) + 1e-6)
                log_prob_mu_raw = gaussian_logprob(noise_for_mu, behavior_log_std_flat) # (B*T, 1)
                squash_correction_mu = jnp.sum(jnp.log(jnp.maximum(1 - sampled_actions_pi_flat**2, 1e-6)), axis=-1, keepdims=True)
                log_prob_mu_flat = (log_prob_mu_raw - squash_correction_mu).squeeze(-1) # (B*T,)

                pi_loss_terms = - (q_values_for_pi_flat - self.cfg.entropy_coef * log_prob_pi_flat + current_beta_policy * log_prob_mu_flat)
            else:
                pi_loss_terms = - (q_values_for_pi_flat - self.cfg.entropy_coef * log_prob_pi_flat)
            
            pi_loss = jnp.mean(pi_loss_terms)
            total_loss = (self.cfg.consistency_coef * consistency_loss_avg + self.cfg.reward_coef * reward_loss_avg + pi_loss)
            
            metrics = {"consistency_loss": consistency_loss_avg, "reward_loss": reward_loss_avg, "pi_loss": pi_loss,
                       "total_model_policy_loss": total_loss, "q_for_pi": q_values_for_pi_flat.mean(),
                       "policy_log_prob_pi": log_prob_pi_flat.mean()}
            if self.cfg.use_policy_constraint:
                metrics["policy_log_prob_mu"] = log_prob_mu_flat.mean()
                metrics["current_beta_policy"] = jnp.array(current_beta_policy) # Ensure it's an array for metrics
            return total_loss, metrics

        def critic_loss_fn(critic_params):
            B, T_critic, _ = actions_norm.shape # Assuming actions_norm is (B,T,A)
            obs_norm_flat_critic = {k: v.reshape(B*T_critic,-1) for k,v in obs_norm.items() if k in obs_norm and obs_norm[k].shape[1] == T_critic} # type: ignore
            next_obs_norm_flat_critic = {k: v.reshape(B*T_critic,-1) for k,v in next_obs_norm.items() if k in next_obs_norm and next_obs_norm[k].shape[1] == T_critic} # type: ignore
            actions_norm_flat_critic = actions_norm.reshape(B*T_critic, -1)
            rewards_flat_critic = rewards_continuous.reshape(B*T_critic)
            
            task_critic_flat = task
            if task is not None:
                if task.ndim == 1 and T_critic > 1 : task_critic_flat = jnp.repeat(task, T_critic, axis=0)
                elif task.ndim == 2 and task.shape[1] == T_critic : task_critic_flat = task.reshape(B*T_critic, -1) if task.shape[-1]>0 else task.reshape(B*T_critic)


            z_current_critic = self.model.apply_fn( {'params': self.model.params}, obs_norm_flat_critic, task_critic_flat,
                                                  method=WorldModel.encode, training=False, rngs={'dropout': key_model_dropout})
            q_logits_current_flat = self.critic.apply_fn( {'params': critic_params}, z_current_critic, actions_norm_flat_critic,
                                                       training=True, rngs={'dropout': key_critic_dropout})
            z_next_critic = self.model.apply_fn( {'params': self.model.params}, next_obs_norm_flat_critic, task_critic_flat,
                                               method=WorldModel.encode, training=False, rngs={'dropout': key_model_dropout})
            z_next_critic_no_grad = jax.lax.stop_gradient(z_next_critic)
            
            key_next_action_sample, _ = jax.random.split(key_model_sample)
            next_actions_pi_flat, _ = self.model.apply_fn( {'params': self.model.params}, z_next_critic_no_grad, key_next_action_sample, task_critic_flat,
                                                         method=WorldModel.pi, eval_mode=True, training=False, rngs={'dropout': key_model_dropout})
            q_logits_next_target_flat = self.target_critic.apply_fn( {'params': self.target_critic.params}, z_next_critic_no_grad,
                                                                   next_actions_pi_flat, training=True, rngs={'dropout': self.target_critic.key})
            q_values_next_target_scalar_flat = jax.vmap(two_hot_inv, in_axes=(0,None))(q_logits_next_target_flat, self.cfg).squeeze(-1)
            q_value_next_min_flat = jnp.min(q_values_next_target_scalar_flat, axis=0)
            td_targets_continuous_flat = rewards_flat_critic + self.cfg.discount * q_value_next_min_flat
            td_targets_continuous_no_grad = jax.lax.stop_gradient(td_targets_continuous_flat)
            
            loss_fn_single_q = lambda q_logits: soft_ce( q_logits, td_targets_continuous_no_grad, self.cfg.num_bins,
                                                       self.cfg.vmin, self.cfg.vmax, self.cfg.bin_size).mean()
            q_loss = jax.vmap(loss_fn_single_q)(q_logits_current_flat).sum()
            q_values_current_scalar_flat = jax.vmap(two_hot_inv, in_axes=(0,None))(q_logits_current_flat, self.cfg).squeeze(-1)
            metrics = {"q_loss": q_loss, "q_current_mean": q_values_current_scalar_flat.mean(),
                       "td_target_mean": td_targets_continuous_flat.mean()}
            return q_loss, metrics

        new_model_ts, model_policy_info = self.model.apply_gradients(loss_fn=world_model_policy_loss_fn, pmap_axis=pmap_axis, has_aux=True) # type: ignore
        new_critic_ts, critic_info = self.critic.apply_gradients(loss_fn=critic_loss_fn, pmap_axis=pmap_axis, has_aux=True) # type: ignore
        new_target_critic_ts = target_update(new_critic_ts, self.target_critic, self.cfg.tau) # type: ignore

        new_moving_q_percentile = self.moving_q_percentile
        if self.cfg.use_policy_constraint and self.cfg.adaptive_regularization:
            current_q_mean = critic_info["q_current_mean"]
            if new_moving_q_percentile is None: new_moving_q_percentile = current_q_mean
            else: new_moving_q_percentile = 0.99 * new_moving_q_percentile + 0.01 * current_q_mean
        
        metrics_all = {**model_policy_info, **critic_info}
        if new_moving_q_percentile is not None: metrics_all["moving_q_percentile"] = new_moving_q_percentile # type: ignore

        updated_agent = self.replace(model=new_model_ts, critic=new_critic_ts, target_critic=new_target_critic_ts, # type: ignore
                                     rng=rng, moving_q_percentile=new_moving_q_percentile)
        return updated_agent, metrics_all