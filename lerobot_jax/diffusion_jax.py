from .model_utils import MLP, WithEncoder, WithMappedEncoders
from .tdmpc2_jax import TDMPC2Config, two_hot_inv, WorldModel 

import jax
import einops
import functools
import jax.numpy as jnp
import numpy as np
import jax.lax as lax
import flax
import flax.linen as nn
from flax import struct
from flax.core import frozen_dict, FrozenDict
import optax

from jaxrl_m.typing import *
from jaxrl_m.common import TrainState
from jaxrl_m.vision import encoders
from jaxrl_m.vision.data_augmentations import random_crop
from jaxrl_m.vision.preprocess import PreprocessEncoder
from jaxrl_m.dataset import Dataset
from jaxrl_m.evaluation import supply_rng
from jaxrl_m.vision.pretrained_utils import load_pretrained_params
import ml_collections
from collections import deque

from diffusers import FlaxDDIMScheduler
from diffusers.schedulers.scheduling_ddim_flax import DDIMSchedulerState
from optax.schedules import Schedule


DEFAULT_HF_UNET_CONFIG = {
    # Size of the action sequence (horizon length)
    'sample_size': 32,  
    
    # Input/output channels
    'in_channels': 1,       # We treat actions as single-channel 2D data
    'out_channels': 7,      # Number of action dimensions
    
    # Architecture details
    'block_out_channels': (512, 1024, 2048),  # Channel dimensions for each U-Net level
    'layers_per_block': 2,                       # Number of residual blocks at each level
    'down_block_types': (                        # Types of down-sampling blocks
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    'up_block_types': (                         # Types of up-sampling blocks
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    
    # Attention parameters
    'cross_attention_dim': 1024,                # Dimension of conditional input (observation embedding)
    'attention_head_dim': 64,                   # Dimension of attention heads
    'num_attention_heads': 8,                   # Number of attention heads
    
    # Performance optimizations
    'use_memory_efficient_attention': True,     # Use memory efficient attention implementation
    'use_linear_projection': True,              # Use linear projections in attention
    
    # Other
    'dropout': 0.1,                            # Dropout rate
    'conv_out_kernel': 5,
    'norm_num_groups': 8,
    # 'addition_time_embed_dim': 128,
}


DEFAULT_UNET_CONFIG = {
    # Architecture details
    'model_channels': 64,    # Number of channels in the model
    'time_embed_dim': 128,   # Dimension for time embedding
    'down_dims': (256, 512, 1024),  # Channel dimensions for each U-Net level
    'kernel_size': 5,        # Kernel size for convolutions
    'n_groups': 8,           # Number of groups for normalization
    'use_film_scale_modulation': True,  # Use FiLM scale modulation
    'image_keys': (),
    'use_random_crop': False,
}


DEFAULT_OPTIM_CONFIG = {
    'learning_rate': 1e-4,
}


DEFAULT_SCHEDULER_CONFIG ={
    # Diffusion process length
    'num_train_timesteps': 100,   # Total number of diffusion steps
    
    # Noise schedule parameters
    # 'beta_schedule': "squaredcos_cap_v2",  # Type of beta schedule
    'beta_start': 0.00005,             # Starting noise level
    'beta_end': 0.02,                 # Ending noise level
    
    # DDIM specific settings
    'clip_sample': False,              # Whether to clip samples during inference
    'set_alpha_to_one': True,         # Set final alpha to 1 for better sample quality
    'steps_offset': 1,                # Offset for number of inference steps
    
    # Training settings
    # 'prediction_type': "epsilon",      # Whether to predict noise ("epsilon") or denoised sample ("sample")
}


def get_default_config(algo="diffusion") -> ml_collections.ConfigDict:
    """
    Extended config that includes both diffusion and world model parameters.
    """
    config = {
        'algo': algo,
        'optim_kwargs': DEFAULT_OPTIM_CONFIG,
        'net_kwargs': DEFAULT_UNET_CONFIG,
        'scheduler_kwargs': DEFAULT_SCHEDULER_CONFIG,
        'num_inference_steps': 100,
        'num_obs_steps': 2,
        'num_action_steps': 8,
        'horizon': 16,
    }
    if algo == "ric":
        config.update({
            "lr": 0.,
            'latent_dim': 64,
            'action_dim': 8,
            'num_q': 2,
            'consistency_coef': 1.0,
            'reward_coef': 1.0,
            'value_coef': 1.0,
            'guidance_scale': 1.0,
            "iql_ckpt_path": "",
            "use_iql": False
        })
    return ml_collections.ConfigDict(config)



class TimeEmbedding(nn.Module):
    """
    Encode diffusion timesteps into a vector (from modeling_diffusion.py style).
    """
    embed_dim: int

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        if len(timesteps.shape) < 1:
            timesteps = timesteps[None]
        half_dim = self.embed_dim // 2

        exponent = jnp.arange(half_dim, dtype=jnp.float32) / (float(half_dim) - 1)
        exponent = 1.0 / (10000.0 ** exponent)

        embeddings = timesteps[:, None] * exponent[None, :]

        
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        
        embeddings = nn.Dense(self.embed_dim * 4)(embeddings)
        
        embeddings = jax.nn.mish(embeddings)
        embeddings = nn.Dense(self.embed_dim)(embeddings)
        return embeddings


class DiffusionUNet(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""
    model_channels: int = 64
    time_embed_dim: int = 128
    image_keys: Tuple[str] = ()
    down_dims: Tuple[int, ...] = (512, 1024, 2048)
    output_dim: int = 8
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True
    use_random_crop: bool = False

    @nn.compact
    def __call__(
            self, 
            global_cond: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
            x: jnp.ndarray,
            timesteps: jnp.ndarray = None, 
            train: bool = False,
            rng: Optional[jnp.ndarray] = None,
            **kwargs
        ) -> jnp.ndarray:
        """
        global_cond: (B, T, D)
        x: (B, T, D)
        timesteps: (B,)
        """
        
        # Time embedding
        b, t, d = global_cond.shape
        timesteps = einops.rearrange(timesteps, "b -> b 1")
        emb = TimeEmbedding(embed_dim=self.time_embed_dim)(timesteps)
        emb = einops.repeat(emb, "b 1 d-> (b t) d", b=b, t=t)
        
        # Combine time embedding with global conditioning
        if global_cond is not None:
            global_cond = einops.rearrange(global_cond, "b t d -> (b t) d", b=b, t=t)
            emb = jnp.concatenate([emb, global_cond], axis=1)
            emb = einops.rearrange(emb, "(b t) d -> b t d", b=b, t=t)

        skip_connections = []
        emb = einops.rearrange(emb, "b t d -> b (t d)", b=b, t=t)
        # print(f"Flattened embedding shape: {emb.shape}")

        # Down blocks
        h = x
        conditional_resblock = functools.partial(
            ConditionalResBlock1D, 
            n_groups=self.n_groups, 
            kernel_size=self.kernel_size,
            use_film_scale_modulation=self.use_film_scale_modulation
        )

        for i, dim_out in enumerate(self.down_dims):
            h = conditional_resblock(dim_out)(h, emb)
            h = conditional_resblock(dim_out)(h, emb)
            skip_connections.append(h)
            # Downsample
            h = nn.Conv(
                features=dim_out,
                kernel_size=3,
                strides=2,
                padding=1
            )(h)

        # Middle blocks
        h = conditional_resblock(self.down_dims[-1])(h, emb)
        h = conditional_resblock(self.down_dims[-1])(h, emb)

        # Up blocks
        for i, (up_idx, dim_out) in enumerate(enumerate(reversed(self.down_dims[:-1]))):
            is_last = (up_idx == len(self.down_dims) - 2)
            
            h = jnp.concatenate([h, skip_connections.pop()], axis=1)
            
            h = conditional_resblock(dim_out)(h, emb)
            h = conditional_resblock(dim_out)(h, emb)
            
            if not is_last:
                # Downsample
                h = nn.ConvTranspose(
                    features=dim_out,
                    kernel_size=4,
                    strides=2,
                    padding=0
                )(h)

        # Final convolutions
        h = nn.GroupNorm(num_groups=self.n_groups)(h)
        
        h = nn.Conv(
            features=x.shape[-1],
            kernel_size=1,
            padding=0
        )(h)
        return h


class UNetConv1DBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    n_groups: int = 8
    out_channels: int = 64
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (B, in_channels, T)
        """
        # print("pre conv1d in ", self.name, "x.shape:", x.shape)
        x = nn.Conv(features=self.out_channels, kernel_size=self.kernel_size, padding='SAME')(x)
        # print(f"post conv1d in {self.name}, x.shape:", x.shape)
        x = nn.GroupNorm(num_groups=self.n_groups)(x)
        x = jax.nn.mish(x)
        return x


class ConditionalResBlock1D(nn.Module):
    """ResNet style 1D convolutional block with FiLM conditioning."""
    out_channels: int
    kernel_size: int = 3
    n_groups: int = 8
    use_film_scale_modulation: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        x: (B, n_act, Dout)
        cond: (B, Dcond)
        """
        # First conv block
        b, t, d = x.shape
        h = UNetConv1DBlock(
            n_groups=self.n_groups,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size)(x)  # (B, n_act, D_emb)
        h = einops.rearrange(h, "b t d -> b d t", b=b, t=t)

        # FiLM conditioning
        cond = jax.nn.mish(cond)
        if self.use_film_scale_modulation:
            cond = nn.Dense(features=self.out_channels * 2)(cond)
            scale, bias = jnp.split(cond, 2, axis=-1)
            h = h * scale[..., None] + bias[..., None]
        else:
            bias = nn.Dense(features=self.out_channels)(cond)
            h = h + bias[..., None]

        h = einops.rearrange(h, "b d t -> b t d", b=b, t=t)

        # Second conv block
        h = nn.GroupNorm(num_groups=self.n_groups)(h)
        h = nn.relu(h)
        h = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(h)

        # Residual connection with optional projection
        if x.shape[1] != self.out_channels:
            x = nn.Conv(
                features=self.out_channels,
                kernel_size=(1,),
                padding='SAME'
            )(x)

        return x + h


@struct.dataclass
class AgentEvalState:
    """State maintained during evaluation."""
    from collections import deque

    observation_queue: deque = struct.field(default_factory=lambda: deque(maxlen=2))
    action_queue: deque = struct.field(default_factory=lambda: deque(maxlen=8))


class SimpleDiffusionAgent(flax.struct.PyTreeNode):
    """
    Diffusion-based agent. Follows a structure similar to BCAgent in continuous_bc.py,
    but uses a diffusion/UNet-based approach for generating actions.
    """
    rng: PRNGKey
    model: TrainState
    scheduler_params: FrozenDict = None
    num_obs_steps: int = 2
    num_action_steps: int = 8
    horizon: int = 16
    shape_meta: FrozenDict = None
    num_inference_steps: int = 100
    eval_state: Optional[AgentEvalState] = None

    def add_noise(
            self,
            actions: jnp.ndarray,
            noise: jnp.ndarray,
            timesteps: jnp.ndarray,
        ) -> jnp.ndarray:
        """
        Adds noise to actions based on the scheduler's configuration and timesteps.

        Args:
            actions (jnp.ndarray): Original actions.
            noise (jnp.ndarray): Noise to be added.
            timesteps (jnp.ndarray): Timesteps corresponding to each action.
            scheduler_config (dict): Configuration parameters of the scheduler.

        Returns:
            jnp.ndarray: Noisy actions.
        """
        scheduler = FlaxDDIMScheduler(
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2", 
            **self.scheduler_params
        )
        scheduler_state = scheduler.create_state()
        return scheduler.add_noise(
            scheduler_state, 
            actions, 
            noise, 
            timesteps
        )

    def init_eval_state(self) -> 'SimpleDiffusionAgent':
        """Initialize evaluation state with empty queues."""
        return self.replace(
            eval_state=AgentEvalState(
                observation_queue=deque(maxlen=self.num_obs_steps), 
                action_queue=deque(maxlen=self.num_action_steps)
            )
        )

    def update(self, batch: Batch, target: Data, output_shape: Tuple[int, ...]):
        """A single gradient update step."""
        rng = jax.random.PRNGKey(self.model.step)
        # scheduler_state = scheduler.create_state()

        def loss_fn(params):
            noise = jax.random.normal(rng, output_shape)
            timesteps = jax.random.randint(rng, (output_shape[0],), 0, 100)
            noisy_actions = self.add_noise(target, noise, timesteps)
            # noisy_actions = agent.add_noise(scheduler, scheduler_state, target, noise, timesteps)
            pred = self.model(
                batch,
                x=noisy_actions,
                timesteps=timesteps,
                train=True,
                params=params
            )

            l2_loss = jnp.mean((noise - pred)**2)
            return l2_loss, {'diffusion_loss': l2_loss}

        new_model, info = self.model.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        return self.replace(model=new_model), info

    def update_eval_state(
        self,
        observations: Optional[Dict[str, np.ndarray]] = None,
        actions: Optional[np.ndarray] = None,
    ) -> 'SimpleDiffusionAgent':
        """Update observation and action queues in eval state."""
        if self.eval_state is None:
            self = self.init_eval_state()
        
        new_state = self.eval_state
        
        if observations is not None:
            queue = new_state.observation_queue
            queue.append(observations)
            
            # Pad queue with first observation if not full
            if len(queue) < self.num_obs_steps:
                while len(queue) < self.num_obs_steps:
                    queue.append(observations)
                
            new_state = self.eval_state.replace(observation_queue=queue)
            
        if actions is not None:
            queue = new_state.action_queue
            queue.extend(actions)
            
            # Pad queue with first action if not full
            if len(queue) < self.num_action_steps:
                first_action = queue[-1]
                while len(queue) < self.num_action_steps:
                    queue.extend(first_action)
                    
            new_state = self.eval_state.replace(action_queue=queue)
        return self.replace(eval_state=new_state)

    def sample_actions(
        self,
        observations: Dict[str, np.ndarray],
        *,
        output_shape: Tuple[int, ...],
        seed: PRNGKey = None
    ) -> jnp.ndarray:
        """Sample actions using the diffusion model."""
        rng = jax.random.PRNGKey(seed) if seed is not None else self.rng
        bsz = observations[list(self.shape_meta["input_shapes"].keys())[0]].shape[0]
        
        # Update observation queue in eval state
        self = self.update_eval_state(observations=observations)

        obs_batch = {
            key: jnp.stack([obs[key] for obs in self.eval_state.observation_queue], axis=1)
            for key in observations.keys()
        }
        
        # Initialize noise
        sample = jax.random.normal(
            rng,
            (bsz, *output_shape[1:])
        )
        
        # Setup scheduler
        scheduler = FlaxDDIMScheduler(
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
            **self.scheduler_params
        )
        scheduler_state = scheduler.create_state()
        scheduler_state = scheduler.set_timesteps(scheduler_state, self.num_inference_steps)

        # Denoise sample
        for t in scheduler_state.timesteps:
            timesteps = jnp.full((bsz,), t)
            pred = self.model(
                obs_batch,
                x=sample,
                timesteps=timesteps,
                train=False
            )
            sample = scheduler.step(
                scheduler_state,
                sample,
                t,
                pred
            ).prev_sample

        # Update action queue in eval state
        start, end = self.num_obs_steps - 1, self.num_obs_steps - 1 + self.num_action_steps
        self = self.update_eval_state(actions=sample.transpose(1, 0, 2)[start:end])
        action = self.eval_state.action_queue.popleft()
        return action


class RICDiffusionAgent(SimpleDiffusionAgent):
    """
    Extends ConditionalDiffusionAgent to incorporate model-based guidance from TDMPC2.
    """
    world_model: Any = None  # TDMPC2's world model
    guidance_scale: Union[float, Schedule] = 1.0  # How much to weight the model-based guidance
    
    def compute_loss_fn(
        self,
        actions: jnp.ndarray,
        observations: jnp.ndarray,
        env_states: jnp.ndarray,  # New parameter for environment states
        rng: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Combines diffusion loss with model-based guidance from TDMPC2.
        """
        # Original diffusion loss
        diffusion_loss = super().compute_loss_fn(actions, observations, env_states, rng)
        
        # Get world model predictions
        z = self.world_model.encode(observations)
        
        # Compute Q-values for the actions
        q_logits = self.world_model.Qs(z, actions)  # [num_q, batch, 5]
        # Convert to values using two_hot_inv (defined in TDMPC2)
        q_values = jax.vmap(two_hot_inv)(q_logits)  # [num_q, batch, 1]
        
        # Take minimum Q-value for conservative estimation
        min_q = jnp.min(q_values, axis=0)  # [batch, 1]
        
        # Guidance loss: encourage actions with higher Q-values
        guidance_loss = -jnp.mean(min_q)
        
        # Combine losses
        total_loss = diffusion_loss + self.guidance_scale * guidance_loss
        
        return total_loss

    def sample_actions(
        self, 
        observations, 
        rng: jnp.ndarray, 
        num_inference_steps: int = 50,
    ):
        """
        Sample actions using classifier-free guidance from the world model.
        """
        # Get initial noise shape
        shape = (observations.shape[0], 10, 6)  # Adjust as needed
        rng, rng_eps = jax.random.split(rng)
        sample = jax.random.normal(rng_eps, shape)
        
        # Process observations
        if self.obs_as_global_cond:
            obs_cond = jnp.mean(observations, axis=1) if observations.ndim == 3 else observations
        else:
            obs_cond = observations
            
        # Setup scheduler
        scheduler_state = self.scheduler.create_state()
        
        def guidance_fn(x_t, t):
            """Compute gradient from world model Q-values."""
            z = self.world_model.encode(observations)
            q_logits = self.world_model.Qs(z, x_t)
            q_values = jax.vmap(two_hot_inv)(q_logits)
            min_q = jnp.min(q_values, axis=0)
            # Gradient of Q with respect to actions
            grad_q = jax.grad(lambda x: jnp.mean(min_q))(x_t)
            return grad_q
            
        # Sampling loop with classifier guidance
        for step in range(num_inference_steps):
            t = scheduler_state.timesteps[step]
            
            # Get model prediction
            pred = self.model.apply_fn(
                {"params": self.model.params}, 
                sample, t, obs_cond, 
                train=False
            )
            
            # # Get guidance gradient
            # if self.guidance_scale > 0:
            #     guidance_grad = guidance_fn(sample, t)
            #     pred = pred + self.guidance_scale * guidance_grad
                
            # Scheduler step
            sample = self.scheduler.step(
                scheduler_state, pred, t, sample
            ).prev_sample
            
        return sample


def create_input_encoder(
    input_keys: Tuple[str, ...],
    image_encoder: str = 'resnetv1-18',
    feature_encoder_dims: Optional[Tuple[int]] = (64, 64),
    fused_feature_encoder_dims: Optional[Tuple[int]] = (64, 64),
) -> nn.Module:
    obs_encoder_defs = {}
    for key in input_keys:
        if 'image' in key:
            encoder = encoders[image_encoder]()
            obs_encoder_defs[key] = PreprocessEncoder(encoder=encoder, normalize=True, resize=True, resize_shape=(224, 224))
        else:
            obs_encoder_defs[key] = MLP(hidden_dims=feature_encoder_dims[:-1], output_dim=feature_encoder_dims[-1])
    
    # fused network for parameter sharing
    if fused_feature_encoder_dims is not None:
        fused_net_def = nn.Sequential([MLP(hidden_dims=fused_feature_encoder_dims[:-1], output_dim=fused_feature_encoder_dims[-1])])
    else:
        fused_net_def = None

    obs_encoder_def = WithMappedEncoders(
        encoders=obs_encoder_defs, 
        network=fused_net_def,
        concatenate_keys=input_keys
    )
    return obs_encoder_def


def create_simple_diffusion_learner(
        seed: int,
        shape_meta: Dict[str, Any],
        output_key: str,
        encoder_def: WithMappedEncoders = None,
        config: Dict[str, Any] = get_default_config(),
        load_pretrained_weights: bool = False,
        **kwargs):
    """
    Creates a SimpleDiffusionAgent instance that uses a diffusion-based UNet model.
    Follows a structure reminiscent of create_bc_learner, but replaces the policy with UNet.
    """
    if config is None:
        scheduler_kwargs = DEFAULT_SCHEDULER_CONFIG
        optim_kwargs = DEFAULT_OPTIM_CONFIG
        net_kwargs = DEFAULT_UNET_CONFIG
    else:
        scheduler_kwargs = config.scheduler_kwargs
        optim_kwargs = config.optim_kwargs
        net_kwargs = config.net_kwargs

    print('Extra kwargs:', kwargs)
    rng = jax.random.PRNGKey(seed)

    assert isinstance(encoder_def, WithMappedEncoders), "encoder_def must be a LowDimEncoder instance"
    # set up scheduler from diffusers
    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    # Build the UNet or encoder+UNet
    unet_def = DiffusionUNet(**net_kwargs)
    if encoder_def is not None:
        model_def = WithEncoder(encoder=encoder_def, network=unet_def)
    else:
        model_def = unet_def

    # Init the model
    bsz = shape_meta["batch_size"]
    input_shapes = shape_meta["input_shapes"]
    output_shape = shape_meta["output_shape"][output_key]

    # Create zero tensors for inputs
    batch_obs = {key: jnp.zeros(shape) for key, shape in input_shapes.items()}
    batch_obs[output_key] = jnp.zeros(output_shape)

    params_dict = model_def.init(
        rng, batch_obs,
        x=batch_obs[output_key],
        timesteps=jax.random.randint(
            rng, (bsz,), 0, scheduler_kwargs["num_train_timesteps"]), 
    )
    print("successfully initialized model")
    tx = optax.adam(**optim_kwargs)
    
    model = TrainState.create(
        model_def, 
        params=params_dict['params'],
        tx=tx
    )

    if load_pretrained_weights:
        load_pretrained_params(
            params_dict['params'],
            params_dict['extra_variables'],
            model.params,
            model.extra_variables,
            prefix_key=f'encoder/encoders/{encoder_def.name}'
    )

    return SimpleDiffusionAgent(
        rng,
        model=model,
        shape_meta=shape_meta,
        scheduler_params=frozen_dict.freeze(scheduler_kwargs),
        num_obs_steps=config.num_obs_steps,
        num_action_steps=config.num_action_steps,
        horizon=config.horizon,
        # output_key=output_key,
    )


def create_ric_guidance_schedule(
        initial_value: float,
        final_value: float = None,
        num_steps: int = None
    ) -> Schedule:

    if final_value is None:
        return optax.constant_schedule(initial_value)
    if num_steps is None:
        num_steps = 100

    return optax.linear_schedule(initial_value, final_value, num_steps)


def create_iql_learner(checkpoint_path, shape_meta, seed, max_steps, **kwargs):
    import os
    import sys

    sys.path.append(os.path.expanduser("~/implicit_q_learning"))

    from learner import Learner # type: ignore
    from train_offline import load_checkpoint as load_iql_checkpoint # type: ignore
    empty_obs = np.zeros(shape_meta["observation_shape"])[np.newaxis]
    empty_action = np.zeros(shape_meta["action_shape"])[np.newaxis]
    agent = Learner(seed,
                    empty_obs,
                    empty_action,
                    max_steps=max_steps,
                    **kwargs)

    load_iql_checkpoint(agent, checkpoint_path)
    return agent


def create_ric_diffusion_learner(
    seed: int,
    shape_meta: Dict[str, Any],
    output_key: str,
    encoder_def: WithMappedEncoders = None,
    config: ml_collections.ConfigDict = get_default_config(),
    load_pretrained_weights: bool = False,
    world_model_condition_on_state: bool = True,
    **kwargs
) -> RICDiffusionAgent:

    scheduler_kwargs = config.scheduler_kwargs
    optim_kwargs = config.optim_kwargs
    net_kwargs = config.net_kwargs

    bsz = config.batch_size
    input_shapes = shape_meta["input_shapes"]
    output_shape = shape_meta["output_shape"][output_key]
    batch_obs = {key: jnp.zeros((bsz,) + shape) for key, shape in input_shapes.items()}
    batch_obs[output_key] = jnp.zeros((bsz,) + output_shape)

    key = jax.random.PRNGKey(seed)
    key, init_rng, critic_rng = jax.random.split(key, 3)
    if config.use_iql:
        shape_meta["observation_shape"] = np.sum(shape_meta["input_shapes"].values(), axis=1)
        shape_meta["action_shape"] = shape_meta["output_shape"][output_key]
        iql_critic = create_iql_learner(config, shape_meta)
    else:
        world_model_config = TDMPC2Config(**config.world_model_config)
        guidance_scale = create_ric_guidance_schedule(**config.guidance_scale_kwargs)
        world_model_def = WorldModel(world_model_config)
        world_model_params = world_model_def.init(
            init_rng,
            x=batch_obs[output_key],
            cond=batch_obs,
            task=jnp.zeros((batch_obs.shape[0],), dtype=jnp.int32)
        )

        if config.world_model_optim_kwargs is not None:
            world_model_optim_kwargs = config.world_model_optim_kwargs
            world_model_tx = optax.adam(**world_model_optim_kwargs)
        else:
            world_model_tx = None
        
        world_model = TrainState.create(
            world_model_def,
            params=world_model_params['params'],
            tx=world_model_tx
        )

    model_def = DiffusionUNet(**net_kwargs)
    if encoder_def is not None:
        model_def = WithEncoder(encoder=encoder_def, network=model_def)

    model_params = model_def.init(
        init_rng,
        batch_obs,
        x=batch_obs[output_key],
        timesteps=jax.random.randint(
            init_rng, (batch_obs.shape[0],), 0, scheduler_kwargs["num_train_timesteps"]), 
    )

    if world_model_condition_on_state:
        batch_obs = {k: batch_obs[k] for k in batch_obs if 'image' not in k} 

    tx = optax.adam(**optim_kwargs)
    model = TrainState.create(
        model_def,
        params=model_params['params'],
        tx=tx
    )
    
    # Combine into guided agent
    return RICDiffusionAgent(
        model=model,
        world_model=world_model,
        guidance_scale=guidance_scale
    )
