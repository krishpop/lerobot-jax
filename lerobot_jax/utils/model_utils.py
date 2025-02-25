import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Module
from jaxrl_m.networks import get_latent, default_init
from jaxrl_m.common import TrainState
from jaxrl_m.typing import *  # noqa: F403
from jaxrl_m.vision.data_augmentations import random_crop


class TrainState(TrainState):
    key: Optional[jax.random.PRNGKey] = None  # noqa: F405


def get_latent(  # noqa: F811
    encoder: nn.Module,
    observations: Union[jnp.ndarray, Dict[str, jnp.ndarray]],  # noqa: F405
):
    """

    Get latent representation from encoder. If observations is a dict
        a state and image component, then concatenate the latents.

    """
    if encoder is None:
        return observations

    elif isinstance(observations, dict) and not isinstance(encoder, WithMappedEncoders):
        return jnp.concatenate(
            [encoder(observations["image"]), observations["state"]], axis=-1
        )

    else:
        return encoder(observations)


class WithEncoder(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class NormedLinear(nn.Dense):
    dropout_rate: float = 0.0
    use_norm: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = super().__call__(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        if self.use_norm:
            x = nn.LayerNorm()(x)
        if self.activations:
            x = self.activations(x)
        return x


class MLP(nn.Module):
    hidden_dims: Tuple[int, ...] = (128, 128)
    output_dim: int = 64
    activations: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.mish
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()
    use_normed_linear: bool = False  # Flag to toggle NormedLinear
    dropout_rate: float = 0.0  # Dropout rate (only used if NormedLinear is enabled)
    use_sim_norm: bool = False
    sim_norm_dim: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        for i, dim in enumerate(self.hidden_dims):
            x = NormedLinear(
                dim,
                use_norm=self.use_normed_linear,
                dropout_rate=self.dropout_rate * (i == 0),
                activations=self.activations,
            )(x, training=training)

        x = nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)  # Final layer
        if self.use_sim_norm:
            x = SimNorm(self.sim_norm_dim)(x)
        elif self.activate_final:
            x = self.activations(x)
        return x


class SimNorm(nn.Module):
    """
    Simplicial normalization in Flax.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    dim: int  # Dimension for reshaping before softmax.

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Store the original shape
        shp = x.shape

        # Reshape: Keep batch dimensions and reshape the last axis
        x = x.reshape(*shp[:-1], -1, self.dim)

        # Apply softmax over the last axis
        x = nn.softmax(x, axis=-1)

        # Reshape back to original shape
        return x.reshape(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class ConcatenateWithEncoders(nn.Module):
    encoders: Tuple[nn.Module, ...]  # Tuple of encoder modules
    network: nn.Module

    @nn.compact
    def __call__(self, observations: Data, *args, **kwargs):
        # Concatenate latent embeddings from all encoders
        latents = [get_latent(encoder, observations) for encoder in self.encoders]
        concatenated_latents = jnp.concatenate(latents, axis=-1)
        return self.network(concatenated_latents, *args, **kwargs)


class WithMappedEncoders(nn.Module):
    encoders: Dict[str, nn.Module]
    concatenate_keys: Tuple[str] = None
    network: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations: Data, *args, **kwargs):
        # Compute latents for each encoder
        latents = {
            key: get_latent(self.encoders[key], observations[key])
            for key in self.concatenate_keys
        }

        # Concatenate latents in the specified order
        concatenated_latents = jnp.concatenate(
            [latents[key] for key in self.concatenate_keys], axis=-1
        )

        # If a network is defined, pass the concatenated latents through it
        if self.network is not None:
            return self.network(concatenated_latents, *args, **kwargs)

        # Otherwise, return the concatenated latents
        return concatenated_latents


class ShiftAug(Module):
    """
    Random shift image augmentation in JAX, similar to the PyTorch version.
    We pad the image on each side and then randomly choose a patch corresponding
    to the original image size, effectively shifting the image in both directions.
    """

    pad: int = 3

    @staticmethod
    def pad_and_shift(x: jnp.ndarray, rng: jax.random.PRNGKey, pad: int) -> jnp.ndarray:
        # x is expected to be of shape (N, H, W, C)
        # pad x
        x_padded = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
        n, h_original, w_original, c = x.shape

        # Sample random offsets
        max_offset = 2 * pad + 1
        dx = jax.random.randint(rng, shape=(n,), minval=0, maxval=max_offset)
        dy = jax.random.randint(rng, shape=(n,), minval=0, maxval=max_offset)

        # For each image in the batch, slice a subwindow of size (h_original, w_original)
        def shift_single_image(args):
            i, (image, ox, oy) = args
            return image[ox : ox + h_original, oy : oy + w_original, :]

        # vmap across the batch dimension
        shifted = jax.vmap(shift_single_image, in_axes=(0, (0, 0, 0)), out_axes=0)(
            jnp.arange(n), (x_padded, dx, dy)
        )
        return shifted

    def __call__(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        return self.pad_and_shift(x, rng, self.pad)


class TDMPC2SimpleConv(nn.Module):
    """
    A simple convolutional encoder in JAX/Flax, inspired by the TDMPC2 architecture.
    Now optionally applies random shift augmentation, normalizes input images to [-0.5, 0.5],
    then applies 4 convolution layers with ReLU, and finally flattens the output. If
    an additional activation is supplied (act), it is applied at the end.
    """

    num_channels: int
    apply_shift_aug: bool = True
    act: Optional[Callable] = None

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        # If shift augmentation is requested and rng is provided
        if self.apply_shift_aug and rng is not None:
            x = random_crop(x, rng, padding=3)

        x -= 0.5  # Normalize to [-0.5, 0.5] from [0, 1]

        # Conv layers
        x = nn.Conv(features=self.num_channels, kernel_size=(7, 7), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=(1, 1))(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Optional final activation
        if self.act is not None:
            x = self.act(x)

        return x


tdmpc2_simple_conv_configs = {"tdmpc2_simple_conv": TDMPC2SimpleConv}
