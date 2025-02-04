import flax.linen as nn
import jax.numpy as jnp
from jaxrl_m.typing import *
from jaxrl_m.networks import get_latent

def get_latent(
    encoder: nn.Module, observations: Union[jnp.ndarray, Dict[str, jnp.ndarray]]
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


class MLP(nn.Module):
    hidden_dims: Tuple[int, ...] = (128, 128)
    output_dim: int = 64
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = self.act_fn(x)
        x = nn.Dense(self.output_dim)(x)
        return x


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
        latents = {key: get_latent(self.encoders[key], observations[key]) for key in self.concatenate_keys}
        
        # Concatenate latents in the specified order
        concatenated_latents = jnp.concatenate([latents[key] for key in self.concatenate_keys], axis=-1)
        
        # If a network is defined, pass the concatenated latents through it
        if self.network is not None:
            return self.network(concatenated_latents, *args, **kwargs)
        
        # Otherwise, return the concatenated latents
        return concatenated_latents

 