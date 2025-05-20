import jax
import jax.numpy as jnp


def mse_loss(val: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Calculates the mean squared error loss."""
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

    x = jnp.clip(symlog(x), vmin, vmax)
    bin_idx = jnp.floor((x - vmin) / bin_size)
    bin_offset = (x - vmin) / bin_size - bin_idx

    soft_two_hot = jnp.zeros((x.shape[0], num_bins))
    bin_idx = bin_idx.astype(jnp.int32)
    soft_two_hat = soft_two_hot.at[:, bin_idx].set(1 - bin_offset)
    soft_two_hat = soft_two_hot.at[:, (bin_idx + 1) % num_bins].set(bin_offset)
    return soft_two_hat


def two_hot_inv(logits: jnp.ndarray, cfg) -> jnp.ndarray:
    """
    Convert discrete reward distribution back to a real value.
    """
    if cfg.num_bins == 0:
        return logits
    elif cfg.num_bins == 1:
        return symexp(logits)

    dreg_bins = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_bins)
    probs = jax.nn.softmax(logits, axis=-1)
    x = jnp.sum(dreg_bins * probs, axis=-1, keepdims=True)
    return symexp(x)
