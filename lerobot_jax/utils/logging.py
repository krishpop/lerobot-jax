import jax.numpy as jnp


def collect_metrics(metrics, names, prefix=None):
    """
    Collect metrics from a dictionary based on a list of names.
    Optionally add a prefix to the metric names.
    """
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            "{}/{}".format(prefix, key): value for key, value in collected.items()
        }
    return collected 