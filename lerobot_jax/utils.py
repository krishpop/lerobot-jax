import jax
import os
import numpy as np
import jax.numpy as jnp
import lerobot
from functools import partial
from jax import lax, pmap


LEROBOT_ROOT = lerobot.__path__[0]


# This function computes the local statistics on each device.
def local_stats(x):
    count = x.shape[0]
    total = jnp.sum(x, axis=0)
    min_val = jnp.min(x, axis=0)
    max_val = jnp.max(x, axis=0)
    total_sq = jnp.sum(x * x, axis=0)
    return count, total, min_val, max_val, total_sq

def shard_data(data, n_devices):
    B, D = data.shape[0], data.shape[1]
    B_adjusted = (B // n_devices) * n_devices
    data = data[:B_adjusted, :]
    data_sharded = data.reshape(n_devices, -1, D)
    return data_sharded

# Use pmap with an axis name so we can do cross-device reductions.
@partial(pmap, axis_name='devices')
def compute_global_stats(x):
    # Compute local stats on this device.
    count, total, min_val, max_val, total_sq = local_stats(x)
    
    # Use collective operations to reduce across devices.
    global_count = lax.psum(count, axis_name='devices')
    global_total = lax.psum(total, axis_name='devices')
    global_total_sq = lax.psum(total_sq, axis_name='devices')
    # For min and max, we use pmin and pmax.
    global_min = lax.pmin(min_val, axis_name='devices')
    global_max = lax.pmax(max_val, axis_name='devices')
    
    return global_count, global_total, global_min, global_max, global_total_sq

def compute_normalization_stats(dataset, keys, n_devices, cache_filepath="normalization_stats.npy"):
    if cache_filepath is not None:
        if os.path.exists(cache_filepath):
            return np.load(cache_filepath, allow_pickle=True)

    stats = {}
    for key in keys:
        data = dataset[key]
        data_sharded = shard_data(data, n_devices)
        global_count, global_total, global_min, global_max, global_total_sq = compute_global_stats(data_sharded)
        stats[key] = {
            'global_count': global_count,
            'global_total': global_total,
            'global_min': global_min,
            'global_max': global_max,
            'global_total_sq': global_total_sq
        }

    if cache_filepath is not None:
        np.save(cache_filepath, stats)
    return global_count, global_total, global_min, global_max, global_total_sq
    