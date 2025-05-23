import functools
import os
import pickle
from dataclasses import asdict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import torch
import tqdm
import wandb
from absl import app, flags
from diffusers import FlaxDDIMScheduler
from flax.core import frozen_dict
from flax.training import checkpoints
from jax.tree_util import tree_map
from jaxrl_m.evaluation import evaluate
from jaxrl_m.typing import *
from jaxrl_m.wandb import default_wandb_config, get_flag_dict, setup_wandb
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.utils.utils import init_hydra_config
from ml_collections import ConfigDict, config_flags
from omegaconf import OmegaConf
from scalax.sharding import FSDPShardingRule, MeshShardingHelper, PartitionSpec
from torch.utils.data import DataLoader

from lerobot_jax.agents.diffusion_jax import (
    create_input_encoder,
    create_ric_diffusion_learner,
    create_simple_diffusion_learner,
    get_default_config,
)
from lerobot_jax.agents.tdmpc2_jax import TDMPC2Agent, TDMPC2Config, create_tdmpc2_learner
from lerobot_jax.utils.norm_utils import LEROBOT_ROOT, compute_normalization_stats

FLAGS = flags.FLAGS
flags.DEFINE_string('algo', 'diffusion', 'Algorithm to use: tdmpc2 or diffusion.')
flags.DEFINE_string('dataset', 'd3il-stacking', 'Dataset to use: lerobot or custom.')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 25000, 'Save interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(32500), 'Number of training steps.')
flags.DEFINE_integer('num_workers', 16, 'Number of workers for dataloader.')
flags.DEFINE_integer('num_devices', 4, 'Number of devices to use.')
flags.DEFINE_string('config_overrides', None, 
    'Comma-separated Hydra config overrides. Eg. --config_overrides="env.param1=value1,env.param2=value2"')

# Add wandb config
wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'ric_experiments',
    'group': 'dmanip',
    'name': 'exp_{algo}_{dataset}'
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)

diffusion_config = get_default_config()
config_flags.DEFINE_config_dict('diffusion', diffusion_config, lock_config=False)

ric_config = get_default_config("ric")
config_flags.DEFINE_config_dict('ric', ric_config, lock_config=False)

tdmpc2_config = TDMPC2Config()
config_flags.DEFINE_config_dict('tdmpc2', ConfigDict(asdict(tdmpc2_config)), lock_config=False)

def load_dataset_and_env(dataset_name, algo_cfg):
    # Parse config overrides if provided
    overrides = FLAGS.config_overrides.split(',') if FLAGS.config_overrides else []
    if hasattr(algo_cfg, 'num_obs_steps'):
        overrides.append(f'policy.n_obs_steps={algo_cfg.num_obs_steps}')
    if hasattr(algo_cfg, 'num_action_steps'):
        overrides.append(f'policy.n_action_steps={algo_cfg.num_action_steps}')
    if hasattr(algo_cfg, 'horizon'):
        overrides.append(f'policy.horizon={algo_cfg.horizon}')

    algo = FLAGS.algo
    if FLAGS.algo == 'ric':
        algo = 'diffusion'
    if dataset_name == 'd3il-stacking-vision':
        base_overrides = [
            "env=d3il_stacking", 
            "+env.use_wrapper=True",
            "device=cpu", 
            f"eval.batch_size={min(50, FLAGS.eval_episodes)}",
            f"eval.n_episodes={FLAGS.eval_episodes}",
            f"policy={algo}_d3il_stacking"
        ]
        cfg = init_hydra_config(
            f"{LEROBOT_ROOT}/lerobot/lerobot/configs/default.yaml",
            overrides=base_overrides + overrides
        )
        env = make_env(cfg)
        dataset = make_dataset(cfg, split="train")
        print(f"Loaded dataset of length {len(dataset)} on CPU")
    elif dataset_name == 'd3il-stacking': 
        policy_name = f"{algo}_d3il_stacking_state" if algo == "diffusion" else "tdmpc2_d3il_stacking"
        base_overrides = [
            "env=d3il_stacking_state", 
            "+env.use_wrapper=True",
            "device=cpu", 
            f"eval.batch_size={min(50, FLAGS.eval_episodes)}",
            f"eval.n_episodes={FLAGS.eval_episodes}",
            f"policy={policy_name}"
        ]
        cfg = init_hydra_config(
            f"{LEROBOT_ROOT}/configs/default.yaml",
            overrides=base_overrides + overrides
        )
        env = make_env(cfg)
        dataset = make_dataset(cfg, split="train")
        print(f"Loaded dataset of length {len(dataset)} on CPU")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return env, dataset, cfg


def main(_):
    if FLAGS.algo == 'tdmpc2':
        algo_cfg = FLAGS.tdmpc2
    elif FLAGS.algo == 'diffusion':
        algo_cfg = FLAGS.diffusion
    elif FLAGS.algo == 'ric':
        algo_cfg = FLAGS.ric
    else:
        raise ValueError(f"Unsupported algorithm: {FLAGS.algo}")

    # Create wandb logger
    setup_wandb(algo_cfg.to_dict(), **FLAGS.wandb)
    # Update config with subset of FLAGS
    wandb.config.update({
        'algo': FLAGS.algo,
        'dataset': FLAGS.dataset,
        'save_dir': FLAGS.save_dir,
        'seed': FLAGS.seed,
        'eval_episodes': FLAGS.eval_episodes,
        'log_interval': FLAGS.log_interval,
        'eval_interval': FLAGS.eval_interval,
        'save_interval': FLAGS.save_interval,
        'batch_size': FLAGS.batch_size,
        'max_steps': FLAGS.max_steps,
        'num_workers': FLAGS.num_workers,
        'num_devices': FLAGS.num_devices,
    })

    # Setup save directory if needed
    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, wandb.config.exp_prefix, wandb.config.experiment_id)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f'Saving config to {FLAGS.save_dir}/config.pkl')
        with open(os.path.join(FLAGS.save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(get_flag_dict(), f)

    # Load dataset
    env, dataset, lerobot_cfg = load_dataset_and_env(FLAGS.dataset, algo_cfg)

    cache_filepath = f"{FLAGS.dataset}_normalization_stats.npy"
    hf_dataset = dataset.hf_dataset.with_format("jax")
    filter_keys = list(lerobot_cfg.policy.input_shapes.keys()) + list(lerobot_cfg.policy.output_shapes.keys())
    if FLAGS.algo == 'tdmpc2':
        filter_keys = filter_keys + ['next.reward']
    hf_dataset = hf_dataset.select_columns(filter_keys)
    normalization_stats = compute_normalization_stats(hf_dataset, filter_keys, FLAGS.num_devices, cache_filepath)
    if not isinstance(normalization_stats, dict):
        normalization_stats = normalization_stats.item()

    normalization_modes = {k: lerobot_cfg.policy.input_normalization_modes[k] == "min_max" for k in lerobot_cfg.policy.input_shapes.keys()}
    normalization_modes.update({k: lerobot_cfg.policy.output_normalization_modes[k] == "min_max" for k in lerobot_cfg.policy.output_shapes.keys()})

    input_shapes = OmegaConf.to_container(lerobot_cfg.policy.input_shapes, resolve=True)
    output_shapes = OmegaConf.to_container(lerobot_cfg.policy.output_shapes, resolve=True)

    # Create numpy dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=FLAGS.num_workers
    )
    batch = next(iter(train_loader))
    sample_batch = tree_map(lambda tensor: tensor.numpy(), {k: batch[k] for k in filter_keys})
    train_iter = cycle(train_loader)

    def get_batch(train_iter):
        batch = next(train_iter)
        batch = tree_map(lambda tensor: tensor.numpy(), {k: batch[k] for k in filter_keys})
        return batch

    # Initialize agent based on selected algorithm
    input_keys = tuple(input_shapes.keys())
    assert len(output_shapes) == 1, "Only one output shape is supported"
    output_key = list(output_shapes.keys())[0]
    rng = jax.random.PRNGKey(FLAGS.seed)
    shape_meta = {
            "batch_size": FLAGS.batch_size,
            "input_shapes": {k: sample_batch[k].shape for k in input_keys},
            "output_shape": {output_key: sample_batch[output_key].shape}
    }
    shape_meta = frozen_dict.freeze(shape_meta)

    if FLAGS.algo == 'tdmpc2':
        # Update config with shapes from environment
        lerobot_cfg = TDMPC2Config(**algo_cfg)
        with jdc.copy_and_mutate(lerobot_cfg, validate=False) as lerobot_cfg:
            lerobot_cfg.action_dim = env.action_space.shape[-1]
            lerobot_cfg.obs = 'rgb' if any(k.startswith('observation.image') for k in input_shapes) else 'state'
            lerobot_cfg.input_shapes = input_shapes
            lerobot_cfg.output_shapes = output_shapes
        agent = create_tdmpc2_learner(lerobot_cfg, rng, normalization_stats, normalization_modes, shape_meta)
        update_fn = functools.partial(agent.update)  #, pmap_axis="i" if FLAGS.num_devices > 1 else None)
        sample_actions = agent.sample_actions

    elif FLAGS.algo == 'diffusion':
        encoder_def = create_input_encoder(input_keys) 
        agent = create_simple_diffusion_learner(
            seed=FLAGS.seed,
            shape_meta=shape_meta,
            output_key=output_key,
            encoder_def=encoder_def,
            config=algo_cfg
        )
        update_fn = functools.partial(agent.update)  #, pmap_axis="i" if FLAGS.num_devices > 1 else None)
        sample_actions = agent.sample_actions
    elif FLAGS.algo == 'ric':
        encoder_def = create_input_encoder(input_keys) 
        agent = create_ric_diffusion_learner(
            seed=FLAGS.seed,
            shape_meta=shape_meta,
            output_key=output_key,
            encoder_def=encoder_def,
            config=algo_cfg
        )
        update_fn = functools.partial(agent.update)  #, pmap_axis="i" if FLAGS.num_devices > 1 else None)
        sample_actions = agent.sample_actions
    else:
        raise ValueError(f"Unsupported algorithm: {FLAGS.algo}")

    # Apply multi-device sharding if needed
    if FLAGS.num_devices > 1:
        mesh = MeshShardingHelper([FLAGS.num_devices], ["fsdp"])

        update_fn = functools.partial(mesh.sjit, in_shardings=[FSDPShardingRule(fsdp_axis_name="fsdp")])(update_fn)
        sample_actions = mesh.sjit(sample_actions)

    else:
        update_fn = jax.jit(update_fn)
        # sample_actions = jax.jit(sample_actions)

    policy_fn = lambda x: np.array(sample_actions(x))  # noqa: E731
    # Training loop
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True):

        # Get next batch (with restart if needed)
        try:
            batch = get_batch(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = get_batch(train_iter)

        # Update step
        agent, metrics = update_fn(batch)

        if i % FLAGS.eval_interval == 0 or i == 1:
            eval_info = evaluate(policy_fn, env, num_episodes=FLAGS.eval_episodes)
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in metrics.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent.model.params, step=i)

if __name__ == '__main__':
    app.run(main) 