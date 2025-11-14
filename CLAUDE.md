# CLAUDE.md - AI Assistant Guide for lerobot-jax

## Overview

**lerobot-jax** is a JAX-based library for training multi-task policies using a Policy Agnostic RL (PA-RL) approach. The package provides modular components for reinforcement learning and imitation learning, with a focus on supporting diverse policy architectures such as diffusion models, autoregressive transformers, and TDMPC2.

**Key Design Principles:**
- Policy Agnostic: Support for multiple policy architectures (Gaussian, diffusion, autoregressive)
- Modular Design: Compose agents from policies, critics, and replay buffers
- JAX-Native: Built entirely on JAX/Flax for high-performance gradient computations
- Multi-Task Learning: Designed for training policies across multiple tasks

## Repository Structure

```
lerobot-jax/
├── lerobot_jax/                   # Main package directory
│   ├── agents/                    # Complete agent implementations
│   │   ├── diffusion_jax.py      # Diffusion-based RL agents
│   │   ├── tdmpc2_jax.py         # TDMPC2 implementation
│   │   └── tdmpc2_jax_v1.py      # TDMPC2 with policy regularization
│   ├── critics/                   # Q-learning and value critics
│   │   └── q_learning.py         # Q-critic implementations
│   ├── models/                    # Policy model definitions
│   │   ├── policy.py             # Base policy classes (Gaussian, Diffusion, Autoregressive)
│   │   └── tdmpc2_jax_v1.py      # TDMPC2 model components
│   ├── modules/                   # Core PA-RL modules
│   │   ├── base.py               # PAModule base class
│   │   └── agent.py              # Agent classes (Imitation, ActorCritic, etc.)
│   ├── replay_buffer/             # Data storage and sampling
│   │   └── buffer.py             # ReplayBuffer and OfflineDataset
│   ├── utils/                     # Utility functions
│   │   ├── model_utils.py        # Model helpers (MLP, encoders)
│   │   ├── norm_utils.py         # Normalization statistics
│   │   ├── logging.py            # Logging utilities
│   │   └── tdmpc2_utils.py       # TDMPC2-specific helpers
│   └── examples/                  # Example usage scripts
│       └── diff_rl_example.py    # Diffusion RL example
├── scripts/                       # Training and experiment scripts
│   └── train_jax.py              # Main training script
├── setup.py                       # Package installation
├── requirements.txt               # Additional dependencies
└── README.md                      # User-facing documentation
```

## Core Architecture

### 1. PAModule System (`lerobot_jax/modules/`)

The **PAModule** (Policy Agnostic Module) is the base abstraction that all components inherit from:

**Location:** `lerobot_jax/modules/base.py:12`

**Key Methods:**
- `update(batch, pmap_axis=None)`: Update module parameters from a batch
- `save(save_path)`: Save parameters to disk
- `load(load_path)`: Load parameters from disk

**Design Rationale:** PAModule provides a common interface for composing agents, policies, critics, and other components. This enables mixing and matching different architectures while maintaining a consistent API.

### 2. Agent Hierarchy (`lerobot_jax/modules/agent.py`)

**Agent Types:**

1. **Agent** (`agent.py:17`) - Base agent class
   - Integrates policy, critic(s), and other components
   - Provides `act()` method for generating actions
   - Abstract `update()` method for learning

2. **ImitationAgent** (`agent.py:69`) - Imitation learning
   - Learns from demonstrations without rewards
   - Updates only the policy via behavior cloning
   - Use case: Learning from expert trajectories

3. **ActorCriticAgent** (`agent.py:97`) - Actor-critic methods
   - Combines policy (actor) with critic for value estimation
   - Updates both policy and critic in tandem
   - Use case: Online RL, policy gradient methods

4. **OfflineRLAgent** (`agent.py:142`) - Offline RL
   - Specializes actor-critic for fixed datasets
   - No environment interaction during training
   - Use case: Learning from pre-collected data

5. **DiffusionPolicyRLAgent** (`agent.py:167`) - Diffusion + RL
   - Combines diffusion policies with Q-critics
   - Supports critic-guided sampling via `guidance_scale`
   - Use case: Complex action distributions with RL optimization

**Factory Function:** `create_agent()` (`agent.py:222`) - Creates agents by type string

### 3. Policy Models (`lerobot_jax/models/policy.py`)

**Policy Types:**

1. **GaussianPolicy** (`policy.py:61`)
   - Outputs mean and log_std for Gaussian distribution
   - Simple baseline for continuous control
   - Temperature-controlled exploration

2. **DiffusionPolicy** (`policy.py:109`)
   - Generates actions via denoising diffusion
   - Captures multimodal action distributions
   - Supports classifier-free guidance

3. **AutoregressivePolicy** (`policy.py:147`)
   - Generates actions sequentially
   - Each element conditions on previous elements
   - Suitable for structured action spaces

**Factory Function:** `create_policy_model()` (`policy.py:180`)

### 4. Critics (`lerobot_jax/critics/q_learning.py`)

**Critic Types:**
- **QCritic**: Base Q-function critic
- **EnsembleQCritic**: Multiple Q-networks to reduce overestimation
- **DistributionalQCritic**: Models full return distribution

**Factory Function:** `create_q_critic()`

### 5. Replay Buffers (`lerobot_jax/replay_buffer/buffer.py`)

- **ReplayBuffer**: For online RL with experience storage
- **OfflineDataset**: For pre-collected datasets

## Key Dependencies

**Core ML Stack:**
- `jax` / `jaxlib`: Core computation framework
- `flax`: Neural network library
- `optax`: Optimization library
- `ml-collections`: Configuration management

**Robot Learning:**
- `lerobot`: Dataset and environment utilities (from HuggingFace)
- `mani_skill`: Simulation environments
- `sapien`: Physics simulation

**Additional:**
- `diffusers`: Diffusion model components
- `wandb`: Experiment tracking
- `hydra-core` / `omegaconf`: Configuration system
- `jax_dataclasses`: JAX-compatible dataclasses
- `scalax`: Distributed training utilities

## Development Workflow

### Training Scripts

**Main Entry Point:** `scripts/train_jax.py`

**Usage Pattern:**
```bash
python scripts/train_jax.py \
  --algo=diffusion \
  --dataset=d3il-stacking \
  --seed=42 \
  --batch_size=128 \
  --max_steps=32500 \
  --num_devices=4
```

**Supported Algorithms:**
- `--algo=diffusion`: Diffusion policy
- `--algo=ric`: RIC (Regularized with Inverse dynamics Constraint) diffusion
- `--algo=tdmpc2`: TD-MPC^2 algorithm

**Key FLAGS:**
- `--dataset`: Dataset name (e.g., 'd3il-stacking', 'd3il-stacking-vision')
- `--save_dir`: Directory for saving checkpoints
- `--eval_episodes`: Number of evaluation episodes (default: 50)
- `--eval_interval`: Steps between evaluations (default: 10000)
- `--num_devices`: Number of GPUs/TPUs for distributed training
- `--config_overrides`: Hydra config overrides (comma-separated)

**Configuration System:**
The codebase uses a hybrid configuration approach:
1. **Hydra configs** from `lerobot` for environments and datasets
2. **ml-collections ConfigDict** for algorithm hyperparameters
3. **absl FLAGS** for command-line arguments

### Data Pipeline

**Flow:** `train_jax.py:74-123`
1. Initialize Hydra config with overrides
2. Create environment via `make_env(cfg)`
3. Load dataset via `make_dataset(cfg, split="train")`
4. Compute normalization statistics (cached to `.npy` file)
5. Create PyTorch DataLoader (converted to JAX arrays)
6. Cycle through batches during training

**Normalization:**
- Statistics computed from entire dataset: `norm_utils.py`
- Modes: `mean_std` or `min_max` (specified in policy config)
- Cached to avoid recomputation: `{dataset}_normalization_stats.npy`

### Distributed Training

**Multi-Device Support:** `train_jax.py:294-303`
- Uses `scalax.sharding.MeshShardingHelper` for FSDP
- Shards data across devices with `FSDPShardingRule`
- Applies `sjit` (sharded JIT) when `num_devices > 1`

### Evaluation

**Pattern:** `train_jax.py:319-322`
- Uses `jaxrl_m.evaluation.evaluate()` for policy evaluation
- Runs `eval_episodes` rollouts in environment
- Logs to WandB with `evaluation/` prefix

### Checkpointing

**Pattern:** `train_jax.py:328-329`
- Uses `flax.training.checkpoints.save_checkpoint()`
- Saves agent parameters at `save_interval` steps
- Directory structure: `{save_dir}/{project}/{exp_prefix}/{experiment_id}/`

## Coding Conventions

### Style Guidelines

1. **Type Annotations:**
   - Use `jaxrl_m.typing.*` for common types (PRNGKey, Batch, InfoDict)
   - Annotate function signatures with types
   - Use `Optional[T]` for nullable parameters

2. **Naming Conventions:**
   - Classes: PascalCase (e.g., `DiffusionPolicy`, `PAModule`)
   - Functions: snake_case (e.g., `create_agent`, `sample_actions`)
   - Constants: UPPER_SNAKE_CASE (e.g., `LEROBOT_ROOT`)
   - Private members: Leading underscore (e.g., `_internal_fn`)

3. **Docstrings:**
   - Use Google-style docstrings
   - Include Args, Returns, Raises sections
   - Provide usage examples for public APIs

4. **JAX Best Practices:**
   - Keep functions pure (no side effects)
   - Use `jax.random.split()` for PRNG management
   - Mark arrays with `jnp.ndarray` instead of `np.ndarray`
   - Use `@jax.jit` for performance-critical functions

### Module Organization

**Pattern for New Components:**

```python
# 1. Imports
from typing import ...
import jax
from flax import struct
from lerobot_jax.modules.base import PAModule

# 2. Class definition
class MyComponent(PAModule):
    """Docstring with purpose and usage."""

    # 3. Attributes (Flax struct.PyTreeNode style)
    rng: PRNGKey
    model: TrainState
    config: ConfigDict = struct.field(pytree_node=False)

    # 4. Methods
    def update(self, batch: Batch, pmap_axis=None) -> InfoDict:
        """Implementation."""
        pass

# 5. Factory function
def create_my_component(...) -> MyComponent:
    """Factory for creating component."""
    pass
```

### Configuration Patterns

**Algorithm Configs:**
- Define in `agents/{algo}_jax.py` as `get_default_config()`
- Return `ml_collections.ConfigDict` with hyperparameters
- Lock configs with `lock_config=False` for overrides

**Example:**
```python
def get_default_config(variant="default"):
    config = ConfigDict()
    config.learning_rate = 3e-4
    config.batch_size = 256
    # ... more hyperparameters
    return config
```

### Testing Patterns

**Current State:**
- No formal test suite in repository
- Testing done via training scripts and examples

**Recommended for Contributors:**
- Add unit tests for new modules in `tests/` directory
- Use `pytest` for test framework
- Test shape invariants with different batch sizes
- Verify gradient flow through custom layers

## Common Tasks

### Adding a New Policy Architecture

1. **Create policy class in `lerobot_jax/models/policy.py`:**
   ```python
   class MyPolicy(PolicyModel):
       def sample_actions(self, observations, *, seed=None, **kwargs):
           # Implementation
           pass

       def update(self, batch, pmap_axis=None, **kwargs):
           # Implementation
           pass
   ```

2. **Update `create_policy_model()` factory:**
   ```python
   elif policy_type == "my_policy":
       return MyPolicy(...)
   ```

3. **Create agent implementation in `lerobot_jax/agents/`:**
   - Define network architecture
   - Implement training loops
   - Add configuration

4. **Update `train_jax.py` to support new policy:**
   - Add config flags
   - Add branch in agent initialization

### Adding a New Environment/Dataset

1. **Create Hydra config in lerobot:**
   - Add environment config: `env=my_env`
   - Add policy config: `policy=diffusion_my_env`

2. **Update `train_jax.py:load_dataset_and_env()`:**
   ```python
   elif dataset_name == 'my-dataset':
       base_overrides = [
           "env=my_env",
           "device=cpu",
           f"policy={algo}_my_env"
       ]
       cfg = init_hydra_config(...)
   ```

3. **Test normalization:**
   - Verify normalization modes in policy config
   - Check filter_keys match dataset fields

### Debugging Training Issues

**Common Issues:**

1. **Shape Mismatches:**
   - Check `shape_meta` in agent creation
   - Verify `input_shapes` and `output_shapes` match dataset
   - Use debug prints: `print(f"DEBUG: {batch[key].shape}")`

2. **Normalization Errors:**
   - Check `normalization_modes` dictionary keys
   - Verify stats computed correctly: `normalization_stats.keys()`
   - Look for missing keys in policy config

3. **OOM (Out of Memory):**
   - Reduce `batch_size`
   - Reduce `num_devices` if using FSDP
   - Check for gradient accumulation issues

4. **NaN Losses:**
   - Check learning rates (may be too high)
   - Verify gradient clipping is enabled
   - Check for division by zero in custom losses

**Debug Print Pattern:** (See `train_jax.py:175-204`)
```python
try:
    # Your code
    print(f"DEBUG: variable_name: {variable_name}")
except Exception as e:
    print(f"DEBUG: Exception: {e}")
    import traceback
    traceback.print_exc()
```

## Integration with Lerobot

**Dependency:** This codebase depends on the `lerobot` library from HuggingFace for:
- Dataset loading: `lerobot.common.datasets.factory.make_dataset`
- Environment creation: `lerobot.common.envs.factory.make_env`
- Configuration: `lerobot.common.utils.utils.init_hydra_config`

**LEROBOT_ROOT:** Environment variable pointing to lerobot installation
- Used in `norm_utils.py` to locate config files
- Required for loading Hydra configs

**Dataset Format:**
- HuggingFace datasets with `.hf_dataset` attribute
- Converted to JAX format: `dataset.hf_dataset.with_format("jax")`
- Fields selected via `select_columns(filter_keys)`

## Git Workflow

**Branch Naming:**
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Development: `claude/claude-md-{session-id}` (for AI assistant work)

**Commit Style:** (from `git log`)
- Imperative mood: "Add X", "Fix Y", "Refactor Z"
- Keep commits atomic and focused
- Include context in commit messages

**Recent Development Focus:** (from commit history)
- TD-M(PC)^2 policy regularization (v1)
- PA-RL module refactoring
- Normalization and data pipeline improvements
- TDMPC2 integration with eval state

## WandB Integration

**Configuration:** `train_jax.py:56-63`
```python
wandb_config = {
    'project': 'ric_experiments',
    'group': 'dmanip',
    'name': 'exp_{algo}_{dataset}'
}
```

**Logging:**
- Training metrics: `training/{metric_name}`
- Evaluation metrics: `evaluation/{metric_name}`
- Config tracking: `wandb.config.update(...)`

## Important Notes for AI Assistants

### DO:
- ✅ Preserve the PAModule abstraction when adding components
- ✅ Use factory functions (`create_agent`, `create_policy_model`, etc.)
- ✅ Follow JAX functional programming patterns (pure functions, no mutation)
- ✅ Add comprehensive docstrings to new classes/functions
- ✅ Test with multiple batch sizes and device counts
- ✅ Cache normalization statistics for faster iteration
- ✅ Use WandB for experiment tracking

### DON'T:
- ❌ Mutate arrays in-place (use `.replace()` for struct updates)
- ❌ Mix NumPy and JAX arrays without explicit conversion
- ❌ Forget to split RNG keys before reuse
- ❌ Hardcode shapes (use `shape_meta` instead)
- ❌ Skip normalization (leads to training instability)
- ❌ Modify lerobot configs directly (use overrides instead)

### When Modifying Code:

1. **Check Dependencies:**
   - Imports from `jaxrl_m` (external dependency)
   - Imports from `lerobot` (external dependency)
   - Internal module structure

2. **Verify Shapes:**
   - Input observations: `Dict[str, jnp.ndarray]`
   - Actions: `jnp.ndarray` with shape `(batch, action_dim, ...)`
   - Batches: Match dataset schema

3. **Test Changes:**
   - Run training script with small `max_steps` (e.g., 100)
   - Check for shape errors, NaNs, OOM
   - Verify metrics logged to WandB

4. **Document Changes:**
   - Update docstrings if API changes
   - Add comments for non-obvious logic
   - Update this CLAUDE.md if architecture changes

## Quick Reference

### File Locations
- Main training: `scripts/train_jax.py`
- Base abstractions: `lerobot_jax/modules/base.py`
- Agent types: `lerobot_jax/modules/agent.py`
- Policy models: `lerobot_jax/models/policy.py`
- Critics: `lerobot_jax/critics/q_learning.py`
- Utilities: `lerobot_jax/utils/`

### Key Classes
- `PAModule`: Base class for all components
- `Agent`: Base agent (integrates policy + critic)
- `PolicyModel`: Base policy class
- `QCritic`: Base Q-function critic

### Factory Functions
- `create_agent(agent_type, ...)`
- `create_policy_model(policy_type, ...)`
- `create_q_critic(critic_type, ...)`

### External Dependencies
- `jaxrl_m`: RL utilities (TrainState, types, evaluation)
- `lerobot`: Datasets, environments, configs
- `scalax`: Distributed training (FSDP, mesh sharding)

---

**Last Updated:** 2025-11-14
**Version:** 0.1.0 (early development)
