# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

lerobot-jax is a JAX-based library for training multi-task policies using a combination of offline RL and imitation learning techniques. The library primarily focuses on two learning approaches:

1. **Diffusion Models**: Generation-based approach for learning action sequences
2. **TDMPC2**: Model-based reinforcement learning with trajectory optimization

## Training Commands

```bash
# Train a diffusion-based policy on d3il-stacking dataset
python scripts/train_jax.py --algo=diffusion --dataset=d3il-stacking --save_dir=./checkpoints

# Train a TDMPC2 policy on d3il-stacking dataset
python scripts/train_jax.py --algo=tdmpc2 --dataset=d3il-stacking --save_dir=./checkpoints

# Training with custom hyperparameters
python scripts/train_jax.py --algo=diffusion --dataset=d3il-stacking --diffusion.optim_kwargs.learning_rate=5e-4

# Training RIC (Reinforcement Implicit Consistency) variant
python scripts/train_jax.py --algo=ric --dataset=d3il-stacking --save_dir=./checkpoints
```

## Code Architecture

### Core Components

1. **Agents**: Policy implementations
   - `diffusion_jax.py`: Diffusion-based policy learning for action generation
   - `tdmpc2_jax.py`: Model-based RL with trajectory optimization

2. **Utils**: Shared utilities 
   - `norm_utils.py`: Data normalization utilities
   - `model_utils.py`: Common model building blocks (MLP, encoders, etc.)
   - `tdmpc2_utils.py`: TDMPC2-specific utilities
   - `logging.py`: Metrics collection utilities

3. **Training Script**: 
   - `train_jax.py`: Entry point for training models

### Agent Flow

#### Diffusion Agent
- Uses denoising diffusion models to generate actions based on observations
- Provides conditional generation for action sequences
- Supports classifier-free guidance in RIC variant

#### TDMPC2 Agent
- Maintains a world model to predict next states, rewards, and Q-values
- Uses trajectory optimization to select actions
- Provides model-based value estimation with distributional critics

## Key Implementation Details

1. **Normalization**
   - Input/output normalization is handled through `norm_utils.py` 
   - Normalization statistics are computed once and cached

2. **Action Sampling**
   - Diffusion model: Denoising process from random noise
   - TDMPC2: Planning with world model predictions

3. **Multi-device Training**
   - Uses JAX's SPMD for multi-device training when available

4. **Data Preprocessing**
   - Support for both state-based and vision-based inputs
   - Normalization statistics are precomputed and cached

## Working with the Codebase

When modifying the code:

1. New agent variants should extend existing agent classes: `SimpleDiffusionAgent` or `TDMPC2Agent`
2. Normalization must be applied consistently during both training and inference
3. Config dataclasses (e.g., `TDMPC2Config`) define all hyperparameters
4. JAX-based training utilizes functional programming patterns
5. Model updates use immutable data structures with TrainState replacements