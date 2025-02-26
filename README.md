# lerobot-jax: Policy Agnostic RL Framework

lerobot-jax is a JAX-based library for training multi-task policies using a Policy Agnostic RL (PA-RL) approach. The package provides modular components for reinforcement learning and imitation learning, with a focus on supporting diverse policy architectures such as diffusion models and autoregressive transformers.

## Features

- **Modular Design**: Compose agents from policies, critics, and replay buffers
- **Policy Agnostic**: Support for multiple policy architectures (Gaussian, diffusion, autoregressive)
- **Q-Learning**: Well-calibrated critics for evaluating and optimizing actions
- **Flexible Training**: Support for offline RL, imitation learning, and online fine-tuning

## Architecture

The package is organized into several modular components:

### Modules

- `PAModule`: Base abstract class for all PA-RL components
- `Agent`: Integrates policies and critics into a complete RL agent
  - `ImitationAgent`: For imitation learning without rewards
  - `ActorCriticAgent`: For methods that combine policy and value learning
  - `OfflineRLAgent`: Specialized for offline RL with fixed datasets
  - `DiffusionPolicyRLAgent`: Combines diffusion policies with RL critics

### Models

- `PolicyModel`: Base class for policy models
  - `GaussianPolicy`: Classic Gaussian policies for simple RL algorithms
  - `DiffusionPolicy`: Generates actions using denoising diffusion
  - `AutoregressivePolicy`: Generates actions sequentially

### Critics

- `QCritic`: Base class for Q-function critics
  - `EnsembleQCritic`: Uses multiple Q-networks to reduce overestimation bias
  - `DistributionalQCritic`: Models the full distribution of returns

### Replay Buffers

- `ReplayBuffer`: For storing and sampling transitions
- `OfflineDataset`: For working with pre-collected datasets

## Installation

To install lerobot-jax, clone the repository and run:

```bash
pip install -e .
```

## Usage

### Creating an Agent

```python
import lerobot_jax as ljx
import jax

# Create a policy
policy = ljx.create_policy_model(
    policy_type="diffusion",
    seed=0,
    shape_meta=shape_meta,
    config=policy_config,
)

# Create a critic
critic = ljx.create_q_critic(
    critic_type="ensemble",
    seed=1,
    shape_meta=shape_meta,
    config=critic_config,
)

# Create an agent
agent = ljx.create_agent(
    agent_type="diffusion_rl",
    policy=policy,
    critic=critic,
    seed=2,
    config=agent_config,
)
```

### Training an Agent

```python
# Load a dataset
dataset = ljx.OfflineDataset.load_from_path(
    path="path/to/dataset",
    normalize=True,
    seed=3,
)

# Training loop
for step in range(num_steps):
    # Sample batch from dataset
    dataset, batch = dataset.sample(batch_size)
    
    # Update agent
    info = agent.update(batch)
    
    # Extract updated agent if returned in info
    if "agent" in info:
        agent = info["agent"]
```

### Generating Actions

```python
# Generate an action
action = agent.act(
    observations,
    evaluation=True,
    guidance_scale=2.0,
)
```

## Examples

See the `examples` directory for complete examples of different agents and training setups:

- `diff_rl_example.py`: Combining a diffusion policy with a Q-critic for offline RL

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details. 

