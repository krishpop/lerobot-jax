"""
Example of using the new Policy Agnostic RL (PA-RL) framework with a diffusion policy and Q-critic.

This example demonstrates how to:
1. Create a diffusion policy
2. Create an ensemble Q-critic
3. Combine them into an offline RL agent
4. Load a dataset and train the agent
5. Evaluate the trained agent
"""

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import os
import time
from typing import Dict

# Import lerobot_jax components
import lerobot_jax as ljx
from lerobot_jax.models.policy import DiffusionPolicy, create_policy_model
from lerobot_jax.critics.q_learning import EnsembleQCritic, create_q_critic
from lerobot_jax.modules.agent import DiffusionPolicyRLAgent, create_agent
from lerobot_jax.replay_buffer.buffer import OfflineDataset

# Example configuration for a diffusion policy + Q-critic agent
def get_config():
    config = ConfigDict()
    
    # General settings
    config.seed = 42
    config.batch_size = 256
    config.num_steps = 500000
    config.eval_interval = 10000
    config.log_interval = 1000
    
    # Diffusion policy settings
    config.policy = ConfigDict()
    config.policy.type = "diffusion"
    config.policy.num_inference_steps = 50
    config.policy.diffusion_steps = 1000
    config.policy.noise_schedule = "linear"
    config.policy.hidden_dims = (256, 256, 256)
    config.policy.learning_rate = 1e-4
    
    # Q-critic settings
    config.critic = ConfigDict()
    config.critic.type = "ensemble"
    config.critic.num_qs = 5
    config.critic.hidden_dims = (256, 256, 256)
    config.critic.learning_rate = 3e-4
    config.critic.discount = 0.99
    config.critic.tau = 0.005
    
    # Agent settings
    config.agent = ConfigDict()
    config.agent.type = "diffusion_rl"
    config.agent.guidance_scale = 2.0
    
    # Dataset settings
    config.dataset = ConfigDict()
    config.dataset.path = "path/to/your/dataset"
    config.dataset.normalize = True
    
    return config


def main():
    # Get configuration
    config = get_config()
    
    # Set random seed
    seed = config.seed
    rng = jax.random.PRNGKey(seed)
    
    # Define shapes for observation and action spaces
    # This would typically come from your environment or dataset
    observation_shape = {"image": (3, 84, 84), "state": (10,)}
    action_shape = (7,)
    
    # Create shape_meta dictionary with detailed shapes
    shape_meta = {
        "observations": {
            "image": {"shape": observation_shape["image"], "type": "continuous"},
            "state": {"shape": observation_shape["state"], "type": "continuous"},
        },
        "actions": {"shape": action_shape, "type": "continuous"},
    }
    
    # Split rng for different components
    rng, policy_rng, critic_rng, dataset_rng = jax.random.split(rng, 4)
    
    print("Creating policy model...")
    # Create policy model
    policy = create_policy_model(
        policy_type=config.policy.type,
        seed=jax.random.randint(policy_rng, (), 0, 2**31 - 1).item(),
        shape_meta=shape_meta,
        config=config.policy,
    )
    
    print("Creating critic model...")
    # Create critic model
    critic = create_q_critic(
        critic_type=config.critic.type,
        seed=jax.random.randint(critic_rng, (), 0, 2**31 - 1).item(),
        shape_meta=shape_meta,
        config=config.critic,
    )
    
    print("Creating agent...")
    # Create agent that combines policy and critic
    agent = create_agent(
        agent_type=config.agent.type,
        policy=policy,
        critic=critic,
        seed=jax.random.randint(rng, (), 0, 2**31 - 1).item(),
        config=config.agent,
    )
    
    print("Loading dataset...")
    # Load dataset
    dataset = OfflineDataset.load_from_path(
        path=config.dataset.path,
        normalize=config.dataset.normalize,
        seed=jax.random.randint(dataset_rng, (), 0, 2**31 - 1).item(),
    )
    
    print("Training agent...")
    # Training loop
    for step in range(config.num_steps):
        # Sample batch from dataset
        dataset, batch = dataset.sample(config.batch_size)
        
        # Update agent
        info = agent.update(batch)
        
        # Extract updated agent if returned in info
        if "agent" in info:
            agent = info["agent"]
        
        # Log progress
        if step % config.log_interval == 0:
            policy_loss = info.get("policy_loss", 0.0)
            critic_loss = info.get("critic_loss", 0.0)
            print(f"Step {step}: Policy Loss = {policy_loss:.4f}, Critic Loss = {critic_loss:.4f}")
        
        # Evaluate agent
        if step % config.eval_interval == 0:
            print(f"Evaluating agent at step {step}...")
            # Evaluation code would go here
    
    print("Training complete!")


if __name__ == "__main__":
    main() 