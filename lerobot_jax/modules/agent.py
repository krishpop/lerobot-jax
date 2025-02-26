from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from flax import struct
from jaxrl_m.common import TrainState
from jaxrl_m.typing import *

from lerobot_jax.modules.base import PAModule
from lerobot_jax.models.policy import PolicyModel
from lerobot_jax.critics.q_learning import QCritic


class Agent(PAModule):
    """
    Agent class for Policy Agnostic RL.
    
    An agent integrates a policy, critic(s), and other components to provide a complete
    RL agent that can interact with environments and learn from data.
    """
    
    rng: PRNGKey
    policy: PolicyModel
    critic: Optional[QCritic] = None
    
    def act(
        self,
        observations: Dict[str, jnp.ndarray],
        *,
        seed: Optional[PRNGKey] = None,
        evaluation: bool = False,
        **kwargs
    ) -> jnp.ndarray:
        """
        Generate an action for the given observations.
        
        Args:
            observations: Observations from the environment.
            seed: Random seed for sampling. If None, uses the agent's internal rng.
            evaluation: Whether to generate actions for evaluation (less randomness).
            **kwargs: Additional keyword arguments specific to the agent implementation.
            
        Returns:
            Action as a JAX array.
        """
        return self.policy.sample_actions(
            observations,
            seed=seed if seed is not None else self.rng,
            **kwargs
        )
    
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """
        Update the agent's components based on a batch of data.
        
        Args:
            batch: A batch of data containing observations, actions, rewards, etc.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        raise NotImplementedError("Subclasses must implement update")


class ImitationAgent(Agent):
    """
    Agent for imitation learning.
    
    Learns a policy from demonstrations without using rewards.
    """
    
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """
        Update the agent's policy based on a batch of demonstrations.
        
        Args:
            batch: A batch of demonstrations containing observations and actions.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        key, rng = jax.random.split(self.rng)
        # Update the policy using behavior cloning
        policy_info = self.policy.update(batch, pmap_axis=pmap_axis)
        
        return {
            **policy_info,
            **{"agent_rng": rng}
        }


class ActorCriticAgent(Agent):
    """
    Agent for actor-critic methods.
    
    Combines a policy (actor) with a critic to learn from rewards.
    The critic is used to guide policy updates.
    """
    
    critic: QCritic  # Required for actor-critic agents
    
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """
        Update the agent's policy and critic based on a batch of data.
        
        Args:
            batch: A batch of data containing observations, actions, rewards, etc.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        key, rng = jax.random.split(self.rng)
        
        # Update the critic
        critic_info = self.critic.update(batch, pmap_axis=pmap_axis)
        
        # Update the policy using the critic
        key, subkey = jax.random.split(key)
        policy_info = self.policy.update(
            batch, 
            critic=self.critic,
            pmap_axis=pmap_axis
        )
        
        # Sync target networks if needed
        new_critic = self.critic.sync_target()
        
        return {
            **critic_info,
            **policy_info,
            **{"agent_rng": rng},
            **{"agent": self.replace(rng=rng, critic=new_critic)}
        }


class OfflineRLAgent(ActorCriticAgent):
    """
    Agent for offline RL methods.
    
    Specializes actor-critic methods for the offline RL setting, where data
    is fixed and collected by another policy.
    """
    
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """
        Update the agent's policy and critic based on a batch of offline data.
        
        Args:
            batch: A batch of offline data.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        # The specific implementation depends on the offline RL algorithm
        # (e.g., Conservative Q-Learning, IQL, BEAR)
        # This is a placeholder for the interface
        return super().update(batch, pmap_axis=pmap_axis)


class DiffusionPolicyRLAgent(ActorCriticAgent):
    """
    Agent that combines diffusion policies with RL critics.
    
    Specializes actor-critic methods for diffusion-based policies,
    which can capture complex action distributions.
    """
    
    def act(
        self,
        observations: Dict[str, jnp.ndarray],
        *,
        seed: Optional[PRNGKey] = None,
        evaluation: bool = False,
        guidance_scale: float = 1.0,
        **kwargs
    ) -> jnp.ndarray:
        """
        Generate an action using the diffusion policy, optionally guided by the critic.
        
        Args:
            observations: Observations from the environment.
            seed: Random seed for sampling. If None, uses the agent's internal rng.
            evaluation: Whether to generate actions for evaluation (less randomness).
            guidance_scale: Scale for critic-guided sampling. Higher values prioritize
                          actions with higher Q-values.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Action as a JAX array.
        """
        return self.policy.sample_actions(
            observations,
            seed=seed if seed is not None else self.rng,
            guidance_scale=guidance_scale if evaluation else 1.0,
            **kwargs
        )
    
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """
        Update the diffusion policy and critic.
        
        Args:
            batch: A batch of data.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        # The specific implementation depends on the particular algorithm
        # (e.g., RIC-diffusion, TD-MPC with diffusion)
        # This is a placeholder for the interface
        return super().update(batch, pmap_axis=pmap_axis)


def create_agent(
    agent_type: str,
    policy: PolicyModel,
    critic: Optional[QCritic] = None,
    seed: int = 0,
    config: Optional[ml_collections.ConfigDict] = None,
    **kwargs
) -> Agent:
    """
    Factory function to create agents based on the specified type.
    
    Args:
        agent_type: Type of agent to create ("imitation", "actor_critic", "offline_rl", "diffusion_rl").
        policy: Policy model to use.
        critic: Optional critic model to use.
        seed: Random seed.
        config: Configuration for the agent.
        **kwargs: Additional keyword arguments for agent initialization.
        
    Returns:
        An initialized agent.
    """
    rng = jax.random.PRNGKey(seed)
    
    if agent_type == "imitation":
        return ImitationAgent(rng=rng, policy=policy)
    
    elif agent_type == "actor_critic":
        if critic is None:
            raise ValueError("Actor-critic agents require a critic")
        return ActorCriticAgent(rng=rng, policy=policy, critic=critic)
    
    elif agent_type == "offline_rl":
        if critic is None:
            raise ValueError("Offline RL agents require a critic")
        return OfflineRLAgent(rng=rng, policy=policy, critic=critic)
    
    elif agent_type == "diffusion_rl":
        if critic is None:
            raise ValueError("Diffusion RL agents require a critic")
        return DiffusionPolicyRLAgent(rng=rng, policy=policy, critic=critic)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 