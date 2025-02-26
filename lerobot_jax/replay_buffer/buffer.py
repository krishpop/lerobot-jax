from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *

from lerobot_jax.modules.base import PAModule


class ReplayBuffer(PAModule):
    """
    Base replay buffer for Policy Agnostic RL.
    
    Replay buffers store transitions (observation, action, reward, next_observation, done)
    and provide methods to sample batches for training.
    """
    
    data: Dict[str, jnp.ndarray]
    rng: PRNGKey
    max_size: int
    current_size: int = 0
    insert_index: int = 0
    
    @classmethod
    def create(
        cls,
        observation_space_shape: Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]],
        action_space_shape: Tuple[int, ...],
        max_size: int,
        seed: int = 0,
    ) -> "ReplayBuffer":
        """
        Create a new replay buffer with empty arrays.
        
        Args:
            observation_space_shape: Shape of observations (or dict of shapes).
            action_space_shape: Shape of actions.
            max_size: Maximum number of transitions to store.
            seed: Random seed for sampling.
            
        Returns:
            An initialized empty replay buffer.
        """
        rng = jax.random.PRNGKey(seed)
        
        if isinstance(observation_space_shape, dict):
            observations = {
                k: np.zeros((max_size, *shape), dtype=np.float32)
                for k, shape in observation_space_shape.items()
            }
        else:
            observations = {
                "observations": np.zeros(
                    (max_size, *observation_space_shape), dtype=np.float32
                )
            }
        
        data = {
            **observations,
            "actions": np.zeros((max_size, *action_space_shape), dtype=np.float32),
            "rewards": np.zeros((max_size, 1), dtype=np.float32),
            "next_observations": {
                k: np.zeros_like(v) for k, v in observations.items()
            },
            "dones": np.zeros((max_size, 1), dtype=bool),
        }
        
        return cls(data=data, rng=rng, max_size=max_size)
    
    def insert(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        action: np.ndarray,
        reward: float,
        next_observation: Union[np.ndarray, Dict[str, np.ndarray]],
        done: bool,
    ) -> "ReplayBuffer":
        """
        Insert a transition into the replay buffer.
        
        Args:
            observation: Observation at the current step.
            action: Action taken at the current step.
            reward: Reward received after taking the action.
            next_observation: Observation at the next step.
            done: Whether the episode is done.
            
        Returns:
            The replay buffer with the inserted transition.
        """
        data = dict(self.data)
        
        # Handle observations
        if isinstance(observation, dict):
            for k, v in observation.items():
                data[k][self.insert_index] = v
                data["next_observations"][k][self.insert_index] = next_observation[k]
        else:
            data["observations"][self.insert_index] = observation
            data["next_observations"]["observations"][self.insert_index] = next_observation
        
        # Insert other data
        data["actions"][self.insert_index] = action
        data["rewards"][self.insert_index] = reward
        data["dones"][self.insert_index] = done
        
        # Update buffer indices
        insert_index = (self.insert_index + 1) % self.max_size
        current_size = min(self.current_size + 1, self.max_size)
        
        return self.replace(
            data=data,
            insert_index=insert_index,
            current_size=current_size,
        )
    
    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            A batch of transitions.
        """
        # Generate random indices
        key, rng = jax.random.split(self.rng)
        indices = jax.random.randint(
            key, (batch_size,), 0, self.current_size
        )
        
        # Sample data
        batch = {}
        for k, v in self.data.items():
            if k == "next_observations":
                for next_k, next_v in self.data[k].items():
                    batch[f"next_{next_k}"] = next_v[indices]
            else:
                batch[k] = v[indices]
        
        return self.replace(rng=rng), batch


class OfflineDataset(PAModule):
    """
    Dataset for offline RL and imitation learning.
    
    Provides an interface to load and sample from pre-collected demonstrations
    or offline RL datasets.
    """
    
    dataset: Dataset
    rng: PRNGKey
    
    @classmethod
    def from_jaxrl_m_dataset(
        cls,
        dataset: Dataset,
        seed: int = 0,
    ) -> "OfflineDataset":
        """
        Create an offline dataset from a jaxrl_m Dataset.
        
        Args:
            dataset: A jaxrl_m Dataset.
            seed: Random seed for sampling.
            
        Returns:
            An offline dataset.
        """
        rng = jax.random.PRNGKey(seed)
        return cls(dataset=dataset, rng=rng)
    
    @classmethod
    def load_from_path(
        cls,
        path: str,
        n_demonstrations: Optional[int] = None,
        normalize: bool = True,
        image_keys: Tuple[str, ...] = (),
        seed: int = 0,
    ) -> "OfflineDataset":
        """
        Load a dataset from a path.
        
        Args:
            path: Path to the dataset.
            n_demonstrations: Number of demonstrations to load (None for all).
            normalize: Whether to normalize the dataset.
            image_keys: Keys for image observations.
            seed: Random seed for sampling.
            
        Returns:
            An offline dataset.
        """
        raise NotImplementedError("Loading datasets from paths to be implemented")
    
    def normalize(self, eps: float = 1e-5) -> "OfflineDataset":
        """
        Normalize observations and actions in the dataset.
        
        Args:
            eps: Small constant for numerical stability.
            
        Returns:
            The normalized dataset.
        """
        # The implementation depends on the dataset format
        # This is a placeholder for the interface
        raise NotImplementedError("Dataset normalization to be implemented")
    
    def sample(self, batch_size: int) -> Batch:
        """
        Sample a batch of transitions from the dataset.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            A batch of transitions.
        """
        key, rng = jax.random.split(self.rng)
        batch = self.dataset.sample(batch_size, key)
        return self.replace(rng=rng), batch
    
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """
        Datasets don't have update logic, so this is a no-op implementation.
        
        Args:
            batch: A batch of data.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        return {} 