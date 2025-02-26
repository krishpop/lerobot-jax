import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from jaxrl_m.common import TrainState
from jaxrl_m.typing import *


class PAModule(abc.ABC, struct.PyTreeNode):
    """
    Base class for Policy Agnostic RL modules.
    
    PA-RL modules follow a common interface to enable composition of agents, models,
    Q-learning, and replay buffer/dataset components across different policy architectures.
    """
    
    @abc.abstractmethod
    def update(self, batch: Batch, pmap_axis: Optional[str] = None) -> InfoDict:
        """Update the module parameters based on a batch of data.
        
        Args:
            batch: A batch of data containing observations, actions, rewards, etc.
            pmap_axis: Axis name used for pmap-based parallel training.
            
        Returns:
            InfoDict containing metrics and information about the update.
        """
        pass
    
    def save(self, save_path: str) -> None:
        """Save module parameters to a file.
        
        Args:
            save_path: Path to save the parameters to.
        """
        # Default implementation - override if needed
        pass
    
    def load(self, load_path: str) -> "PAModule":
        """Load module parameters from a file.
        
        Args:
            load_path: Path to load the parameters from.
            
        Returns:
            The module with loaded parameters.
        """
        # Default implementation - override if needed
        return self 