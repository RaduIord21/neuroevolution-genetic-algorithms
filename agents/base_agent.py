"""
Base agent class providing common functionality for all agent types.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Defines the interface and shared functionality.
    """
    
    def __init__(self, agent_id: int = None):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent (e.g., in a population)
        """
        self.agent_id = agent_id
        self.fitness = 0.0
        self.episode_rewards = []
        self.birth_generation = 0
    
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> int:
        """
        Compute action from observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get all trainable parameters as a flat array."""
        pass
    
    @abstractmethod
    def set_weights(self, weights: np.ndarray):
        """Set trainable parameters from a flat array."""
        pass
    
    @abstractmethod
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        pass
    
    @abstractmethod
    def clone(self) -> 'BaseAgent':
        """Create a deep copy of the agent."""
        pass
    
    def reset_fitness(self):
        """Reset fitness metrics for new evaluation."""
        self.fitness = 0.0
        self.episode_rewards = []
    
    def add_episode_reward(self, reward: float):
        """
        Add reward from an episode for fitness calculation.
        
        Args:
            reward: Total reward from one episode
        """
        self.episode_rewards.append(reward)
    
    def calculate_fitness(self) -> float:
        """
        Calculate fitness from collected episode rewards.
        Uses mean reward across episodes.
        
        Returns:
            Fitness value
        """
        if not self.episode_rewards:
            self.fitness = 0.0
        else:
            self.fitness = np.mean(self.episode_rewards)
        return self.fitness
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, fitness={self.fitness:.2f})"
