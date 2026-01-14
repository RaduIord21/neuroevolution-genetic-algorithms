"""
Policy agent implementation using a PyTorch neural network.
"""

import numpy as np
import torch
from agents.base_agent import BaseAgent
from agents.neural_network import NeuralNetwork


class PolicyAgent(BaseAgent):
    """
    Agent with a PyTorch-based feedforward neural network policy.
    The network takes sensor observations and outputs action values.
    Supports GPU acceleration via CUDA.
    """
    
    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, 
                 agent_id: int = None, device: str = None):
        """
        Initialize policy agent with neural network.
        
        Args:
            input_size: Number of sensor inputs
            hidden_sizes: List of hidden layer sizes
            output_size: Number of possible actions
            agent_id: Unique identifier
            device: 'cuda' or 'cpu'. If None, auto-detect
        """
        super().__init__(agent_id)
        self.network = NeuralNetwork(input_size, hidden_sizes, output_size, device=device)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = self.network.device
    
    def get_action(self, observation: np.ndarray) -> int:
        """
        Select action based on current observation.
        Uses argmax of network output (greedy policy).
        
        Args:
            observation: Current environment observation
            
        Returns:
            Action index (0 to output_size-1)
        """
        # Ensure observation is a torch tensor on correct device
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        elif not observation.is_cuda and self.device.type == 'cuda':
            observation = observation.to(self.device)
        
        # Forward pass through network
        with torch.no_grad():  # No gradient computation during inference
            action_values = self.network(observation)
        
        # Select action with highest value (greedy)
        action = torch.argmax(action_values).cpu().item()
        
        return int(action)
    
    def get_action_values(self, observation: np.ndarray) -> np.ndarray:
        """
        Get raw action values from network (useful for analysis).
        
        Args:
            observation: Current environment observation
            
        Returns:
            Raw output values from network as numpy array
        """
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).to(self.device)
        elif not observation.is_cuda and self.device.type == 'cuda':
            observation = observation.to(self.device)
        
        with torch.no_grad():
            action_values = self.network(observation)
        
        return action_values.cpu().numpy()
    
    def get_weights(self) -> np.ndarray:
        """Get network weights as flat array."""
        return self.network.get_weights()
    
    def set_weights(self, weights: np.ndarray):
        """Set network weights from flat array."""
        self.network.set_weights(weights)
    
    def get_num_parameters(self) -> int:
        """Get total number of network parameters."""
        return self.network.get_num_parameters()
    
    def clone(self) -> 'PolicyAgent':
        """Create a deep copy of the agent."""
        clone = PolicyAgent(self.input_size, self.hidden_sizes, self.output_size,
                            device=self.device.type)
        clone.set_weights(self.get_weights())
        clone.agent_id = self.agent_id
        clone.birth_generation = self.birth_generation
        return clone
    
    def to_device(self, device: str):
        """Move agent's network to specified device ('cuda' or 'cpu')."""
        self.network.to_device(device)
        self.device = self.network.device
    
    def __repr__(self):
        return f"PolicyAgent({self.input_size}-{self.hidden_sizes}-{self.output_size}, id={self.agent_id}, fitness={self.fitness:.2f}, device={self.device})"
