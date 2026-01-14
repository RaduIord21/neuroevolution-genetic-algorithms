"""
Neural network module for policy representation.
Implements a feedforward neural network using PyTorch with CUDA support.
Weights are accessible as flat arrays for genetic operations.
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np


class NeuralNetwork(nn.Module):
    """
    Feedforward neural network with arbitrary hidden layer sizes.
    Built with PyTorch for GPU acceleration via CUDA.
    Parameters can be extracted/set as flat arrays for genetic algorithms.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 device: str = None, seed: int = None):
        """
        Initialize the neural network with specified architecture.
        
        Args:
            input_size: Number of input neurons (sensor dimensions)
            hidden_sizes: List of hidden layer sizes, e.g., [16, 16]
            output_size: Number of output neurons (action space)
            device: 'cuda' or 'cpu'. If None, auto-detect CUDA availability
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layer sizes: [input, hidden1, hidden2, ..., output]
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input observation (tensor of shape (input_size,))
            
        Returns:
            Output activation (tensor of shape (output_size,))
        """
        # Ensure input is on correct device
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        elif not x.is_cuda and self.device.type == 'cuda':
            x = x.to(self.device)
        
        activation = x
        
        # Forward through all layers except output
        for i in range(len(self.layers) - 1):
            activation = torch.tanh(self.layers[i](activation))
        
        # Output layer (linear)
        output = self.layers[-1](activation)
        
        return output
    
    def get_weights(self) -> np.ndarray:
        """Get all weights and biases as a flat numpy array."""
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_weights(self, weights: np.ndarray):
        """
        Set weights from a flat numpy array.
        
        Args:
            weights: Flat array of all weights and biases
        """
        if isinstance(weights, np.ndarray):
            weights = torch.FloatTensor(weights)
        
        idx = 0
        for param in self.parameters():
            param_size = param.data.numel()
            param.data = weights[idx:idx + param_size].reshape(param.data.shape).to(self.device)
            idx += param_size
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def clone(self) -> 'NeuralNetwork':
        """Create a deep copy of the network."""
        clone = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size,
                              device=self.device.type)
        clone.set_weights(self.get_weights())
        return clone
    
    def to_device(self, device: str):
        """Move network to specified device ('cuda' or 'cpu')."""
        self.device = torch.device(device)
        self.to(self.device)
    
    def __repr__(self):
        return f"NeuralNetwork({self.input_size} -> {self.hidden_sizes} -> {self.output_size}, device={self.device})"
