"""
Crossover operators for genetic algorithms.
Implements various recombination strategies for neural network weights.
"""

import numpy as np
from agents.policy_agent import PolicyAgent


class CrossoverOperator:
    """Base class for crossover operators."""
    
    def crossover(self, parent1: PolicyAgent, parent2: PolicyAgent) -> tuple:
        """
        Create offspring by combining parent weights.
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            Tuple of two offspring agents (child1, child2)
        """
        raise NotImplementedError


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover: each weight is randomly selected from one parent.
    Maintains genetic diversity, good for heterogeneous populations.
    """
    
    def __init__(self, crossover_rate: float = 0.5):
        """
        Initialize uniform crossover.
        
        Args:
            crossover_rate: Probability each weight comes from parent2 (vs parent1)
        """
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: PolicyAgent, parent2: PolicyAgent) -> tuple:
        """
        Create offspring via uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Create crossover mask
        crossover_mask = np.random.rand(len(weights1)) < self.crossover_rate
        
        # Child1: mostly parent1, some from parent2
        child1_weights = weights1.copy()
        child1_weights[crossover_mask] = weights2[crossover_mask]
        child1.set_weights(child1_weights)
        
        # Child2: mostly parent2, some from parent1
        child2_weights = weights2.copy()
        child2_weights[crossover_mask] = weights1[crossover_mask]
        child2.set_weights(child2_weights)
        
        return child1, child2


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover: split weights at one point, swap segments.
    Maintains weight correlations better than uniform crossover.
    """
    
    def crossover(self, parent1: PolicyAgent, parent2: PolicyAgent) -> tuple:
        """
        Create offspring via single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Random crossover point
        crossover_point = np.random.randint(1, len(weights1))
        
        # Create offspring
        child1_weights = np.concatenate([weights1[:crossover_point], weights2[crossover_point:]])
        child2_weights = np.concatenate([weights2[:crossover_point], weights1[crossover_point:]])
        
        child1.set_weights(child1_weights)
        child2.set_weights(child2_weights)
        
        return child1, child2


class TwoPointCrossover(CrossoverOperator):
    """
    Two-point crossover: split at two points, swap middle segment.
    Balance between genetic diversity and structure preservation.
    """
    
    def crossover(self, parent1: PolicyAgent, parent2: PolicyAgent) -> tuple:
        """
        Create offspring via two-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Two random crossover points
        point1, point2 = sorted(np.random.choice(len(weights1), 2, replace=False))
        
        # Create offspring by swapping middle segment
        child1_weights = np.concatenate([
            weights1[:point1],
            weights2[point1:point2],
            weights1[point2:]
        ])
        child2_weights = np.concatenate([
            weights2[:point1],
            weights1[point1:point2],
            weights2[point2:]
        ])
        
        child1.set_weights(child1_weights)
        child2.set_weights(child2_weights)
        
        return child1, child2


class BlendCrossover(CrossoverOperator):
    """
    Blend crossover (BLX-alpha): interpolate between parent weights.
    Creates smooth intermediate solutions, useful for continuous optimization.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize blend crossover.
        
        Args:
            alpha: Blending factor. 0.5 = average, >0.5 = extrapolation
        """
        self.alpha = alpha
    
    def crossover(self, parent1: PolicyAgent, parent2: PolicyAgent) -> tuple:
        """
        Create offspring via blend crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Blend with random interpolation
        gamma = np.random.randn(len(weights1)) * self.alpha
        
        child1_weights = weights1 + gamma * (weights2 - weights1)
        child2_weights = weights2 - gamma * (weights2 - weights1)
        
        child1.set_weights(child1_weights)
        child2.set_weights(child2_weights)
        
        return child1, child2


class IntermediateCrossover(CrossoverOperator):
    """
    Intermediate crossover: each weight is randomly sampled from interval between parents.
    Keeps offspring close to parents while exploring intermediate space.
    """
    
    def crossover(self, parent1: PolicyAgent, parent2: PolicyAgent) -> tuple:
        """
        Create offspring via intermediate crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Sample from range [min(w1,w2), max(w1,w2)]
        min_weights = np.minimum(weights1, weights2)
        max_weights = np.maximum(weights1, weights2)
        
        child1_weights = np.random.uniform(min_weights, max_weights)
        child2_weights = np.random.uniform(min_weights, max_weights)
        
        child1.set_weights(child1_weights)
        child2.set_weights(child2_weights)
        
        return child1, child2
