"""
Mutation operators for genetic algorithms.
Implements various mutation strategies for neural network weights.
"""

import numpy as np
from agents.policy_agent import PolicyAgent


class MutationOperator:
    """Base class for mutation operators."""
    
    def mutate(self, agent: PolicyAgent) -> PolicyAgent:
        """
        Mutate an agent's weights.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            New agent with mutated weights
        """
        raise NotImplementedError


class GaussianMutation(MutationOperator):
    """
    Gaussian mutation: add random noise to weights.
    Standard mutation operator for neuroevolution.
    """
    
    def __init__(self, mutation_rate: float = 0.1, mutation_std: float = 0.1):
        """
        Initialize Gaussian mutation operator.
        
        Args:
            mutation_rate: Probability each weight is mutated (0-1)
            mutation_std: Standard deviation of Gaussian noise
        """
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
    
    def mutate(self, agent: PolicyAgent) -> PolicyAgent:
        """
        Apply Gaussian mutation to agent.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            Cloned agent with mutated weights
        """
        mutant = agent.clone()
        weights = mutant.get_weights()
        
        # Create mutation mask
        mutation_mask = np.random.rand(len(weights)) < self.mutation_rate
        
        # Apply Gaussian noise to selected weights
        if mutation_mask.any():
            noise = np.random.randn(mutation_mask.sum()) * self.mutation_std
            weights[mutation_mask] += noise
        
        mutant.set_weights(weights)
        return mutant


class AdaptiveGaussianMutation(MutationOperator):
    """
    Adaptive Gaussian mutation: mutation strength decreases over generations.
    Helps balance exploration early and exploitation late in evolution.
    """
    
    def __init__(self, initial_mutation_rate: float = 0.2, 
                 initial_mutation_std: float = 0.3,
                 final_mutation_rate: float = 0.05,
                 final_mutation_std: float = 0.05,
                 generations: int = 100):
        """
        Initialize adaptive Gaussian mutation.
        
        Args:
            initial_mutation_rate: Starting mutation rate
            initial_mutation_std: Starting mutation std dev
            final_mutation_rate: Ending mutation rate
            final_mutation_std: Ending mutation std dev
            generations: Total generations for scheduling
        """
        self.initial_mutation_rate = initial_mutation_rate
        self.initial_mutation_std = initial_mutation_std
        self.final_mutation_rate = final_mutation_rate
        self.final_mutation_std = final_mutation_std
        self.generations = generations
        self.current_generation = 0
    
    def set_generation(self, generation: int):
        """Update current generation for adaptive scheduling."""
        self.current_generation = min(generation, self.generations)
    
    def _get_adaptive_value(self, initial: float, final: float) -> float:
        """Linearly interpolate between initial and final values."""
        progress = self.current_generation / max(self.generations, 1)
        return initial + (final - initial) * progress
    
    def mutate(self, agent: PolicyAgent) -> PolicyAgent:
        """
        Apply adaptive Gaussian mutation.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            Cloned agent with mutated weights
        """
        mutation_rate = self._get_adaptive_value(
            self.initial_mutation_rate, 
            self.final_mutation_rate
        )
        mutation_std = self._get_adaptive_value(
            self.initial_mutation_std,
            self.final_mutation_std
        )
        
        mutant = agent.clone()
        weights = mutant.get_weights()
        
        mutation_mask = np.random.rand(len(weights)) < mutation_rate
        if mutation_mask.any():
            noise = np.random.randn(mutation_mask.sum()) * mutation_std
            weights[mutation_mask] += noise
        
        mutant.set_weights(weights)
        return mutant


class UniformMutation(MutationOperator):
    """
    Uniform mutation: replace weights with random values from uniform distribution.
    More exploratory than Gaussian mutation.
    """
    
    def __init__(self, mutation_rate: float = 0.05, weight_range: float = 1.0):
        """
        Initialize uniform mutation.
        
        Args:
            mutation_rate: Probability each weight is mutated
            weight_range: Range for uniform distribution [-weight_range, weight_range]
        """
        self.mutation_rate = mutation_rate
        self.weight_range = weight_range
    
    def mutate(self, agent: PolicyAgent) -> PolicyAgent:
        """
        Apply uniform mutation to agent.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            Cloned agent with mutated weights
        """
        mutant = agent.clone()
        weights = mutant.get_weights()
        
        mutation_mask = np.random.rand(len(weights)) < self.mutation_rate
        if mutation_mask.any():
            random_values = np.random.uniform(
                -self.weight_range, 
                self.weight_range, 
                mutation_mask.sum()
            )
            weights[mutation_mask] = random_values
        
        mutant.set_weights(weights)
        return mutant


class PolynomialMutation(MutationOperator):
    """
    Polynomial mutation: bounded mutation with polynomial distribution.
    Used in CMA-ES-style algorithms, creates more controlled mutations.
    """
    
    def __init__(self, mutation_rate: float = 0.1, eta: float = 20.0,
                 weight_min: float = -5.0, weight_max: float = 5.0):
        """
        Initialize polynomial mutation.
        
        Args:
            mutation_rate: Probability each weight is mutated
            eta: Distribution index (higher = smaller mutations)
            weight_min: Minimum weight bound
            weight_max: Maximum weight bound
        """
        self.mutation_rate = mutation_rate
        self.eta = eta
        self.weight_min = weight_min
        self.weight_max = weight_max
    
    def mutate(self, agent: PolicyAgent) -> PolicyAgent:
        """
        Apply polynomial mutation to agent.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            Cloned agent with mutated weights
        """
        mutant = agent.clone()
        weights = mutant.get_weights()
        
        mutation_mask = np.random.rand(len(weights)) < self.mutation_rate
        
        for idx in np.where(mutation_mask)[0]:
            u = np.random.rand()
            if u <= 0.5:
                delta = (2 * u) ** (1.0 / (self.eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (self.eta + 1))
            
            weights[idx] = np.clip(weights[idx] + delta, self.weight_min, self.weight_max)
        
        mutant.set_weights(weights)
        return mutant
