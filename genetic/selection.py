"""
Selection operators for genetic algorithms.
Implements various selection mechanisms for parent selection and population management.
"""

import numpy as np
from typing import List
from agents.policy_agent import PolicyAgent


class SelectionOperator:
    """Base class for selection operators."""
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select agents from population based on fitness.
        
        Args:
            population: List of agents to select from
            num_selections: Number of agents to select
            
        Returns:
            List of selected agents
        """
        raise NotImplementedError


class TournamentSelection(SelectionOperator):
    """
    Tournament selection: randomly sample k agents, select best.
    Simple, efficient, pressure-tunable via tournament size.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of agents in each tournament
                           Larger = stronger selection pressure
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select agents via tournament selection.
        
        Args:
            population: List of agents to select from
            num_selections: Number of agents to select
            
        Returns:
            List of selected agents
        """
        selected = []
        for _ in range(num_selections):
            # Random tournament
            tournament_indices = np.random.choice(len(population), self.tournament_size)
            tournament = [population[i] for i in tournament_indices]
            
            # Select best in tournament
            winner = max(tournament, key=lambda a: a.fitness)
            selected.append(winner)
        
        return selected


class RoulettWheelSelection(SelectionOperator):
    """
    Roulette wheel selection: probability proportional to fitness.
    Classic selection method, fitness-based but can lose diversity.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize roulette wheel selection.
        
        Args:
            temperature: Controls selection sharpness
                        High T = uniform, Low T = sharp
        """
        self.temperature = temperature
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select agents via roulette wheel.
        
        Args:
            population: List of agents to select from
            num_selections: Number of agents to select
            
        Returns:
            List of selected agents
        """
        # Get fitness values
        fitnesses = np.array([a.fitness for a in population])
        
        # Shift to positive and apply temperature
        min_fitness = fitnesses.min()
        shifted_fitnesses = fitnesses - min_fitness + 1e-8
        probabilities = np.exp(shifted_fitnesses / self.temperature)
        probabilities = probabilities / probabilities.sum()
        
        # Select based on probabilities
        indices = np.random.choice(
            len(population),
            size=num_selections,
            p=probabilities,
            replace=True
        )
        
        return [population[i] for i in indices]


class RankSelection(SelectionOperator):
    """
    Rank selection: rank agents by fitness, select by rank.
    Less sensitive to outliers, maintains diversity better.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Exponential pressure (1.0 = linear ranking)
                               Higher = stronger selection
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select agents via rank selection.
        
        Args:
            population: List of agents to select from
            num_selections: Number of agents to select
            
        Returns:
            List of selected agents
        """
        # Sort by fitness and assign ranks
        sorted_agents = sorted(population, key=lambda a: a.fitness, reverse=True)
        ranks = np.arange(len(population), 0, -1)
        
        # Apply selection pressure
        probabilities = (ranks / ranks.sum()) ** self.selection_pressure
        probabilities = probabilities / probabilities.sum()
        
        # Select based on rank probabilities
        indices = np.random.choice(
            len(sorted_agents),
            size=num_selections,
            p=probabilities,
            replace=True
        )
        
        return [sorted_agents[i] for i in indices]


class BoltzmannSelection(SelectionOperator):
    """
    Boltzmann selection: fitness-proportional with temperature schedule.
    Temperature decreases over time for progressive pressure increase.
    """
    
    def __init__(self, initial_temperature: float = 5.0, cooling_rate: float = 0.99):
        """
        Initialize Boltzmann selection.
        
        Args:
            initial_temperature: Starting temperature
            cooling_rate: Multiplicative cooling (T *= cooling_rate each call)
        """
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select agents via Boltzmann selection.
        
        Args:
            population: List of agents to select from
            num_selections: Number of agents to select
            
        Returns:
            List of selected agents
        """
        fitnesses = np.array([a.fitness for a in population])
        min_fitness = fitnesses.min()
        shifted_fitnesses = fitnesses - min_fitness + 1e-8
        
        # Boltzmann probability
        probabilities = np.exp(shifted_fitnesses / self.temperature)
        probabilities = probabilities / probabilities.sum()
        
        # Select
        indices = np.random.choice(
            len(population),
            size=num_selections,
            p=probabilities,
            replace=True
        )
        
        # Cool temperature
        self.temperature *= self.cooling_rate
        
        return [population[i] for i in indices]
    
    def reset_temperature(self):
        """Reset temperature to initial value."""
        self.temperature = self.initial_temperature


class EliteSelection(SelectionOperator):
    """
    Elite selection: select top k agents deterministically.
    Used for elitism (preserving best solutions each generation).
    """
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select top agents by fitness.
        
        Args:
            population: List of agents to select from
            num_selections: Number of top agents to select
            
        Returns:
            List of best agents
        """
        sorted_agents = sorted(population, key=lambda a: a.fitness, reverse=True)
        return sorted_agents[:min(num_selections, len(sorted_agents))]


class RandomSelection(SelectionOperator):
    """
    Random selection: select uniformly at random.
    Baseline for comparison, preserves genetic diversity.
    """
    
    def select(self, population: List[PolicyAgent], num_selections: int = 1) -> List[PolicyAgent]:
        """
        Select agents uniformly at random.
        
        Args:
            population: List of agents to select from
            num_selections: Number of agents to select
            
        Returns:
            List of randomly selected agents
        """
        indices = np.random.choice(len(population), size=num_selections, replace=True)
        return [population[i] for i in indices]
