"""
Population management for genetic algorithms.
Handles agent lifecycle, generation tracking, and evolution statistics.
"""

import numpy as np
from typing import List, Dict, Tuple
from agents.policy_agent import PolicyAgent
from genetic.mutation import MutationOperator, GaussianMutation
from genetic.crossover import CrossoverOperator, UniformCrossover
from genetic.selection import SelectionOperator, TournamentSelection, EliteSelection


class Population:
    """
    Population manager for evolutionary algorithms.
    Manages population dynamics, reproduction, and statistics tracking.
    """
    
    def __init__(self, population_size: int, input_size: int, hidden_sizes: List[int],
                 output_size: int, device: str = None):
        """
        Initialize population with random agents.
        
        Args:
            population_size: Number of agents per generation
            input_size: Neural network input size
            hidden_sizes: Neural network hidden layer sizes
            output_size: Neural network output size
            device: 'cuda' or 'cpu'
        """
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device
        
        # Initialize population
        self.agents: List[PolicyAgent] = []
        self.generation = 0
        
        for i in range(population_size):
            agent = PolicyAgent(input_size, hidden_sizes, output_size,
                              agent_id=i, device=device)
            agent.birth_generation = self.generation
            self.agents.append(agent)
        
        # Statistics tracking
        self.generation_stats: List[Dict] = []
        self.best_agent: PolicyAgent = None
        self.best_fitness = -np.inf
        
        # Genetic operators (can be customized)
        self.mutation_op: MutationOperator = GaussianMutation()
        self.crossover_op: CrossoverOperator = UniformCrossover()
        self.selection_op: SelectionOperator = TournamentSelection(tournament_size=3)
        self.elite_selection: SelectionOperator = EliteSelection()
    
    def set_mutation_operator(self, mutation_op: MutationOperator):
        """Set custom mutation operator."""
        self.mutation_op = mutation_op
    
    def set_crossover_operator(self, crossover_op: CrossoverOperator):
        """Set custom crossover operator."""
        self.crossover_op = crossover_op
    
    def set_selection_operator(self, selection_op: SelectionOperator):
        """Set custom selection operator."""
        self.selection_op = selection_op
    
    def reset_fitness(self):
        """Reset fitness for all agents (before evaluation)."""
        for agent in self.agents:
            agent.reset_fitness()
    
    def evaluate_fitness(self):
        """
        Evaluate population fitness and track statistics.
        Assumes agents already have recorded fitness via add_episode_reward().
        """
        for agent in self.agents:
            agent.calculate_fitness()
        
        # Track best agent globally
        best_in_generation = max(self.agents, key=lambda a: a.fitness)
        if best_in_generation.fitness > self.best_fitness:
            self.best_fitness = best_in_generation.fitness
            self.best_agent = best_in_generation.clone()
        
        # Record generation statistics
        fitnesses = np.array([a.fitness for a in self.agents])
        stats = {
            'generation': self.generation,
            'best_fitness': fitnesses.max(),
            'worst_fitness': fitnesses.min(),
            'mean_fitness': fitnesses.mean(),
            'std_fitness': fitnesses.std(),
            'global_best': self.best_fitness
        }
        self.generation_stats.append(stats)
    
    def evolve(self, elite_size: int = 5) -> List[PolicyAgent]:
        """
        Create next generation through selection, crossover, and mutation.
        
        Args:
            elite_size: Number of best agents to preserve (elitism)
            
        Returns:
            List of new agents for next generation
        """
        new_population = []
        
        # Elitism: preserve best agents
        elite_agents = self.elite_selection.select(self.agents, elite_size)
        for agent in elite_agents:
            new_agent = agent.clone()
            new_agent.birth_generation = self.generation + 1
            new_population.append(new_agent)
        
        # Fill rest of population through reproduction
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.selection_op.select(self.agents, 1)[0]
            parent2 = self.selection_op.select(self.agents, 1)[0]
            
            # Crossover
            child1, child2 = self.crossover_op.crossover(parent1, parent2)
            child1.birth_generation = self.generation + 1
            child2.birth_generation = self.generation + 1
            
            # Mutation
            if np.random.rand() < 0.8:  # 80% get mutated
                child1 = self.mutation_op.mutate(child1)
            if np.random.rand() < 0.8:
                child2 = self.mutation_op.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Replace population
        self.agents = new_population
        self.generation += 1
        
        return self.agents
    
    def step(self, elite_size: int = 5) -> Dict:
        """
        Complete one generation cycle: evaluate → evolve → return stats.
        
        Args:
            elite_size: Number of elite agents to preserve
            
        Returns:
            Dictionary of generation statistics
        """
        self.evaluate_fitness()
        self.evolve(elite_size)
        return self.generation_stats[-1]
    
    def get_statistics(self) -> Dict:
        """
        Get evolution statistics so far.
        
        Returns:
            Dictionary with generation histories and best agent
        """
        return {
            'generation': self.generation,
            'best_fitness_history': [s['best_fitness'] for s in self.generation_stats],
            'mean_fitness_history': [s['mean_fitness'] for s in self.generation_stats],
            'std_fitness_history': [s['std_fitness'] for s in self.generation_stats],
            'global_best_fitness': self.best_fitness,
            'best_agent': self.best_agent
        }
    
    def print_progress(self):
        """Print current generation progress."""
        if not self.generation_stats:
            return
        
        stats = self.generation_stats[-1]
        print(f"Gen {stats['generation']:4d} | "
              f"Best: {stats['best_fitness']:8.4f} | "
              f"Mean: {stats['mean_fitness']:8.4f} ± {stats['std_fitness']:6.4f} | "
              f"Worst: {stats['worst_fitness']:8.4f} | "
              f"Global Best: {stats['global_best']:8.4f}")
    
    def sort_by_fitness(self) -> List[PolicyAgent]:
        """
        Sort population by fitness (descending).
        
        Returns:
            Sorted list of agents
        """
        return sorted(self.agents, key=lambda a: a.fitness, reverse=True)
    
    def __repr__(self):
        return (f"Population(size={self.population_size}, "
                f"generation={self.generation}, "
                f"best_fitness={self.best_fitness:.4f})")
