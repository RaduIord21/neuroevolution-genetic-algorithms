"""
Main training script for neuroevolution of agent policies.
Orchestrates population initialization, fitness evaluation, and evolution.
"""

import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents.policy_agent import PolicyAgent
from genetic.population import Population
from genetic.mutation import GaussianMutation, AdaptiveGaussianMutation
from genetic.crossover import UniformCrossover
from genetic.selection import TournamentSelection
from evaluation.fitness import FitnessEvaluator, AdaptiveFitnessEvaluator
from utils.config import (
    POPULATION_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, ELITE_SIZE,
    ENV_NAME, MAX_STEPS, NUM_EPISODES, INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE
)


class NeuroevolutionTrainer:
    """
    Complete neuroevolution training pipeline.
    Manages population, fitness evaluation, and evolution.
    """
    
    def __init__(self, population_size=POPULATION_SIZE, generations=GENERATIONS,
                 env_name=ENV_NAME, max_steps=MAX_STEPS, num_episodes=NUM_EPISODES,
                 input_size=INPUT_SIZE, hidden_sizes=HIDDEN_SIZES, output_size=OUTPUT_SIZE,
                 elite_size=ELITE_SIZE, device=None, save_dir="results", visualize=False,
                 visualize_interval=20):
        """
        Initialize trainer.
        
        Args:
            population_size: Size of population per generation
            generations: Number of generations to evolve
            env_name: Environment name
            max_steps: Max steps per episode
            num_episodes: Episodes per fitness evaluation
            input_size: NN input size
            hidden_sizes: NN hidden layer sizes
            output_size: NN output size
            elite_size: Number of elite agents to preserve
            device: 'cuda' or 'cpu'
            save_dir: Directory to save results
            visualize: Whether to visualize best agent during training
            visualize_interval: How often to visualize (every N generations)
        """
        self.population_size = population_size
        self.generations = generations
        self.env_name = env_name
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.elite_size = elite_size
        self.device = device
        self.visualize = visualize
        self.visualize_interval = visualize_interval
        
        # Create results directory
        self.save_dir = Path(save_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("NEUROEVOLUTION TRAINING")
        print(f"{'='*70}")
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        print(f"Environment: {env_name}")
        print(f"Max steps/episode: {max_steps}")
        print(f"Episodes/evaluation: {num_episodes}")
        print(f"Network: {input_size} -> {hidden_sizes} -> {output_size}")
        print(f"Device: {device if device else 'auto-detect'}")
        if visualize:
            print(f"Visualization: enabled (every {visualize_interval} generations)")
        else:
            print(f"Visualization: disabled")
        print(f"Save directory: {self.save_dir}")
        print(f"{'='*70}\n")
        
        # Initialize components
        self.population = Population(
            population_size=population_size,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            device=device
        )
        
        # Set up genetic operators
        self.population.set_mutation_operator(
            AdaptiveGaussianMutation(
                initial_mutation_rate=0.15,
                initial_mutation_std=0.2,
                final_mutation_rate=0.05,
                final_mutation_std=0.05,
                generations=generations
            )
        )
        self.population.set_crossover_operator(UniformCrossover(crossover_rate=0.5))
        self.population.set_selection_operator(TournamentSelection(tournament_size=3))
        
        # Fitness evaluator
        self.evaluator = FitnessEvaluator(env_name, max_steps=max_steps)
        
        # Visualization evaluator (with rendering)
        if visualize:
            self.vis_evaluator = FitnessEvaluator(env_name, max_steps=max_steps, render_mode='human')
        else:
            self.vis_evaluator = None
        
        # Training history
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'worst_fitness': [],
            'generation': []
        }
    
    def visualize_best_agent(self, num_episodes=2):
        """Visualize the best agent's behavior."""
        if not self.vis_evaluator or not self.population.best_agent:
            return
        
        print(f"\n{'='*70}")
        print(f"Visualizing best agent (fitness: {self.population.best_fitness:.4f})")
        print(f"{'='*70}")
        
        for episode in range(num_episodes):
            reward = self.vis_evaluator.evaluate_single_episode(
                self.population.best_agent,
                render=True
            )
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {reward:.2f}")
        
        print(f"{'='*70}\n")
    
    def train(self):
        """Run complete training loop."""
        try:
            for gen in tqdm(range(self.generations)):
                # Reset fitness for all agents
                self.population.reset_fitness()
                
                # Evaluate population
                print(f"Generation {gen+1}/{self.generations} - Evaluating...", end=" ")
                results = self.evaluator.evaluate_population(
                    self.population.agents,
                    num_episodes=self.num_episodes
                )
                print("Complete")
                
                # Evaluate and print progress
                self.population.evaluate_fitness()
                stats = self.population.generation_stats[-1]
                
                # Record history
                self.history['generation'].append(stats['generation'])
                self.history['best_fitness'].append(stats['best_fitness'])
                self.history['mean_fitness'].append(stats['mean_fitness'])
                self.history['std_fitness'].append(stats['std_fitness'])
                self.history['worst_fitness'].append(stats['worst_fitness'])
                
                # Print progress
                self.population.print_progress()
                
                # Evolve population
                self.population.evolve(elite_size=self.elite_size)
                
                # Update mutation operator generation
                if hasattr(self.population.mutation_op, 'set_generation'):
                    self.population.mutation_op.set_generation(gen + 1)
                
                # Save checkpoint every 10 generations
                if (gen + 1) % 10 == 0:
                    self.save_checkpoint(gen + 1)
                    self.plot_progress()
                
                # Visualize best agent at configured interval
                if self.visualize and (gen + 1) % self.visualize_interval == 0:
                    self.visualize_best_agent(num_episodes=1)
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_results()
            self.cleanup()
            return
        
        except Exception as e:
            print(f"\n\nTraining failed with error:")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.save_results()
            self.cleanup()
            return
        
        # Finalize
        self.save_results()
        self.plot_progress()
        self.print_summary()
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.evaluator.close()
        if self.vis_evaluator:
            self.vis_evaluator.close()
    
    def save_checkpoint(self, generation):
        """Save population checkpoint."""
        checkpoint = {
            'generation': generation,
            'best_agent': self.population.best_agent,
            'best_fitness': self.population.best_fitness,
            'population_weights': [a.get_weights() for a in self.population.agents],
            'history': self.history
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_gen_{generation:04d}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Checkpoint saved: {checkpoint_path.name}")
    
    def save_results(self):
        """Save final results and metadata."""
        # Save best agent
        if self.population.best_agent:
            best_agent_path = self.save_dir / "best_agent.pkl"
            with open(best_agent_path, 'wb') as f:
                pickle.dump(self.population.best_agent, f)
            
            best_weights_path = self.save_dir / "best_agent_weights.npy"
            np.save(best_weights_path, self.population.best_agent.get_weights())
        
        # Save history
        history_path = self.save_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save training metadata
        metadata = {
            'generations_completed': self.population.generation,
            'population_size': self.population_size,
            'environment': self.env_name,
            'best_fitness': float(self.population.best_fitness),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'timestamp': datetime.now().isoformat()
        }
        metadata_path = self.save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to: {self.save_dir}")
    
    def plot_progress(self):
        """Plot and save training progress."""
        if not self.history['generation']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Best fitness
        ax = axes[0, 0]
        ax.plot(self.history['generation'], self.history['best_fitness'], 'g-o', label='Best')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Best Fitness Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mean fitness
        ax = axes[0, 1]
        mean = self.history['mean_fitness']
        std = self.history['std_fitness']
        ax.plot(self.history['generation'], mean, 'b-o', label='Mean')
        ax.fill_between(self.history['generation'], 
                        np.array(mean) - np.array(std),
                        np.array(mean) + np.array(std),
                        alpha=0.3, label='±Std')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Mean Fitness with Std Dev')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fitness range
        ax = axes[1, 0]
        ax.fill_between(self.history['generation'],
                        self.history['worst_fitness'],
                        self.history['best_fitness'],
                        alpha=0.3, label='Min-Max Range')
        ax.plot(self.history['generation'], self.history['mean_fitness'], 'r-', label='Mean')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Population Fitness Range')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convergence (gap between best and worst)
        ax = axes[1, 1]
        gap = np.array(self.history['best_fitness']) - np.array(self.history['worst_fitness'])
        ax.semilogy(self.history['generation'], gap, 'purple', marker='o')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best - Worst Fitness (log scale)')
        ax.set_title('Population Convergence')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        print(f"Plot saved: {plot_path.name}")
    
    def print_summary(self):
        """Print training summary."""
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Generations completed: {self.population.generation}")
        print(f"Best fitness achieved: {self.population.best_fitness:.4f}")
        print(f"Final mean fitness: {self.history['mean_fitness'][-1]:.4f}")
        print(f"Final std fitness: {self.history['std_fitness'][-1]:.4f}")
        
        if len(self.history['best_fitness']) > 1:
            improvement = self.history['best_fitness'][-1] - self.history['best_fitness'][0]
            print(f"Fitness improvement: {improvement:.4f}")
        
        print(f"\nResults saved to: {self.save_dir}")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Neuroevolution training")
    parser.add_argument('--generations', type=int, default=GENERATIONS,
                       help='Number of generations')
    parser.add_argument('--population', type=int, default=POPULATION_SIZE,
                       help='Population size')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                       help='Episodes per fitness evaluation')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--env', type=str, default=ENV_NAME,
                       help='Gymnasium environment name')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS,
                       help='Max steps per episode')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize best agent during training')
    parser.add_argument('--visualize-interval', type=int, default=20,
                       help='How often to visualize (every N generations)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = NeuroevolutionTrainer(
        population_size=args.population,
        generations=args.generations,
        env_name=args.env,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        device=args.device,
        save_dir=args.save_dir,
        visualize=args.visualize,
        visualize_interval=args.visualize_interval
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
