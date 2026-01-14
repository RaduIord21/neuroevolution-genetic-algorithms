"""
Fitness evaluation module for agents.
Runs agents in environment and computes fitness metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from agents.policy_agent import PolicyAgent
from environment.env_wrapper import EnvironmentWrapper


class FitnessEvaluator:
    """
    Evaluates agent fitness by running episodes in environment.
    Supports various fitness aggregation methods.
    """
    
    def __init__(self, env_name: str = "LunarLander-v3", max_steps: int = 1000,
                 render_mode: str = None):
        """
        Initialize fitness evaluator.
        
        Args:
            env_name: Gymnasium environment name
            max_steps: Maximum steps per episode
            render_mode: 'human' for rendering, None for headless
        """
        self.env = EnvironmentWrapper(env_name, render_mode=render_mode)
        self.max_steps = max_steps
        self.env_name = env_name
        self.evaluation_count = 0
        
    def evaluate_single_episode(self, agent: PolicyAgent, render: bool = False) -> float:
        """
        Run a single episode and return total reward.
        
        Args:
            agent: Agent to evaluate
            render: Whether to render the episode
            
        Returns:
            Total reward for the episode
        """
        observation = self.env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Agent selects action
            action = agent.get_action(observation)
            
            # Environment step
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
        
        return total_reward
    
    def evaluate_episodes(self, agent: PolicyAgent, num_episodes: int = 3,
                         render: bool = False) -> Dict:
        """
        Run multiple episodes and aggregate statistics.
        
        Args:
            agent: Agent to evaluate
            num_episodes: Number of episodes to run
            render: Whether to render
            
        Returns:
            Dictionary with episode statistics
        """
        episode_rewards = []
        
        for _ in range(num_episodes):
            reward = self.evaluate_single_episode(agent, render=render)
            episode_rewards.append(reward)
        
        self.evaluation_count += 1
        
        return {
            'rewards': episode_rewards,
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'max': np.max(episode_rewards),
            'min': np.min(episode_rewards),
            'sum': np.sum(episode_rewards)
        }
    
    def evaluate_population(self, population: List[PolicyAgent], num_episodes: int = 3,
                           parallel: bool = False) -> List[Dict]:
        """
        Evaluate entire population.
        
        Args:
            population: List of agents to evaluate
            num_episodes: Episodes per agent
            parallel: If True, would use parallel evaluation (not implemented)
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for agent in population:
            agent.reset_fitness()
            eval_result = self.evaluate_episodes(agent, num_episodes=num_episodes)
            
            # Record rewards on agent
            for reward in eval_result['rewards']:
                agent.add_episode_reward(reward)
            agent.calculate_fitness()
            
            results.append({
                'agent_id': agent.agent_id,
                'fitness': agent.fitness,
                'details': eval_result
            })
        
        return results
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def __repr__(self):
        return f"FitnessEvaluator({self.env_name}, max_steps={self.max_steps}, evaluations={self.evaluation_count})"


class AdaptiveFitnessEvaluator(FitnessEvaluator):
    """
    Adaptive fitness evaluator that adjusts evaluation difficulty.
    Increases max_steps as population improves (curriculum learning).
    """
    
    def __init__(self, env_name: str = "LunarLander-v3", 
                 initial_max_steps: int = 500,
                 final_max_steps: int = 1000,
                 improvement_threshold: float = 50.0):
        """
        Initialize adaptive evaluator.
        
        Args:
            env_name: Environment name
            initial_max_steps: Starting max steps per episode
            final_max_steps: Final max steps per episode
            improvement_threshold: Fitness improvement to trigger difficulty increase
        """
        super().__init__(env_name, max_steps=initial_max_steps)
        self.initial_max_steps = initial_max_steps
        self.final_max_steps = final_max_steps
        self.improvement_threshold = improvement_threshold
        self.best_fitness_seen = -np.inf
        self.generation = 0
    
    def evaluate_population(self, population: List[PolicyAgent], num_episodes: int = 3) -> List[Dict]:
        """
        Evaluate population with adaptive difficulty.
        
        Args:
            population: List of agents
            num_episodes: Episodes per agent
            
        Returns:
            List of evaluation results
        """
        results = super().evaluate_population(population, num_episodes=num_episodes)
        
        # Check if population improved
        current_best = max(population, key=lambda a: a.fitness)
        if current_best.fitness > self.best_fitness_seen + self.improvement_threshold:
            self.best_fitness_seen = current_best.fitness
            self._increase_difficulty()
        
        self.generation += 1
        return results
    
    def _increase_difficulty(self):
        """Increase evaluation difficulty (more steps)."""
        old_steps = self.max_steps
        self.max_steps = min(
            int(self.max_steps * 1.1),  # 10% increase
            self.final_max_steps
        )
        if old_steps < self.max_steps:
            print(f"  Increasing difficulty: {old_steps} → {self.max_steps} steps")


class NoveltyBasedEvaluator(FitnessEvaluator):
    """
    Novelty-based fitness that combines reward with behavioral diversity.
    Encourages exploration of diverse strategies (NEAT-like novelty search).
    """
    
    def __init__(self, env_name: str = "LunarLander-v3", max_steps: int = 1000,
                 reward_weight: float = 0.5, novelty_weight: float = 0.5):
        """
        Initialize novelty-based evaluator.
        
        Args:
            env_name: Environment name
            max_steps: Max steps per episode
            reward_weight: Weight for reward in fitness
            novelty_weight: Weight for novelty in fitness
        """
        super().__init__(env_name, max_steps=max_steps)
        self.reward_weight = reward_weight
        self.novelty_weight = novelty_weight
        self.behavior_archive = []  # Store behavior vectors
    
    def _extract_behavior(self, agent: PolicyAgent, num_samples: int = 5) -> np.ndarray:
        """
        Extract behavior vector from agent (action distribution on random observations).
        
        Args:
            agent: Agent to characterize
            num_samples: Number of observations to sample
            
        Returns:
            Behavior vector
        """
        behaviors = []
        for _ in range(num_samples):
            obs = np.random.randn(self.env.input_size).astype(np.float32)
            action_values = agent.get_action_values(obs)
            behaviors.append(action_values)
        
        return np.concatenate(behaviors)
    
    def _calculate_novelty(self, behavior: np.ndarray, k: int = 5) -> float:
        """
        Calculate novelty as average distance to k-nearest neighbors.
        
        Args:
            behavior: Behavior vector
            k: Number of neighbors
            
        Returns:
            Novelty score
        """
        if len(self.behavior_archive) < k:
            return 1.0  # Max novelty if archive is small
        
        distances = np.array([
            np.linalg.norm(behavior - archived_behavior)
            for archived_behavior in self.behavior_archive
        ])
        
        k_nearest = np.partition(distances, min(k-1, len(distances)-1))[:k]
        novelty = np.mean(k_nearest)
        
        return novelty
    
    def evaluate_episodes(self, agent: PolicyAgent, num_episodes: int = 3) -> Dict:
        """
        Evaluate with novelty consideration.
        
        Args:
            agent: Agent to evaluate
            num_episodes: Episodes to run
            
        Returns:
            Evaluation results with novelty
        """
        # Standard reward evaluation
        result = super().evaluate_episodes(agent, num_episodes=num_episodes)
        
        # Extract behavior
        behavior = self._extract_behavior(agent)
        novelty = self._calculate_novelty(behavior)
        
        # Add to archive
        self.behavior_archive.append(behavior)
        
        # Combine reward and novelty
        normalized_reward = result['mean'] / (abs(result['mean']) + 1e-8)  # Normalize
        combined_fitness = (
            self.reward_weight * normalized_reward +
            self.novelty_weight * novelty
        )
        
        result['novelty'] = novelty
        result['combined_fitness'] = combined_fitness
        
        return result


class RobustnessFitnessEvaluator(FitnessEvaluator):
    """
    Robustness-based fitness that evaluates agents on perturbed observations.
    Encourages policies robust to sensor noise (NEAT-like robustness evaluation).
    """
    
    def __init__(self, env_name: str = "LunarLander-v3", max_steps: int = 1000,
                 noise_std: float = 0.05):
        """
        Initialize robustness evaluator.
        
        Args:
            env_name: Environment name
            max_steps: Max steps per episode
            noise_std: Standard deviation of observation noise
        """
        super().__init__(env_name, max_steps=max_steps)
        self.noise_std = noise_std
    
    def evaluate_single_episode(self, agent, render=False, add_noise=False):
        """
        Run episode with optional observation noise.
        
        Args:
            agent: Agent to evaluate
            render: Whether to render
            add_noise: Whether to add noise to observations
            
        Returns:
            Total reward
        """
        observation = self.env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Add noise if requested
            if add_noise:
                noisy_obs = observation + np.random.randn(len(observation)) * self.noise_std
            else:
                noisy_obs = observation
            
            action = agent.get_action(noisy_obs)
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
        
        return total_reward
    
    def evaluate_episodes(self, agent, num_episodes=3, render=False):
        """
        Evaluate agent with and without noise, compute robustness.
        
        Args:
            agent: Agent to evaluate
            num_episodes: Episodes per condition
            render: Whether to render
            
        Returns:
            Evaluation results
        """
        # Clean evaluation
        clean_rewards = []
        for _ in range(num_episodes):
            reward = self.evaluate_single_episode(agent, render=render, add_noise=False)
            clean_rewards.append(reward)
        
        # Noisy evaluation
        noisy_rewards = []
        for _ in range(num_episodes):
            reward = self.evaluate_single_episode(agent, render=render, add_noise=True)
            noisy_rewards.append(reward)
        
        self.evaluation_count += 1
        
        return {
            'clean_rewards': clean_rewards,
            'clean_mean': np.mean(clean_rewards),
            'noisy_rewards': noisy_rewards,
            'noisy_mean': np.mean(noisy_rewards),
            'robustness': np.mean(noisy_rewards) / (np.mean(clean_rewards) + 1e-8),
            'combined': (np.mean(clean_rewards) + np.mean(noisy_rewards)) / 2
        }
