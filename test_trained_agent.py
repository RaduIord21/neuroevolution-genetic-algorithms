"""
Test and visualize trained agents.
Loads a saved agent and runs it in the environment for visualization.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from environment.env_wrapper import EnvironmentWrapper


def load_agent(agent_path):
    """Load a saved agent from file."""
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    return agent


def visualize_agent(agent, env, num_episodes=5, max_steps=1000):
    """
    Visualize agent performance.
    
    Args:
        agent: Trained PolicyAgent
        env: EnvironmentWrapper
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode
    """
    episode_rewards = []
    
    print(f"\n{'='*70}")
    print(f"Visualizing Agent Performance")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        observation = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Agent selects action
            action = agent.get_action(observation)
            
            # Environment step
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1:2d}/{num_episodes}: Reward = {total_reward:8.2f} | Steps = {step:4d}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"Statistics:")
    print(f"{'='*70}")
    print(f"Mean reward: {np.mean(episode_rewards):8.2f}")
    print(f"Std reward:  {np.std(episode_rewards):8.2f}")
    print(f"Max reward:  {np.max(episode_rewards):8.2f}")
    print(f"Min reward:  {np.min(episode_rewards):8.2f}")
    print(f"{'='*70}\n")
    
    return episode_rewards


def find_latest_run(results_dir="results"):
    """Find the latest training run directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None
    
    # Get all run directories sorted by modification time
    runs = sorted(
        results_path.glob("run_*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    return runs[0] if runs else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize trained agent")
    parser.add_argument('--agent', type=str, default=None,
                       help='Path to saved agent (best_agent.pkl or checkpoint). '
                            'If not specified, uses latest trained agent')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Max steps per episode')
    parser.add_argument('--env', type=str, default='LunarLander-v3',
                       help='Gymnasium environment name')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory where training results are saved')
    
    args = parser.parse_args()
    
    # Find agent to load
    agent_path = None
    if args.agent:
        agent_path = Path(args.agent)
        if not agent_path.exists():
            print(f"Error: Agent file not found at {agent_path}")
            return
    else:
        # Find latest run
        latest_run = find_latest_run(args.results_dir)
        if not latest_run:
            print(f"Error: No training runs found in {args.results_dir}")
            print("Please train an agent first with: python main.py")
            return
        
        agent_path = latest_run / "best_agent.pkl"
        if not agent_path.exists():
            print(f"Error: No best_agent.pkl found in {latest_run}")
            return
        
        print(f"Using latest agent from: {latest_run}")
    
    # Load agent
    print(f"Loading agent from: {agent_path}")
    try:
        agent = load_agent(agent_path)
        print(f"Agent loaded successfully: {agent}")
    except Exception as e:
        print(f"Error loading agent: {type(e).__name__}: {e}")
        return
    
    # Create environment with rendering
    print(f"\nCreating environment: {args.env}")
    try:
        env = EnvironmentWrapper(args.env, render_mode='human')
        print(f"Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {type(e).__name__}: {e}")
        return
    
    # Run visualization
    try:
        rewards = visualize_agent(agent, env, num_episodes=args.episodes, 
                                 max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    except Exception as e:
        print(f"\nError during visualization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    main()
