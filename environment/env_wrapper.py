import gymnasium as gym
import numpy as np
# Add this to the very top of env_wrapper.py after imports
print("Script started!", flush=True)

class EnvironmentWrapper:
    def __init__(self, env_name="LunarLander-v3", render_mode=None):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name: Name of the Gymnasium environment
            render_mode: 'human' for visualization, None for headless
        """
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = gym.make(env_name, render_mode=render_mode)
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.input_size = self.observation_space.shape[0]
        self.output_size = self.action_space.n  # Discrete action space
        
    def reset(self):
        """Reset the environment and return initial observation."""
        observation, info = self.env.reset()
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        return self.env.step(action)
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_dimensions(self):
        """Return input and output dimensions for neural network."""
        return self.input_size, self.output_size


# Quick test
if __name__ == "__main__":
    env = EnvironmentWrapper()
    print(f"Input size: {env.input_size}")
    print(f"Output size: {env.output_size}")
    
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After random action: reward={reward:.2f}")
    
    env.close()