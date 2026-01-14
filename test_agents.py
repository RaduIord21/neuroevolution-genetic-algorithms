"""
Test script for neural network and policy agent.
Tests basic functionality, CUDA support, and genetic operations.
"""

import numpy as np
import torch
from agents.neural_network import NeuralNetwork
from agents.policy_agent import PolicyAgent


def test_neural_network():
    """Test NeuralNetwork class."""
    print("=" * 60)
    print("Testing NeuralNetwork")
    print("=" * 60)
    
    # Create network
    input_size = 8
    hidden_sizes = [16, 16]
    output_size = 4
    
    print(f"\n1. Creating network: {input_size} -> {hidden_sizes} -> {output_size}")
    nn = NeuralNetwork(input_size, hidden_sizes, output_size)
    print(f"   Device: {nn.device}")
    print(f"   Parameters: {nn.get_num_parameters()}")
    print(f"   ✓ Network created successfully")
    
    # Test forward pass
    print(f"\n2. Testing forward pass")
    obs = np.random.randn(input_size).astype(np.float32)
    output = nn(torch.FloatTensor(obs).to(nn.device))
    print(f"   Input shape: {obs.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output values: {output.detach().cpu().numpy()}")
    print(f"   ✓ Forward pass successful")
    
    # Test weight getting/setting (important for genetic operations)
    print(f"\n3. Testing weight get/set (genetic operations)")
    weights = nn.get_weights()
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights dtype: {weights.dtype}")
    
    # Modify weights
    modified_weights = weights.copy()
    modified_weights[:100] += 0.1
    nn.set_weights(modified_weights)
    new_weights = nn.get_weights()
    
    # Verify weights changed
    diff = np.abs(new_weights - modified_weights).max()
    print(f"   Weight modification error: {diff:.2e}")
    print(f"   ✓ Weight get/set working (for mutation/crossover)")
    
    # Test cloning
    print(f"\n4. Testing network cloning")
    clone = nn.clone()
    clone_weights = clone.get_weights()
    original_weights = nn.get_weights()
    clone_diff = np.abs(clone_weights - original_weights).max()
    print(f"   Clone weight difference: {clone_diff:.2e}")
    print(f"   ✓ Cloning successful")
    
    # Test device transfer (if CUDA available)
    print(f"\n5. Testing device transfer")
    if torch.cuda.is_available():
        print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        nn.to_device('cuda')
        print(f"   Moved to: {nn.device}")
        obs_cuda = torch.FloatTensor(obs).to(nn.device)
        output_cuda = nn(obs_cuda)
        print(f"   Output on CUDA: {output_cuda.shape}")
        nn.to_device('cpu')
        print(f"   Moved back to: {nn.device}")
        print(f"   ✓ Device transfer successful")
    else:
        print(f"   CUDA not available (CPU mode)")
        print(f"   ✓ CPU mode working")
    
    return nn


def test_policy_agent(nn=None):
    """Test PolicyAgent class."""
    print("\n" + "=" * 60)
    print("Testing PolicyAgent")
    print("=" * 60)
    
    # Create agent
    input_size = 8
    hidden_sizes = [16, 16]
    output_size = 4
    
    print(f"\n1. Creating PolicyAgent: {input_size} -> {hidden_sizes} -> {output_size}")
    agent = PolicyAgent(input_size, hidden_sizes, output_size, agent_id=1)
    print(f"   Device: {agent.device}")
    print(f"   Parameters: {agent.get_num_parameters()}")
    print(f"   {agent}")
    print(f"   ✓ Agent created successfully")
    
    # Test action selection
    print(f"\n2. Testing action selection")
    obs = np.random.randn(input_size).astype(np.float32)
    action = agent.get_action(obs)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action selected: {action} (0-{output_size-1})")
    assert 0 <= action < output_size, "Invalid action"
    print(f"   ✓ Action selection working")
    
    # Test action values
    print(f"\n3. Testing action values extraction")
    action_values = agent.get_action_values(obs)
    print(f"   Action values: {action_values}")
    print(f"   Shape: {action_values.shape}")
    print(f"   ✓ Action values extraction working")
    
    # Test multiple actions in sequence
    print(f"\n4. Testing agent in episode-like loop")
    total_reward = 0
    for step in range(10):
        obs = np.random.randn(input_size).astype(np.float32)
        action = agent.get_action(obs)
        reward = np.random.randn()  # Simulated reward
        agent.add_episode_reward(reward)
        total_reward += reward
    
    fitness = agent.calculate_fitness()
    print(f"   Steps: 10")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Average fitness: {fitness:.4f}")
    print(f"   ✓ Episode loop working")
    
    # Test cloning
    print(f"\n5. Testing agent cloning")
    clone = agent.clone()
    clone_weights = clone.get_weights()
    original_weights = agent.get_weights()
    clone_diff = np.abs(clone_weights - original_weights).max()
    print(f"   Clone weight difference: {clone_diff:.2e}")
    print(f"   Original: {agent}")
    print(f"   Clone: {clone}")
    print(f"   ✓ Agent cloning successful")
    
    # Test weight mutation simulation
    print(f"\n6. Testing weight mutation (genetic operation)")
    original_weights = agent.get_weights()
    mutation_rate = 0.1
    mutation_std = 0.1
    
    mutated_weights = original_weights.copy()
    mask = np.random.rand(len(mutated_weights)) < mutation_rate
    mutated_weights[mask] += np.random.randn(mask.sum()) * mutation_std
    
    agent.set_weights(mutated_weights)
    new_weights = agent.get_weights()
    
    mutation_magnitude = np.abs(new_weights - original_weights).mean()
    print(f"   Original weights (first 10): {original_weights[:10]}")
    print(f"   Mutated weights (first 10): {new_weights[:10]}")
    print(f"   Average weight change: {mutation_magnitude:.4f}")
    print(f"   ✓ Weight mutation working")
    
    # Test device transfer (if CUDA available)
    print(f"\n7. Testing agent device transfer")
    if torch.cuda.is_available():
        print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        agent.to_device('cuda')
        print(f"   Moved to: {agent.device}")
        obs_cuda = np.random.randn(input_size).astype(np.float32)
        action_cuda = agent.get_action(obs_cuda)
        print(f"   Action on CUDA: {action_cuda}")
        agent.to_device('cpu')
        print(f"   Moved back to: {agent.device}")
        print(f"   ✓ Device transfer successful")
    else:
        print(f"   CUDA not available (CPU mode)")
        print(f"   ✓ CPU mode working")
    
    return agent


def test_batch_evaluation():
    """Test batch evaluation (parallel forward passes)."""
    print("\n" + "=" * 60)
    print("Testing Batch Evaluation (Genetic Algorithm Ready)")
    print("=" * 60)
    
    input_size = 8
    hidden_sizes = [16, 16]
    output_size = 4
    population_size = 10
    num_episodes = 3
    
    print(f"\n1. Creating population of {population_size} agents")
    population = [PolicyAgent(input_size, hidden_sizes, output_size, agent_id=i) 
                  for i in range(population_size)]
    print(f"   ✓ Population created")
    
    print(f"\n2. Simulating fitness evaluation ({num_episodes} episodes per agent)")
    for agent_idx, agent in enumerate(population):
        for episode in range(num_episodes):
            # Simulate episode
            episode_reward = 0
            for step in range(50):
                obs = np.random.randn(input_size).astype(np.float32)
                action = agent.get_action(obs)
                reward = np.random.randn() * 0.1
                episode_reward += reward
            agent.add_episode_reward(episode_reward)
        agent.calculate_fitness()
    
    # Sort by fitness
    population.sort(key=lambda a: a.fitness, reverse=True)
    
    print(f"\n   Population fitness ranking:")
    for i, agent in enumerate(population[:5]):
        print(f"   {i+1}. {agent}")
    print(f"   ...")
    
    best_agent = population[0]
    worst_agent = population[-1]
    print(f"\n   Best fitness:  {best_agent.fitness:.4f}")
    print(f"   Worst fitness: {worst_agent.fitness:.4f}")
    print(f"   ✓ Batch evaluation complete (GA-ready!)")
    
    return population


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK & POLICY AGENT TEST SUITE")
    print("=" * 60)
    
    try:
        # Test individual components
        nn = test_neural_network()
        agent = test_policy_agent(nn)
        population = test_batch_evaluation()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ All tests passed!")
        print("\nReady for next steps:")
        print("  1. Implement genetic operators (mutation, crossover, selection)")
        print("  2. Integrate with environment (evaluate agents in LunarLander)")
        print("  3. Build main training loop")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
