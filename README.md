# Neuroevolution with Genetic Algorithms

A comprehensive implementation of neuroevolution using genetic algorithms to train neural network policies for control tasks. This project evolves neural network weights using evolutionary operators like mutation, crossover, and selection to optimize agent behavior in OpenAI Gym environments.

## Features

- **Neuroevolution Framework**: Evolve neural network policies using genetic algorithms
- **Flexible Neural Networks**: Configurable architectures with multiple hidden layers
- **Genetic Operators**: 
  - Tournament selection
  - Uniform crossover
  - Gaussian and adaptive Gaussian mutation
- **Fitness Evaluation**: Multi-episode evaluation with adaptive fitness assessment
- **Environment Support**: Compatible with OpenAI Gym environments (e.g., LunarLander-v3)
- **Results Tracking**: Automatic save of best agents, training history, and metadata
- **Comprehensive Testing**: Unit tests for agents, genetic algorithms, and trained models

## Project Structure

```
├── agents/                 # Neural network agent implementations
│   ├── base_agent.py       # Base agent class
│   ├── neural_network.py   # Neural network models
│   └── policy_agent.py     # Policy-based agent
├── environment/            # Environment wrappers
│   └── env_wrapper.py      # Gym environment wrapper
├── evaluation/             # Fitness evaluation
│   └── fitness.py          # Fitness evaluator implementations
├── genetic/                # Genetic algorithm components
│   ├── crossover.py        # Crossover operators
│   ├── mutation.py         # Mutation operators
│   ├── population.py       # Population management
│   └── selection.py        # Selection strategies
├── results/                # Training results and checkpoints
├── utils/                  # Configuration and utilities
│   └── config.py           # Training configuration
├── main.py                 # Main training script
└── test_*.py              # Unit tests
```

## Installation

### Requirements
- Python 3.8+
- NumPy
- Gymnasium (or OpenAI Gym)
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/RaduIord21/neuroevolution-genetic-algorithms.git
cd neuroevolution-genetic-algorithms

# Install dependencies
pip install numpy gymnasium matplotlib tqdm
```

## Configuration

Edit `utils/config.py` to customize the genetic algorithm and environment settings:

```python
# Genetic Algorithm Configuration
POPULATION_SIZE = 50
GENERATIONS = 1000
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5

# Environment Configuration
ENV_NAME = "LunarLander-v3"
MAX_STEPS = 1000
NUM_EPISODES = 5

# Neural Network Configuration
INPUT_SIZE = 8
HIDDEN_SIZES = [16, 16]
OUTPUT_SIZE = 4
```

## Usage

### Training a New Model

```bash
python main.py
```

The training script will:
1. Initialize a population of random neural network agents
2. Evaluate fitness for each agent in the population
3. Apply selection, crossover, and mutation operators
4. Save the best agent and training history after each generation

### Training with Custom Parameters

```bash
python main.py --population-size 100 --generations 500 --visualize
```

### Testing Agents

```bash
# Test the genetic algorithm
python test_genetic_algorithms.py

# Test agent implementations
python test_agents.py

# Test a trained agent
python test_trained_agent.py --weights results/run_XXXXXXX/best_agent_weights.npy
```

## Results

Training results are automatically saved to the `results/` directory with the following structure:

```
results/run_YYYYMMDD_HHMMSS/
├── best_agent_weights.npy    # Best agent's neural network weights
├── history.json              # Training history (fitness, population stats)
└── metadata.json             # Training metadata and configuration
```

### Analyzing Results

Results include:
- **Fitness progression**: Track best, mean, and population statistics across generations
- **Agent weights**: NumPy arrays of the best evolved agent
- **Training metadata**: Configuration, environment info, and execution details

## How It Works

### 1. Population Initialization
A population of agents with randomly initialized neural network weights is created.

### 2. Fitness Evaluation
Each agent is evaluated by running multiple episodes in the environment and computing average reward/fitness.

### 3. Selection
Tournament selection identifies the fittest agents as parents for reproduction.

### 4. Genetic Operations
- **Crossover**: Combine parent weights using uniform crossover
- **Mutation**: Add Gaussian noise to weights with configurable rates

### 5. Elite Preservation
The best agents are preserved in each generation to prevent loss of good solutions.

### 6. Iteration
Steps 2-5 repeat for a specified number of generations until convergence.

## Key Components

### Agents
- `PolicyAgent`: Neural network-based policy agent
- `NeuralNetwork`: Flexible multi-layer neural network implementation

### Genetic Operators
- `UniformCrossover`: Mix parent genes uniformly
- `GaussianMutation`: Add Gaussian noise to weights
- `AdaptiveGaussianMutation`: Adaptive mutation based on population diversity
- `TournamentSelection`: Select parents via tournament selection

### Fitness Evaluation
- `FitnessEvaluator`: Standard fitness computation
- `AdaptiveFitnessEvaluator`: Fitness with adaptive normalization

## Example Results

The project includes pre-trained agents for LunarLander-v3 in the `results/` directory. These agents were successfully evolved to land the lunar module with high rewards.

## Testing

Run the test suite to verify the implementation:

```bash
python test_agents.py
python test_genetic_algorithms.py
python test_trained_agent.py
```

