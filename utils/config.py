# Genetic Algorithm Configuration
POPULATION_SIZE = 50  # Reduced for faster testing
GENERATIONS = 1000
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5

# Environment Configuration
ENV_NAME = "LunarLander-v3"
MAX_STEPS = 1000
NUM_EPISODES = 5  # Episodes per fitness evaluation

# Neural Network Configuration
INPUT_SIZE = 8   # LunarLander observation space
HIDDEN_SIZES = [16, 16]  # Two hidden layers
OUTPUT_SIZE = 4  # LunarLander action space (discrete)