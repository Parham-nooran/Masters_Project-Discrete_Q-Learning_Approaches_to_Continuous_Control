import numpy as np
import torch
from config import Config
from agents import DecQNAgent


def train_decqn():
    """training function."""
    config = Config()
    config.algorithm = 'decqnvis'  # Use decoupled + vision
    config.use_double_q = True
    config.num_episodes = 1000

    # Mock environment setup - replace with actual environment
    obs_shape = (3, 84, 84) if config.use_pixels else (17,)  # Walker state dim
    action_spec = {'low': np.array([-1.0, -1.0]), 'high': np.array([1.0, 1.0])}  # 2D continuous

    agent = DecQNAgent(config, obs_shape, action_spec, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Training loop would go here
    print("DecQN Agent initialized successfully!")
    print(f"Using algorithm: {config.algorithm}")
    print(f"Decouple: {config.decouple}, Use pixels: {config.use_pixels}")
    return agent


if __name__ == "__main__":
    agent = train_decqn()