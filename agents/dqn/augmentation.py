import torch
import numpy as np
from config import *
from agents import DecQNAgent
import torch.nn.functional as F


def random_shift(images, pad_size=4):
    """Apply random shift augmentation to images."""
    n, c, h, w = images.shape

    # Pad images
    padded = F.pad(images, [pad_size] * 4, mode='replicate')

    # Random crop back to original size
    top = torch.randint(0, 2 * pad_size + 1, (n,))
    left = torch.randint(0, 2 * pad_size + 1, (n,))

    shifted_images = torch.zeros_like(images)
    for i in range(n):
        shifted_images[i] = padded[i, :, top[i]:top[i] + h, left[i]:left[i] + w]

    return shifted_images


# usage and training loop
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