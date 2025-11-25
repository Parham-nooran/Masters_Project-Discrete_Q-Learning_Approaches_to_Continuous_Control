"""
Simple replay buffer for CQN agent.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL.

    Stores transitions and samples batches for training.
    """

    def __init__(
            self,
            capacity: int,
            obs_shape: tuple,
            action_shape: tuple,
            device: torch.device,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            obs_shape: Shape of observations.
            action_shape: Shape of actions.
            device: Device to store tensors on.
        """
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.discounts = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            discount: float,
            next_obs: np.ndarray,
    ):
        """
        Add transition to buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            discount: Discount factor.
            next_obs: Next observation.
        """
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.discounts[self.idx], discount)
        np.copyto(self.next_obses[self.idx], next_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        """
        Sample batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (obs, action, reward, discount, next_obs).
        """
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        discounts = self.discounts[idxs]
        next_obses = self.next_obses[idxs]

        return obses, actions, rewards, discounts, next_obses

    def __len__(self):
        """Return current size of buffer."""
        return self.capacity if self.full else self.idx