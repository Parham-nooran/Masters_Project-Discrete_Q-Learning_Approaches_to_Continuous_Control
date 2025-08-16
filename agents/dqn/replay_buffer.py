import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ['obs', 'action', 'reward', 'next_obs', 'done', 'n_step_return', 'n_step_discount'])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=1, discount=0.99):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent
        self.beta = beta  # importance sampling exponent
        self.n_step = n_step
        self.discount = discount

        # Storage
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        # N-step tracking
        self.n_step_buffer = []

    def add(self, obs, action, reward, next_obs, done):
        """Add transition with n-step returns."""
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate n-step return
        n_step_return = 0.0
        n_step_discount = 1.0

        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_return += n_step_discount * r
            n_step_discount *= self.discount
            if d:
                break

        # Get transition to store
        obs_0, action_0, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_obs_n, done_n = self.n_step_buffer[-1]

        transition = Transition(obs_0, action_0, reward, next_obs_n, done_n, n_step_return, n_step_discount)

        # Store in buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        # Set priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

        # Remove processed transition
        self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return None

        # Calculate probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Get transitions
        batch = [self.buffer[idx] for idx in indices]

        # Convert to tensors
        obs = torch.stack([torch.FloatTensor(t.obs) for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.n_step_return for t in batch])
        next_obs = torch.stack([torch.FloatTensor(t.next_obs) for t in batch])
        dones = torch.BoolTensor([t.done for t in batch])
        discounts = torch.FloatTensor([t.n_step_discount for t in batch])
        weights = torch.FloatTensor(weights)

        return obs, actions, rewards, next_obs, dones, discounts, weights, indices

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
