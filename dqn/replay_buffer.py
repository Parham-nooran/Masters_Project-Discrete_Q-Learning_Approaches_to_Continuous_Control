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

        # Handle actions - ensure consistent shape handling
        actions_list = []
        for t in batch:
            if isinstance(t.action, np.ndarray):
                action = t.action
            else:
                action = np.array(t.action)

            # Ensure action has the right shape
            if action.ndim == 0:  # scalar
                action = np.array([action])
            elif action.ndim == 1 and len(action) == 1:
                # If it's a 1D array with single element, might need to expand for decoupled case
                # This will be handled when we know the expected shape
                pass

            actions_list.append(action)

        # Convert actions to tensor with proper shape handling
        # First, check if all actions have the same shape
        action_shapes = [a.shape for a in actions_list]
        if len(set(action_shapes)) == 1:
            # All actions have same shape - simple case
            actions = torch.LongTensor(np.array(actions_list))
        else:
            # Different shapes - need to handle carefully
            max_dims = max(len(shape) for shape in action_shapes)
            if max_dims == 1:
                # All are 1D but different lengths - pad or truncate as needed
                max_len = max(shape[0] for shape in action_shapes)
                padded_actions = []
                for action in actions_list:
                    if len(action) < max_len:
                        padded = np.pad(action, (0, max_len - len(action)), mode='constant')
                        padded_actions.append(padded)
                    else:
                        padded_actions.append(action[:max_len])
                actions = torch.LongTensor(np.array(padded_actions))
            else:
                # Mixed dimensions - convert all to same format
                actions = torch.LongTensor(np.array(actions_list, dtype=object).tolist())

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