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
        # Convert to CPU tensors for storage efficiency
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu()
        else:
            obs = torch.tensor(obs, dtype=torch.float32)

        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu()
        else:
            next_obs = torch.tensor(next_obs, dtype=torch.float32)

        if isinstance(action, torch.Tensor):
            action = action.cpu()
        elif isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        else:
            action = torch.tensor(action)

        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        if done:
            # Process remaining transitions properly when episode ends
            while len(self.n_step_buffer) > 0:
                current_len = len(self.n_step_buffer)

                # Calculate n-step return for current transition
                n_step_return = 0.0
                n_step_discount = 1.0

                for i in range(min(current_len, self.n_step)):
                    _, _, r, _, d = self.n_step_buffer[i]
                    n_step_return += n_step_discount * r
                    n_step_discount *= self.discount
                    if d:
                        break

                # Get the correct transition components
                obs_0, action_0, reward_0, _, _ = self.n_step_buffer[0]

                # For next_obs, use the observation from the appropriate step
                if current_len < self.n_step:
                    _, _, _, next_obs_n, done_n = self.n_step_buffer[current_len - 1]
                else:
                    _, _, _, next_obs_n, done_n = self.n_step_buffer[self.n_step - 1]

                transition = Transition(obs_0, action_0, reward_0, next_obs_n, done_n, n_step_return, n_step_discount)

                # Store and update position
                if len(self.buffer) < self.capacity:
                    self.buffer.append(transition)
                else:
                    self.buffer[self.position] = transition

                self.priorities[self.position] = self.max_priority
                self.position = (self.position + 1) % self.capacity

                # Remove the processed transition
                self.n_step_buffer.pop(0)
        # Regular n-step processing when buffer is full
        elif len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_return = 0.0
            n_step_discount = 1.0

            for i in range(self.n_step):
                _, _, r, _, d = self.n_step_buffer[i]
                n_step_return += n_step_discount * r
                n_step_discount *= self.discount
                if d:
                    break

            # Get transition components
            obs_0, action_0, reward_0, _, _ = self.n_step_buffer[0]
            _, _, _, next_obs_n, done_n = self.n_step_buffer[-1]

            transition = Transition(obs_0, action_0, reward_0, next_obs_n, done_n, n_step_return, n_step_discount)

            # Store transition
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.position] = transition

            self.priorities[self.position] = self.max_priority
            self.position = (self.position + 1) % self.capacity

            # Remove the oldest transition from n_step_buffer
            self.n_step_buffer.pop(0)

    def sample(self, batch_size):
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return None

        # Calculate probabilities (keep numpy for efficiency)
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

        # Convert directly to tensors
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obs_list = []
        dones_list = []
        discounts_list = []

        for t in batch:
            obs_list.append(t.obs)
            actions_list.append(t.action)
            rewards_list.append(t.n_step_return)
            next_obs_list.append(t.next_obs)
            dones_list.append(t.done)
            discounts_list.append(t.n_step_discount)

        # Stack tensors
        obs = torch.stack(obs_list)
        next_obs = torch.stack(next_obs_list)

        # Handle actions with proper shape
        if all(a.shape == actions_list[0].shape for a in actions_list):
            actions = torch.stack(actions_list)
        else:
            # Handle different action shapes by padding to max dimensions
            max_shape = max(a.shape for a in actions_list)
            padded_actions = []
            for action in actions_list:
                if action.shape != max_shape:
                    # Pad or expand to match max shape
                    if len(action.shape) == 0:  # scalar
                        action = action.unsqueeze(0)
                    if action.shape[0] < max_shape[0]:
                        padding = [0] * (len(action.shape) * 2)
                        padding[-1] = max_shape[0] - action.shape[0]
                        action = torch.nn.functional.pad(action, padding)
                padded_actions.append(action)
            actions = torch.stack(padded_actions)

        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        dones = torch.tensor(dones_list, dtype=torch.bool)
        discounts = torch.tensor(discounts_list, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        return obs, actions, rewards, next_obs, dones, discounts, weights, indices


    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)