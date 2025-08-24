import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple(
    "Transition",
    ["obs", "action", "reward", "next_obs", "done", "n_step_return", "n_step_discount"],
)


def stack_actions(actions_list):
    try:
        return torch.stack(actions_list)
    except RuntimeError:
        # Handle mixed shapes - find the target shape
        shapes = [a.shape for a in actions_list]

        # Check if all actions are scalars or 1D with different lengths
        if all(len(shape) <= 1 for shape in shapes):
            # Convert all to 1D tensors and find max length
            max_len = max(a.numel() for a in actions_list)

            padded_actions = []
            for action in actions_list:
                # Flatten to 1D
                flat_action = action.flatten()
                # Pad if necessary
                if flat_action.numel() < max_len:
                    padding = max_len - flat_action.numel()
                    flat_action = torch.nn.functional.pad(flat_action, (0, padding))
                padded_actions.append(flat_action)

            return torch.stack(padded_actions)

        else:
            # Handle multi-dimensional case - pad to largest shape in each dimension
            max_shape = list(shapes[0])
            for shape in shapes[1:]:
                for i, dim_size in enumerate(shape):
                    if i < len(max_shape):
                        max_shape[i] = max(max_shape[i], dim_size)
                    else:
                        max_shape.append(dim_size)

            padded_actions = []
            for action in actions_list:
                current_shape = list(action.shape)

                # Add missing dimensions
                while len(current_shape) < len(max_shape):
                    action = action.unsqueeze(-1)
                    current_shape.append(1)

                # Pad existing dimensions
                pad_sizes = []
                for i in range(len(max_shape)):
                    pad_size = max_shape[-(i + 1)] - current_shape[-(i + 1)]
                    pad_sizes.extend([0, pad_size])

                if any(p > 0 for p in pad_sizes):
                    action = torch.nn.functional.pad(action, pad_sizes)

                padded_actions.append(action)

            return torch.stack(padded_actions)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""

    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=1, discount=0.99):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent
        self.beta = beta  # importance sampling exponent
        self.n_step = n_step
        self.discount = discount
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Storage
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        # N-step tracking
        self.n_step_buffer = []

    def to_device(self, device):
        """Move the replay buffer to the specified device."""
        self.device = device
        # Convert all stored tensors to the new device
        new_buffer = []
        for transition in self.buffer:
            if transition is not None:
                new_transition = Transition(
                    obs=self._ensure_tensor_device(transition.obs, device),
                    action=self._ensure_tensor_device(transition.action, device),
                    reward=transition.reward,
                    next_obs=self._ensure_tensor_device(transition.next_obs, device),
                    done=transition.done,
                    n_step_return=transition.n_step_return,
                    n_step_discount=transition.n_step_discount
                )
                new_buffer.append(new_transition)
            else:
                new_buffer.append(None)
        self.buffer = new_buffer

    def _ensure_tensor_device(self, tensor, device):
        """Ensure tensor is on the correct device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(device)
        return tensor

    def _ensure_tensor_on_cpu(self, data):
        """Convert data to tensor and ensure it's on CPU for storage."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            return torch.tensor(data, dtype=torch.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Add transition with n-step returns."""
        # Always store tensors on CPU for memory efficiency, but ensure they're detached
        obs = self._ensure_tensor_on_cpu(obs)
        next_obs = self._ensure_tensor_on_cpu(next_obs)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu()
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

                transition = Transition(
                    obs_0,
                    action_0,
                    reward_0,
                    next_obs_n,
                    done_n,
                    n_step_return,
                    n_step_discount,
                )

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

            transition = Transition(
                obs_0,
                action_0,
                reward_0,
                next_obs_n,
                done_n,
                n_step_return,
                n_step_discount,
            )

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
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        batch = [self.buffer[idx] for idx in indices]
        obs_list = [t.obs.to(self.device, non_blocking=True) for t in batch]
        next_obs_list = [t.next_obs.to(self.device, non_blocking=True) for t in batch]
        actions_list = [t.action.to(self.device, non_blocking=True) for t in batch]
        rewards_list = [t.n_step_return for t in batch]
        dones_list = [t.done for t in batch]
        discounts_list = [t.n_step_discount for t in batch]

        # Stack tensors that are now all on the same device
        obs = torch.stack(obs_list)
        next_obs = torch.stack(next_obs_list)
        actions = stack_actions(actions_list)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
        discounts = torch.tensor(
            discounts_list, dtype=torch.float32, device=self.device
        )
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return (
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            discounts,
            weights_tensor,
            indices,
        )

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, 1e-6)  # Avoid zero priorities
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)