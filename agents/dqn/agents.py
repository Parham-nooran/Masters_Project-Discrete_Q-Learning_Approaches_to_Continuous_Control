import itertools
import random

import torch.nn.functional as F
import torch.optim as optim

from actors import CustomDiscreteFeedForwardActor
from critic import *
from encoder import *
from replay_buffer import *


def make_epsilon_greedy_policy(q_values, epsilon, decouple=False):
    """Create epsilon-greedy policy from Q-values."""
    batch_size = q_values[0].shape[0] if isinstance(q_values, tuple) else q_values.shape[0]

    if isinstance(q_values, tuple):  # Double Q
        q_max = torch.max(q_values[0], q_values[1])
    else:
        q_max = q_values

    if decouple:
        # For decoupled actions, sample each dimension independently
        actions = []
        for b in range(batch_size):
            action_per_dim = []
            for dim in range(q_max.shape[1]):
                if random.random() < epsilon:
                    action_per_dim.append(random.randint(0, q_max.shape[2] - 1))
                else:
                    action_per_dim.append(q_max[b, dim].argmax().item())
            actions.append(action_per_dim)
        return torch.LongTensor(actions)
    else:
        # Standard epsilon-greedy
        if random.random() < epsilon:
            return torch.randint(0, q_max.shape[1], (batch_size,))
        else:
            return q_max.argmax(dim=1)


class ActionDiscretizer:
    """Handles continuous to discrete action conversion."""

    def __init__(self, action_spec, num_bins, decouple=False):
        self.num_bins = num_bins
        self.decouple = decouple
        self.action_min = action_spec.low if hasattr(action_spec, 'low') else action_spec['low']
        self.action_max = action_spec.high if hasattr(action_spec, 'high') else action_spec['high']
        self.action_dim = len(self.action_min)

        if decouple:
            # Per-dimension discretization
            self.action_bins = np.linspace(self.action_min, self.action_max, num_bins).T
        else:
            # Joint discretization - create all combinations
            bins_per_dim = [np.linspace(self.action_min[i], self.action_max[i], num_bins)
                            for i in range(self.action_dim)]
            self.action_bins = list(itertools.product(*bins_per_dim))

    def discrete_to_continuous(self, discrete_actions):
        """Convert discrete actions to continuous."""
        if self.decouple:
            # discrete_actions shape: [batch_size, action_dim]
            continuous_actions = []
            for b in range(discrete_actions.shape[0]):
                action = []
                for dim in range(discrete_actions.shape[1]):
                    bin_idx = discrete_actions[b, dim].item()
                    action.append(self.action_bins[dim][bin_idx])
                continuous_actions.append(action)
            return np.array(continuous_actions)
        else:
            # discrete_actions shape: [batch_size]
            return np.array([self.action_bins[a.item()] for a in discrete_actions])

