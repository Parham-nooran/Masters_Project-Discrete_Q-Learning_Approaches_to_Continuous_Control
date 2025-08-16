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

