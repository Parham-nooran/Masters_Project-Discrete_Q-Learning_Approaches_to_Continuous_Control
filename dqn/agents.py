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

        # Handle different action_spec formats
        if isinstance(action_spec, dict):
            if 'low' in action_spec and 'high' in action_spec:
                self.action_min = np.array(action_spec['low'])
                self.action_max = np.array(action_spec['high'])
            else:
                raise ValueError(f"Invalid action_spec format: {action_spec}")
        elif hasattr(action_spec, 'low') and hasattr(action_spec, 'high'):
            # Handle gym-style Box spaces
            self.action_min = action_spec.low
            self.action_max = action_spec.high
        else:
            raise ValueError(f"Unsupported action_spec format: {type(action_spec)}")

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
        if isinstance(discrete_actions, torch.Tensor):
            discrete_actions = discrete_actions.cpu().numpy()

        if self.decouple:
            # discrete_actions shape: [batch_size, action_dim]
            continuous_actions = []
            for b in range(discrete_actions.shape[0]):
                action = []
                for dim in range(discrete_actions.shape[1]):
                    bin_idx = discrete_actions[b][dim]  # Use direct indexing
                    action.append(self.action_bins[dim][bin_idx])
                continuous_actions.append(action)
            return np.array(continuous_actions)
        else:
            # discrete_actions shape: [batch_size]
            return np.array([self.action_bins[a] for a in discrete_actions.flatten()])


class DecQNAgent:
    """DecQN Agent with PyTorch implementation."""

    def __init__(self, config, obs_shape, action_spec, device='cpu'):
        self.config = config
        self.device = device
        self.obs_shape = obs_shape
        self.action_spec = action_spec

        # Action discretization
        self.action_discretizer = ActionDiscretizer(action_spec, config.num_bins, config.decouple)

        # Networks
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.learning_rate)
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)

        # Q-networks
        self.q_network = CriticDQN(config, encoder_output_size, action_spec).to(device)
        self.target_q_network = CriticDQN(config, encoder_output_size, action_spec).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount
        )

        # Create actors for training and evaluation
        self.actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=config.epsilon,
            decouple=config.decouple,
            device=device
        )

        self.eval_actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=0.0,  # No exploration during evaluation
            decouple=config.decouple,
            device=device
        )

        # Training state
        self.training_step = 0
        self.epsilon = config.epsilon

    def select_action(self, obs, evaluate=False):
        """Select action using the appropriate actor."""
        if evaluate:
            return self.eval_actor.select_action(obs)
        else:
            return self.actor.select_action(obs)

    def observe_first(self, obs):
        """Handle first observation in an episode."""
        # Store initial observation if needed
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Observe transition and store in replay buffer."""
        if hasattr(self, 'last_obs'):
            # Convert continuous action back to discrete for storage
            discrete_action = self._continuous_to_discrete_action(action)
            self.store_transition(self.last_obs, discrete_action, reward, next_obs, done)
        self.last_obs = next_obs

    def _continuous_to_discrete_action(self, continuous_action):
        """Convert continuous action back to discrete indices for storage."""
        continuous_action = np.array(continuous_action)

        if self.config.decouple:
            # Find closest bin for each dimension
            discrete_action = []
            for dim in range(len(continuous_action)):
                bins = self.action_discretizer.action_bins[dim]
                closest_idx = np.argmin(np.abs(bins - continuous_action[dim]))
                discrete_action.append(closest_idx)
            return np.array(discrete_action)
        else:
            # Find closest action in joint space
            min_dist = float('inf')
            best_idx = 0
            for i, bin_action in enumerate(self.action_discretizer.action_bins):
                dist = np.linalg.norm(np.array(bin_action) - continuous_action)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            return best_idx

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        # Convert discrete action back if needed
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save_checkpoint(self, path, episode):
        """Save agent checkpoint."""
        checkpoint = {
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }

        if self.encoder:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
            checkpoint['encoder_optimizer_state_dict'] = self.encoder_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)

        if self.encoder and 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])

        return checkpoint['episode']

    def update(self):
        """Update the agent."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch
        obs, actions, rewards, next_obs, dones, discounts, weights = (
            x.to(self.device) for x in [obs, actions, rewards, next_obs, dones, discounts, weights])


        # Encode observations
        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

        # Current Q-values
        q1_current, q2_current = self.q_network(obs_encoded)

        # Next Q-values for target
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded)
            q1_next_online, q2_next_online = self.q_network(next_obs_encoded)

            # Double Q-learning target
            if self.config.decouple:
                # For decoupled actions
                next_actions = make_epsilon_greedy_policy((q1_next_online, q2_next_online), 0.0, True)
                q_next_target = 0.5 * (q1_next_target + q2_next_target)  # Average Q-values

                # Select values for each action dimension
                batch_indices = torch.arange(q_next_target.shape[0], device=self.device)
                action_indices = torch.arange(next_actions.shape[1], device=self.device)

                # Use advanced indexing to select Q-values for each action dimension
                q_target_values = q_next_target[batch_indices.unsqueeze(1), action_indices.unsqueeze(0), next_actions]

                # Expand rewards and discounts for decoupled case
                rewards_expanded = rewards.unsqueeze(-1).expand(-1, actions.shape[1])
                discounts_expanded = discounts.unsqueeze(-1).expand(-1, actions.shape[1])
                dones_expanded = dones.unsqueeze(-1).expand(-1, actions.shape[1])

                targets = rewards_expanded + discounts_expanded * q_target_values * ~dones_expanded
            else:
                # Standard case
                next_actions = make_epsilon_greedy_policy((q1_next_online, q2_next_online), 0.0, False)
                q_next_target = 0.5 * (q1_next_target + q2_next_target)
                q_target_values = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones

        # Current Q-values for selected actions
        if self.config.decouple:
            # Ensure actions has the right shape for decoupled case
            expected_action_dims = self.action_discretizer.action_dim

            if actions.shape[1] != expected_action_dims:
                # If actions is smaller than expected, expand it
                if actions.shape[1] < expected_action_dims:
                    # Repeat the action for missing dimensions
                    actions_expanded = actions.repeat(1, expected_action_dims)[:, :expected_action_dims]
                    actions = actions_expanded

            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            action_indices = torch.arange(actions.shape[1], device=self.device)

            # Use advanced indexing to select Q-values for each action dimension
            q1_selected = q1_current[batch_indices.unsqueeze(1), action_indices.unsqueeze(0), actions]
            q2_selected = q2_current[batch_indices.unsqueeze(1), action_indices.unsqueeze(0), actions]
        else:
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute losses
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        if self.config.decouple:
            td_error1 = td_error1.mean(dim=1)  # Average across action dimensions
            td_error2 = td_error2.mean(dim=1)

        # Huber loss
        loss1 = F.smooth_l1_loss(q1_selected, targets, reduction='none')
        loss2 = F.smooth_l1_loss(q2_selected, targets, reduction='none')

        if self.config.decouple:
            loss1 = loss1.mean(dim=1)
            loss2 = loss2.mean(dim=1)

        # Apply importance sampling weights
        loss1 = (loss1 * weights).mean()
        loss2 = (loss2 * weights).mean() if self.config.use_double_q else 0

        total_loss = loss1 + loss2

        # Optimize
        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping
        if self.config.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_gradients_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.clip_gradients_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        # Update priorities in replay buffer
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()
        priorities = 0.5 * (priorities1 + priorities2) if self.config.use_double_q else priorities1
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            'loss': total_loss.item(),
            'q1_mean': q1_selected.mean().item(),
            'q2_mean': q2_selected.mean().item() if self.config.use_double_q else 0,
        }