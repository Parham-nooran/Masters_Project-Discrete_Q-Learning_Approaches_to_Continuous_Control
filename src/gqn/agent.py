import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.gqn.network import DualDecoupledQNetwork
from src.gqn.action_space_manager import ActionSpaceManager
from src.gqn.scheduler import GrowthScheduler
import sys

sys.path.append('src')
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.encoder import VisionEncoder


class GQNAgent:
    """Growing Q-Networks Agent."""

    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = obs_shape[0] if len(obs_shape) == 1 else config.layer_size_bottleneck if hasattr(config,
                                                                                                        'layer_size_bottleneck') else 50

        self.encoder = None
        if config.use_pixels:
            self.encoder = VisionEncoder(config, obs_shape[-1]).to(self.device)
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.learning_rate)

        self.action_space_manager = ActionSpaceManager(
            action_spec, config.initial_bins, config.final_bins, self.device
        )

        self.q_network = DualDecoupledQNetwork(
            self.obs_dim,
            self.action_space_manager.action_dim,
            config.final_bins,
            config.layer_size,
            config.num_layers
        ).to(self.device)

        self.target_network = DualDecoupledQNetwork(
            self.obs_dim,
            self.action_space_manager.action_dim,
            config.final_bins,
            config.layer_size,
            config.num_layers
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(
            config.max_replay_size,
            alpha=config.per_alpha,
            beta=config.per_beta,
            n_step=config.n_step,
            discount=config.discount
        )
        self.replay_buffer.to_device(self.device)

        self.growth_scheduler = GrowthScheduler(
            config.growing_schedule,
            config.num_episodes,
            len(self.action_space_manager.growth_sequence)
        )

        self.epsilon = config.epsilon
        self.update_counter = 0
        self.last_obs = None

    def observe_first(self, obs):
        """Observe first state of episode."""
        self.last_obs = obs

    def select_action(self, obs, epsilon=None):
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon

        with torch.no_grad():
            if self.encoder:
                obs_encoded = self.encoder(obs.unsqueeze(0))
            else:
                obs_encoded = obs.unsqueeze(0).flatten(1)

            q1, q2 = self.q_network(obs_encoded)
            q_combined = torch.max(q1, q2)

            active_q = self.action_space_manager.get_active_q_values(q_combined)

            discrete_action = self._epsilon_greedy_select(active_q, epsilon)
            continuous_action = self.action_space_manager.discrete_to_continuous(discrete_action)

            return continuous_action[0]

    def _epsilon_greedy_select(self, q_values, epsilon):
        """Select action with epsilon-greedy."""
        batch_size = q_values.shape[0]
        action_dim = q_values.shape[1]
        num_bins = q_values.shape[2]

        random_mask = torch.rand(batch_size, action_dim, device=self.device) < epsilon
        random_actions = torch.randint(0, num_bins, (batch_size, action_dim), device=self.device)
        greedy_actions = q_values.argmax(dim=2)

        actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions

    def observe(self, action, reward, next_obs, done):
        """Observe transition and add to replay buffer."""
        if self.config.action_penalty_coeff > 0:
            reward = self._apply_action_penalty(reward, action)

        self.replay_buffer.add(self.last_obs, action, reward, next_obs, done)
        self.last_obs = next_obs

    def _apply_action_penalty(self, reward, action):
        """Apply action penalty to reward."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        action_norm_sq = np.sum(action ** 2)
        penalty = self.config.action_penalty_coeff * action_norm_sq / len(action)
        penalty = min(penalty, abs(reward) * 0.1)

        return reward - penalty

    def update(self):
        """Update Q-networks."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch

        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

        total_loss, metrics = self._compute_loss(obs_encoded, actions, rewards, next_obs_encoded,
                                                 dones, discounts, weights, indices)

        self.optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        if self.encoder:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.gradient_clip)

        self.optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.config.target_update_period == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return metrics

    def _compute_loss(self, obs, actions, rewards, next_obs, dones, discounts, weights, indices):
        """Compute TD loss."""
        q1, q2 = self.q_network(obs)

        discrete_actions = self._continuous_to_discrete_actions(actions)

        q1_values = self._gather_q_values(q1, discrete_actions)
        q2_values = self._gather_q_values(q2, discrete_actions)

        with torch.no_grad():
            next_q1, next_q2 = self.target_network(next_obs)
            next_q_combined = torch.max(next_q1, next_q2)

            active_next_q = self.action_space_manager.get_active_q_values(next_q_combined)

            next_values = active_next_q.max(dim=2)[0].mean(dim=1)

            targets = rewards + discounts * next_values * (~dones)

        td_error1 = targets - q1_values
        td_error2 = targets - q2_values

        loss1 = self._huber_loss(td_error1)
        loss2 = self._huber_loss(td_error2)

        loss1_weighted = (loss1 * weights).mean()
        loss2_weighted = (loss2 * weights).mean()

        total_loss = loss1_weighted + loss2_weighted

        priorities = (torch.abs(td_error1) + torch.abs(td_error2)) / 2.0
        self.replay_buffer.update_priorities(indices, priorities.detach().cpu().numpy())

        metrics = {
            'loss': float(total_loss.item()),
            'q1_mean': float(q1_values.mean().item()),
            'q2_mean': float(q2_values.mean().item()),
            'mean_abs_td_error': float(torch.abs(td_error1).mean().item()),
            'mean_squared_td_error': float((td_error1 ** 2).mean().item()),
            'mse_loss1': float(loss1.mean().item()),
            'mse_loss2': float(loss2.mean().item())
        }
        return total_loss, metrics

    def _continuous_to_discrete_actions(self, actions):
        """Convert continuous actions to discrete indices."""
        batch_size = actions.shape[0]
        action_dim = self.action_space_manager.action_dim

        discrete_actions = torch.zeros(batch_size, action_dim, dtype=torch.long, device=self.device)

        active_indices = self.action_space_manager._get_active_bin_indices()

        for dim in range(action_dim):
            bins = self.action_space_manager.action_bins[dim, active_indices]
            distances = torch.abs(bins.unsqueeze(0) - actions[:, dim].unsqueeze(1))
            discrete_actions[:, dim] = distances.argmin(dim=1)

        return discrete_actions

    def _gather_q_values(self, q_values, actions):
        """Gather Q-values for selected actions."""
        batch_size = q_values.shape[0]
        action_dim = q_values.shape[1]

        active_q = self.action_space_manager.get_active_q_values(q_values)

        q_vals = []
        for b in range(batch_size):
            q_sum = 0.0
            for d in range(action_dim):
                q_sum += active_q[b, d, actions[b, d]]
            q_vals.append(q_sum / action_dim)

        return torch.stack(q_vals)

    def _huber_loss(self, td_error):
        """Compute Huber loss."""
        abs_error = torch.abs(td_error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.config.huber_delta, device=self.device))
        linear = abs_error - quadratic
        return 0.5 * quadratic ** 2 + self.config.huber_delta * linear

    def check_and_grow(self, episode, episode_return):
        """Check if action space should grow."""
        if not self.action_space_manager.can_grow():
            return False

        should_grow = self.growth_scheduler.should_grow(episode, episode_return)

        if should_grow:
            self.action_space_manager.grow_action_space()
            return True

        return False

    def update_epsilon(self, decay_rate=None, min_epsilon=None):
        """Update exploration rate."""
        if decay_rate is None:
            decay_rate = self.config.epsilon_decay
        if min_epsilon is None:
            min_epsilon = self.config.min_epsilon

        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_checkpoint_state(self):
        """Get state for checkpointing."""
        state = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_counter': self.update_counter,
            'epsilon': self.epsilon,
            'action_space_state': self.action_space_manager.get_growth_info()
        }

        if self.encoder:
            state['encoder'] = self.encoder.state_dict()
            state['encoder_optimizer'] = self.encoder_optimizer.state_dict()

        return state

    def load_checkpoint_state(self, checkpoint):
        """Load state from checkpoint."""
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_counter = checkpoint['update_counter']
        self.epsilon = checkpoint['epsilon']

        if 'encoder' in checkpoint and self.encoder:
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])