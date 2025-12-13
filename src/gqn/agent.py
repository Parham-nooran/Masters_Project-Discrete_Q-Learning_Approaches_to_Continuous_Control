import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.gqn.network import DualDecoupledQNetwork
from src.gqn.action_space_manager import ActionSpaceManager
from src.gqn.scheduler import GrowthScheduler
import sys

sys.path.append("src")
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.encoder import VisionEncoder


class GQNAgent:
    """Growing Q-Networks Agent implementing progressive action space refinement."""

    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = self._determine_observation_dimension(config, obs_shape)
        self.encoder = self._create_encoder_if_needed(config, obs_shape)
        self.action_space_manager = self._create_action_space_manager(
            config, action_spec
        )
        self.q_network, self.target_network = self._create_networks(config)
        self.optimizer = self._create_optimizer(config)
        self.encoder_optimizer = self._create_encoder_optimizer_if_needed(config)
        self.replay_buffer = self._create_replay_buffer(config)
        self.growth_scheduler = self._create_growth_scheduler(config)

        self.epsilon = config.epsilon
        self.update_counter = 0
        self.last_obs = None

    def _determine_observation_dimension(self, config, obs_shape):
        """Determine observation dimension based on whether pixels are used."""
        if len(obs_shape) == 1:
            return obs_shape[0]
        if hasattr(config, "layer_size_bottleneck"):
            return config.layer_size_bottleneck
        return 50

    def _create_encoder_if_needed(self, config, obs_shape):
        """Create vision encoder if using pixel observations."""
        if not config.use_pixels:
            return None
        encoder = VisionEncoder(config, obs_shape[-1]).to(self.device)
        return encoder

    def _create_encoder_optimizer_if_needed(self, config):
        """Create optimizer for encoder if it exists."""
        if self.encoder is None:
            return None
        return optim.Adam(self.encoder.parameters(), lr=config.learning_rate)

    def _create_action_space_manager(self, config, action_spec):
        """Create manager for growing action space."""
        return ActionSpaceManager(
            action_spec, config.initial_bins, config.final_bins, self.device
        )

    def _create_networks(self, config):
        """Create Q-network and target network."""
        q_network = DualDecoupledQNetwork(
            self.obs_dim,
            self.action_space_manager.action_dim,
            config.final_bins,
            config.layer_size,
            config.num_layers,
        ).to(self.device)

        target_network = DualDecoupledQNetwork(
            self.obs_dim,
            self.action_space_manager.action_dim,
            config.final_bins,
            config.layer_size,
            config.num_layers,
        ).to(self.device)

        target_network.load_state_dict(q_network.state_dict())
        return q_network, target_network

    def _create_optimizer(self, config):
        """Create optimizer for Q-network."""
        return optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

    def _create_replay_buffer(self, config):
        """Create prioritized replay buffer."""
        buffer = PrioritizedReplayBuffer(
            config.max_replay_size,
            alpha=config.per_alpha,
            beta=config.per_beta,
            n_step=config.n_step,
            discount=config.discount,
        )
        buffer.to_device(self.device)
        return buffer

    def _create_growth_scheduler(self, config):
        """Create scheduler for managing action space growth."""
        return GrowthScheduler(
            config.growing_schedule,
            config.num_episodes,
            len(self.action_space_manager.growth_sequence),
        )

    def observe_first(self, obs):
        """Observe first state of episode."""
        self.last_obs = obs

    def select_action(self, obs, epsilon=None):
        """Select action using epsilon-greedy policy."""
        epsilon = epsilon if epsilon is not None else self.epsilon

        with torch.no_grad():
            obs_encoded = self._encode_observation(obs)
            q_combined = self._compute_combined_q_values(obs_encoded)
            active_q = self.action_space_manager.get_active_q_values(q_combined)
            discrete_action = self._epsilon_greedy_select(active_q, epsilon)
            continuous_action = self.action_space_manager.discrete_to_continuous(
                discrete_action
            )
            return continuous_action[0]

    def _encode_observation(self, obs):
        """Encode observation using encoder or flatten."""
        if self.encoder:
            return self.encoder(obs.unsqueeze(0))
        return obs.unsqueeze(0).flatten(1)

    def _compute_combined_q_values(self, obs_encoded):
        """Compute combined Q-values from dual networks."""
        q1, q2 = self.q_network(obs_encoded)
        return torch.max(q1, q2)

    def _epsilon_greedy_select(self, q_values, epsilon):
        """Select action with epsilon-greedy exploration."""
        batch_size, action_dim, num_bins = q_values.shape

        random_mask = torch.rand(batch_size, action_dim, device=self.device) < epsilon
        random_actions = torch.randint(
            0, num_bins, (batch_size, action_dim), device=self.device
        )
        greedy_actions = q_values.argmax(dim=2)

        return torch.where(random_mask, random_actions, greedy_actions)

    def observe(self, action, reward, next_obs, done):
        """Observe transition and add to replay buffer."""
        if self.config.action_penalty_coeff > 0:
            reward = self._apply_action_penalty(reward, action)

        self.replay_buffer.add(self.last_obs, action, reward, next_obs, done)
        self.last_obs = next_obs

    def _apply_action_penalty(self, reward, action):
        """Apply action penalty to reward."""
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        action_norm_sq = np.sum(action_np**2)
        penalty = self.config.action_penalty_coeff * action_norm_sq / len(action_np)
        penalty = min(penalty, abs(reward) * 0.1)
        return reward - penalty

    def update(self):
        """Update Q-networks using sampled batch from replay buffer."""
        if not self._has_sufficient_replay_data():
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return None

        obs_encoded, next_obs_encoded = self._encode_batch_observations(batch)
        metrics, total_loss = self._compute_loss_from_batch(
            batch, obs_encoded, next_obs_encoded
        )
        self._perform_gradient_update(total_loss)
        self._update_target_network_if_needed()

        return metrics

    def _has_sufficient_replay_data(self):
        """Check if replay buffer has enough data for training."""
        return len(self.replay_buffer) >= self.config.min_replay_size

    def _encode_batch_observations(self, batch):
        """Encode batch of observations and next observations."""
        obs, _, _, next_obs, _, _, _, _ = batch

        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

        return obs_encoded, next_obs_encoded

    def _compute_loss_from_batch(self, batch, obs_encoded, next_obs_encoded):
        """Compute loss from batch data."""
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch
        return self._compute_loss(
            obs_encoded,
            actions,
            rewards,
            next_obs_encoded,
            dones,
            discounts,
            weights,
            indices,
        )

    def _perform_gradient_update(self, total_loss):
        """Perform gradient descent step."""
        self._zero_all_gradients()
        total_loss.backward()
        self._clip_all_gradients()
        self._step_all_optimizers()

    def _zero_all_gradients(self):
        """Zero gradients for all optimizers."""
        self.optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

    def _clip_all_gradients(self):
        """Clip gradients for all networks."""
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        if self.encoder:
            nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.config.gradient_clip
            )

    def _step_all_optimizers(self):
        """Step all optimizers."""
        self.optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

    def _update_target_network_if_needed(self):
        """Update target network periodically."""
        self.update_counter += 1
        if self.update_counter % self.config.target_update_period == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_loss(
        self, obs, actions, rewards, next_obs, dones, discounts, weights, indices
    ):
        """Compute TD loss for dual Q-networks."""
        q1, q2 = self.q_network(obs)
        discrete_actions = self._continuous_to_discrete_actions(actions)
        q1_values = self._gather_q_values(q1, discrete_actions)
        q2_values = self._gather_q_values(q2, discrete_actions)

        targets = self._compute_td_targets(next_obs, rewards, dones, discounts)

        td_error1 = targets - q1_values
        td_error2 = targets - q2_values

        total_loss = self._compute_weighted_loss(td_error1, td_error2, weights)
        self._update_replay_priorities(indices, td_error1, td_error2)

        metrics = self._create_metrics_dict(
            total_loss, q1_values, q2_values, td_error1, td_error2
        )
        return metrics, total_loss

    def _compute_td_targets(self, next_obs, rewards, dones, discounts):
        """Compute TD targets using target network."""
        with torch.no_grad():
            next_q1, next_q2 = self.target_network(next_obs)
            next_q_combined = torch.max(next_q1, next_q2)
            active_next_q = self.action_space_manager.get_active_q_values(
                next_q_combined
            )
            next_values = active_next_q.max(dim=2)[0].mean(dim=1)
            targets = rewards + discounts * next_values * (~dones)
        return targets

    def _compute_weighted_loss(self, td_error1, td_error2, weights):
        """Compute weighted Huber loss for both Q-networks."""
        loss1 = self._huber_loss(td_error1)
        loss2 = self._huber_loss(td_error2)
        loss1_weighted = (loss1 * weights).mean()
        loss2_weighted = (loss2 * weights).mean()
        return loss1_weighted + loss2_weighted

    def _update_replay_priorities(self, indices, td_error1, td_error2):
        """Update priorities in replay buffer based on TD errors."""
        priorities = (torch.abs(td_error1) + torch.abs(td_error2)) / 2.0
        self.replay_buffer.update_priorities(indices, priorities.detach().cpu().numpy())

    def _create_metrics_dict(
        self, total_loss, q1_values, q2_values, td_error1, td_error2
    ):
        """Create dictionary of training metrics."""
        loss1 = self._huber_loss(td_error1)
        loss2 = self._huber_loss(td_error2)

        return {
            "loss": float(total_loss.item()),
            "q1_mean": float(q1_values.mean().item()),
            "q2_mean": float(q2_values.mean().item()),
            "mean_abs_td_error": float(torch.abs(td_error1).mean().item()),
            "mean_squared_td_error": float((td_error1**2).mean().item()),
            "mse_loss1": float(loss1.mean().item()),
            "mse_loss2": float(loss2.mean().item()),
        }

    def _continuous_to_discrete_actions(self, actions):
        """Convert continuous actions to discrete indices (vectorized)."""
        batch_size = actions.shape[0]
        action_dim = self.action_space_manager.action_dim

        discrete_actions = torch.zeros(
            batch_size, action_dim, dtype=torch.long, device=self.device
        )
        active_indices = torch.tensor(
            self.action_space_manager._get_active_bin_indices(), device=self.device
        )

        for dim in range(action_dim):
            bins = self.action_space_manager.action_bins[dim, active_indices]
            distances = torch.abs(bins.unsqueeze(0) - actions[:, dim].unsqueeze(1))
            discrete_actions[:, dim] = distances.argmin(dim=1)

        return discrete_actions

    def _gather_q_values(self, q_values, actions):
        """Gather Q-values for selected actions (Paper Eq. 2: Q(s,a) = (1/M) * Σ Q_j(s, a_j))."""
        batch_size = q_values.shape[0]
        action_dim = q_values.shape[1]

        active_q = self.action_space_manager.get_active_q_values(q_values)

        batch_indices = (
            torch.arange(batch_size, device=self.device)
            .unsqueeze(1)
            .expand(-1, action_dim)
        )
        dim_indices = (
            torch.arange(action_dim, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        selected_q = active_q[batch_indices, dim_indices, actions]
        return selected_q.sum(dim=1) / action_dim

    def _huber_loss(self, td_error):
        """Compute Huber loss."""
        abs_error = torch.abs(td_error)
        quadratic = torch.minimum(
            abs_error, torch.tensor(self.config.huber_delta, device=self.device)
        )
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + self.config.huber_delta * linear

    def check_and_grow(self, episode, episode_return):
        """Check if action space should grow based on scheduler."""
        if not self.action_space_manager.can_grow():
            return False

        should_grow = self.growth_scheduler.should_grow(episode, episode_return)
        if should_grow:
            self.action_space_manager.grow_action_space()
            return True

        return False

    def update_epsilon(self, decay_rate=None, min_epsilon=None):
        """Update exploration rate with decay."""
        decay_rate = decay_rate if decay_rate is not None else self.config.epsilon_decay
        min_epsilon = (
            min_epsilon if min_epsilon is not None else self.config.min_epsilon
        )
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_checkpoint_state(self):
        """Get state dictionary for checkpointing."""
        state = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_counter": self.update_counter,
            "epsilon": self.epsilon,
            "action_space_state": self.action_space_manager.get_growth_info(),
        }

        if self.encoder:
            state["encoder"] = self.encoder.state_dict()
            state["encoder_optimizer"] = self.encoder_optimizer.state_dict()

        return state

    def load_checkpoint_state(self, checkpoint):
        """Load state from checkpoint dictionary."""
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_counter = checkpoint["update_counter"]
        self.epsilon = checkpoint["epsilon"]

        if "encoder" in checkpoint and self.encoder:
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
