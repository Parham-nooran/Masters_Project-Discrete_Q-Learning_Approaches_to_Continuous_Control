import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.common.encoder import VisionEncoder
from src.common.logger import Logger
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import (
    continuous_to_discrete_action,
    get_batch_components,
    encode_observation,
    check_and_sample_batch_from_replay_buffer,
)
from src.gqn.critic import GrowingQCritic
from src.gqn.discretizer import GrowingActionDiscretizer
from src.gqn.scheduler import GrowingScheduler


class GrowingQNAgent(Logger):
    """Growing Q-Networks Agent with softmax exploration and min double Q-learning."""

    def __init__(self, config, obs_shape, action_spec, working_dir):
        super().__init__(working_dir)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec

        self.action_discretizer = GrowingActionDiscretizer(
            action_spec, config.max_bins, config.decouple
        )
        self.scheduler = GrowingScheduler(config.num_episodes)

        self._init_networks(obs_shape, action_spec)
        self._init_replay_buffer()
        self._init_training_state()

    def _init_networks(self, obs_shape, action_spec):
        """Initialize Q-networks with decorrelated initialization."""
        if self.config.use_pixels:
            self.encoder = VisionEncoder(self.config, self.config.num_pixels).to(self.device)
            encoder_output_size = self.config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=self.config.learning_rate
            )
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)

        self.q_network = GrowingQCritic(
            self.config, encoder_output_size, action_spec, init_seed=42
        ).to(self.device)

        self.target_q_network = GrowingQCritic(
            self.config, encoder_output_size, action_spec, init_seed=123
        ).to(self.device)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

    def _init_replay_buffer(self):
        """Initialize prioritized replay buffer."""
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.max_replay_size,
            alpha=self.config.priority_exponent,
            beta=self.config.importance_sampling_exponent,
            n_step=self.config.adder_n_step,
            discount=self.config.discount,
        )
        self.replay_buffer.device = self.device

    def _init_training_state(self):
        """Initialize training state variables."""
        self.training_step = 0
        self.episode_count = 0
        self.steps_since_growth = 0
        self.growth_history = [self.action_discretizer.num_bins]
        self.base_temperature = 1.0
        self.min_temperature = 0.01
        self.temperature_decay = 0.9995

    def get_current_action_mask(self):
        """Get action mask for current resolution level with caching."""
        current_bins = self.action_discretizer.num_bins
        max_bins = self.config.max_bins

        if not hasattr(self, "_cached_mask") or self._cached_bins != current_bins:
            self._cached_mask = self._create_action_mask(current_bins, max_bins)
            self._cached_bins = current_bins

        return self._cached_mask

    def _create_action_mask(self, current_bins, max_bins):
        """Create action mask for given bin configuration."""
        if self.config.decouple:
            mask = torch.zeros(
                self.action_discretizer.action_dim,
                max_bins,
                dtype=torch.bool,
                device=self.device,
            )
            for dim in range(self.action_discretizer.action_dim):
                mask[dim, :current_bins] = True
        else:
            total_actions = current_bins ** self.action_discretizer.action_dim
            mask = torch.zeros(
                max_bins ** self.action_discretizer.action_dim,
                dtype=torch.bool,
                device=self.device,
            )
            mask[:total_actions] = True
        return mask

    def get_temperature(self):
        """Compute current softmax temperature with reset after growth."""
        steps_for_decay = self.steps_since_growth
        temp = self.base_temperature * (self.temperature_decay ** steps_for_decay)
        return max(temp, self.min_temperature)

    def select_action(self, obs, evaluate=False):
        """Select action using softmax policy with adaptive temperature."""
        with torch.no_grad():
            if self.encoder:
                obs_encoded = self.encoder(obs.unsqueeze(0))
            else:
                obs_encoded = obs.unsqueeze(0)

            action_mask = self.get_current_action_mask()
            q1, q2 = self.q_network(obs_encoded, action_mask)
            q_values = torch.min(q1, q2)

            if evaluate:
                return self._greedy_action_selection(q_values)

            return self._softmax_action_selection(q_values)

    def _greedy_action_selection(self, q_values):
        """Select greedy action for evaluation."""
        if self.config.decouple:
            actions = q_values.argmax(dim=2).squeeze(0)
            return self.action_discretizer.action_bins[
                torch.arange(self.action_discretizer.action_dim, device=actions.device), actions
            ]
        else:
            action_idx = q_values.argmax(dim=1).squeeze(0).item()
            return self.action_discretizer.action_bins[action_idx]

    def _softmax_action_selection(self, q_values):
        """Select action using softmax with current temperature."""
        temperature = self.get_temperature()

        if self.config.decouple:
            probs = F.softmax(q_values / temperature, dim=2)
            actions = torch.multinomial(probs.squeeze(0), 1).squeeze(1)
            return self.action_discretizer.action_bins[
                torch.arange(self.action_discretizer.action_dim, device=actions.device), actions
            ]
        else:
            probs = F.softmax(q_values / temperature, dim=1)
            action_idx = torch.multinomial(probs.squeeze(0), 1).squeeze(0).item()
            return self.action_discretizer.action_bins[action_idx]

    def observe_first(self, obs):
        """Store first observation of episode."""
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Add transition to replay buffer."""
        if hasattr(self, "last_obs"):
            discrete_action = self._to_discrete_action(action)
            self.replay_buffer.add(
                self.last_obs, discrete_action, reward, next_obs, done
            )
        self.last_obs = next_obs.detach() if isinstance(next_obs, torch.Tensor) else next_obs

    def _to_discrete_action(self, action):
        """Convert continuous action to discrete if needed."""
        if isinstance(action, torch.Tensor):
            return continuous_to_discrete_action(
                self.config, self.action_discretizer, action
            )
        return action

    def maybe_grow_action_space(self, episode_return):
        """Check and perform action space growth with temperature reset."""
        if not self._should_check_growth():
            return False

        if self.scheduler.should_grow(self.episode_count, episode_return):
            return self._perform_growth()

        return False

    def _should_check_growth(self):
        """Determine if growth should be checked."""
        return (
                self.episode_count > 200
                and len(self.replay_buffer) > self.config.min_replay_size * 2
        )

    def _perform_growth(self):
        """Perform action space growth with temperature reset."""
        old_bins = self.action_discretizer.num_bins
        growth_occurred = self.action_discretizer.grow_action_space()

        if growth_occurred:
            current_bins = self.action_discretizer.num_bins
            self.growth_history.append(current_bins)
            self.steps_since_growth = 0

            self._update_target_network()
            self._clear_mask_cache()

            self.logger.info(
                f"Episode {self.episode_count}: Grew from {old_bins} to {current_bins} bins"
            )
            self.logger.info(f"Temperature reset to {self.base_temperature}")
            return True

        return False

    def _update_target_network(self):
        """Update target network immediately after growth."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _clear_mask_cache(self):
        """Clear cached action mask."""
        if hasattr(self, "_cached_mask"):
            delattr(self, "_cached_mask")

    def update(self):
        """Update networks using min double Q-learning."""
        batch = check_and_sample_batch_from_replay_buffer(
            self.replay_buffer, self.config.min_replay_size, self.config.num_bins
        )
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = (
            get_batch_components(batch, self.device)
        )

        obs_encoded, next_obs_encoded = encode_observation(self.encoder, obs, next_obs)
        action_mask = self.get_current_action_mask()

        q1_current, q2_current = self.q_network(obs_encoded, action_mask)
        targets = self._compute_targets(
            next_obs_encoded, rewards, dones, discounts, action_mask
        )

        q1_selected, q2_selected = self._select_q_values(q1_current, q2_current, actions)
        loss, td_errors = self._compute_loss(
            q1_selected, q2_selected, targets, weights
        )

        self._optimize(loss)
        self._update_priorities(indices, td_errors)
        self._maybe_update_target()

        self.training_step += 1
        self.steps_since_growth += 1

        return self._compute_metrics(loss, td_errors, q1_selected, q2_selected)

    def _compute_targets(self, next_obs_encoded, rewards, dones, discounts, action_mask):
        """Compute target Q-values using min of double Q-networks."""
        with torch.no_grad():
            q1_target, q2_target = self.target_q_network(next_obs_encoded, action_mask)
            q1_online, q2_online = self.q_network(next_obs_encoded, action_mask)

            q_min = torch.min(q1_online, q2_online)

            if self.config.decouple:
                return self._compute_decoupled_targets(
                    q1_target, q2_target, q_min, rewards, dones, discounts
                )
            else:
                return self._compute_coupled_targets(
                    q1_target, q2_target, q_min, rewards, dones, discounts
                )

    def _compute_decoupled_targets(self, q1_target, q2_target, q_min, rewards, dones, discounts):
        """Compute targets for decoupled action space using min double Q."""
        next_actions = q_min.argmax(dim=2)

        batch_indices = torch.arange(q1_target.shape[0], device=self.device)
        dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

        q1_selected = q1_target[
            batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
        ]
        q2_selected = q2_target[
            batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
        ]

        q_target_values = torch.min(q1_selected, q2_selected).sum(dim=1) / self.action_discretizer.action_dim
        return rewards + discounts * q_target_values * (~dones).float()

    def _compute_coupled_targets(self, q1_target, q2_target, q_min, rewards, dones, discounts):
        """Compute targets for coupled action space using min double Q."""
        next_actions = q_min.argmax(dim=1)

        q1_values = q1_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        q2_values = q2_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        q_target_values = torch.min(q1_values, q2_values)
        return rewards + discounts * q_target_values * ~dones

    def _select_q_values(self, q1_current, q2_current, actions):
        """Select Q-values for taken actions."""
        if self.config.decouple:
            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

            q1_per_dim = q1_current[
                batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions
            ]
            q2_per_dim = q2_current[
                batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions
            ]

            q1_selected = q1_per_dim.sum(dim=1) / self.action_discretizer.action_dim
            q2_selected = q2_per_dim.sum(dim=1) / self.action_discretizer.action_dim
        else:
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        return q1_selected, q2_selected

    def _compute_loss(self, q1_selected, q2_selected, targets, weights):
        """Compute weighted Huber loss for both Q-networks."""
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        huber_loss = nn.SmoothL1Loss(reduction='none', beta=self.config.huber_loss_parameter)

        loss1 = huber_loss(q1_selected, targets)
        loss2 = huber_loss(q2_selected, targets)

        weighted_loss1 = (loss1 * weights).mean()
        weighted_loss2 = (loss2 * weights).mean()

        total_loss = weighted_loss1 + weighted_loss2

        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        td_errors = 0.5 * (torch.abs(td_error1) + torch.abs(td_error2))

        return total_loss, td_errors

    def _optimize(self, loss):
        """Perform optimization step with gradient clipping."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        if getattr(self.config, "clip_gradients", False):
            clip_norm = getattr(self.config, "clip_gradients_norm", 40.0)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), clip_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

    def _update_priorities(self, indices, td_errors):
        """Update replay buffer priorities based on TD errors."""
        priorities = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, priorities)

    def _maybe_update_target(self):
        """Periodically update target network."""
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _compute_metrics(self, loss, td_errors, q1_selected, q2_selected):
        """Compute and return training metrics."""
        return {
            "loss": loss.item(),
            "mean_abs_td_error": td_errors.mean().item(),
            "mean_squared_td_error": (td_errors ** 2).mean().item(),
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item(),
            "current_bins": self.action_discretizer.num_bins,
            "temperature": self.get_temperature(),
        }

    def end_episode(self, episode_return):
        """Handle end of episode and potential growth."""
        self.episode_count += 1
        self.maybe_grow_action_space(episode_return)

    @property
    def epsilon(self):
        """Compatibility property for logging. Returns current temperature instead."""
        return self.get_temperature()

    def update_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Compatibility method for train.py. Temperature decays automatically per step."""
        pass

    def get_growth_info(self):
        """Get information about current growth state."""
        return {
            "current_bins": self.action_discretizer.num_bins,
            "growth_history": self.growth_history,
            "max_bins": self.config.max_bins,
            "temperature": self.get_temperature(),
        }

    def save_checkpoint(self, path, episode):
        """Save agent checkpoint including GQN-specific state."""
        checkpoint = {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "growth_history": self.growth_history,
            "steps_since_growth": self.steps_since_growth,
            "action_discretizer_current_bins": self.action_discretizer.num_bins,
            "action_discretizer_current_growth_idx": self.action_discretizer.current_growth_idx,
            "replay_buffer_buffer": self.replay_buffer.buffer,
            "replay_buffer_position": self.replay_buffer.position,
            "replay_buffer_priorities": self.replay_buffer.priorities,
            "replay_buffer_max_priority": self.replay_buffer.max_priority,
            "scheduler_returns_history": list(self.scheduler.returns_history),
            "scheduler_last_growth_episode": self.scheduler.last_growth_episode,
        }

        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore state."""
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

            self._restore_training_state(checkpoint)
            self._restore_growth_state(checkpoint)
            self._restore_replay_buffer(checkpoint)
            self._restore_scheduler(checkpoint)
            self._restore_encoder(checkpoint)

            self._clear_mask_cache()
            self._log_checkpoint_info(checkpoint)

            return checkpoint["episode"]
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0

    def _restore_training_state(self, checkpoint):
        """Restore training state from checkpoint."""
        self.training_step = checkpoint.get("training_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.steps_since_growth = checkpoint.get("steps_since_growth", 0)

    def _restore_growth_state(self, checkpoint):
        """Restore growth state from checkpoint."""
        if "growth_history" in checkpoint:
            self.growth_history = checkpoint["growth_history"]

        if "action_discretizer_current_bins" in checkpoint:
            self.action_discretizer.num_bins = checkpoint["action_discretizer_current_bins"]

        if "action_discretizer_current_growth_idx" in checkpoint:
            self.action_discretizer.current_growth_idx = checkpoint[
                "action_discretizer_current_growth_idx"
            ]
            self.action_discretizer.action_bins = self.action_discretizer.all_action_bins[
                self.action_discretizer.num_bins
            ]

    def _restore_replay_buffer(self, checkpoint):
        """Restore replay buffer from checkpoint."""
        if "replay_buffer_buffer" in checkpoint:
            self.replay_buffer.buffer = checkpoint["replay_buffer_buffer"]
            self.replay_buffer.position = checkpoint["replay_buffer_position"]
            self.replay_buffer.priorities = checkpoint["replay_buffer_priorities"]
            self.replay_buffer.max_priority = checkpoint["replay_buffer_max_priority"]
            self.replay_buffer.to_device(self.device)

    def _restore_scheduler(self, checkpoint):
        """Restore scheduler state from checkpoint."""
        if "scheduler_returns_history" in checkpoint:
            from collections import deque
            self.scheduler.returns_history = deque(
                checkpoint["scheduler_returns_history"],
                maxlen=self.scheduler.window_size,
            )
            self.scheduler.last_growth_episode = checkpoint.get(
                "scheduler_last_growth_episode", 0
            )

    def _restore_encoder(self, checkpoint):
        """Restore encoder state from checkpoint if exists."""
        if self.encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )

    def _log_checkpoint_info(self, checkpoint):
        """Log checkpoint loading information."""
        self.logger.info(f"Loaded checkpoint from episode {checkpoint['episode']}")
        self.logger.info(f"Current resolution: {self.action_discretizer.num_bins} bins")
        self.logger.info(f"Growth history: {self.growth_history}")