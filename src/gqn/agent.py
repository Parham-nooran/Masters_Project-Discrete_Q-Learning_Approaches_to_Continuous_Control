import os
import numpy as np
import torch
import torch.optim as optim

from src.common.actors import CustomDiscreteFeedForwardActor
from src.common.encoder import VisionEncoder
from src.common.logger import Logger
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import (
    continuous_to_discrete_action,
    get_batch_components,
    encode_observation,
    calculate_losses,
    check_and_sample_batch_from_replay_buffer,
)
from src.gqn.critic import GrowingQCritic
from src.gqn.discretizer import GrowingActionDiscretizer
from src.gqn.scheduler import GrowingScheduler


class GrowingQNAgent(Logger):
    """Growing Q-Networks Agent - minimal implementation using DecQN as base."""

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
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=config.learning_rate
            )
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)
        original_num_bins = config.max_bins
        config.num_bins = config.max_bins
        self.q_network = GrowingQCritic(config, encoder_output_size, action_spec).to(
            self.device
        )
        self.target_q_network = GrowingQCritic(
            config, encoder_output_size, action_spec
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        config.num_bins = original_num_bins
        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount,
        )
        self.replay_buffer.device = self.device
        self.actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=config.epsilon,
            decouple=config.decouple,
        )
        self.training_step = 0
        self.episode_count = 0
        self.epsilon = config.epsilon
        self.growth_history = [self.action_discretizer.num_bins]

    def get_current_action_mask(self):
        """Get action mask for current resolution level."""
        current_bins = self.action_discretizer.num_bins
        max_bins = self.config.max_bins

        if not hasattr(self, "_cached_mask") or self._cached_bins != current_bins:
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
                total_actions = current_bins**self.action_discretizer.action_dim
                mask = torch.zeros(
                    max_bins**self.action_discretizer.action_dim,
                    dtype=torch.bool,
                    device=self.device,
                )
                mask[:total_actions] = True

            self._cached_mask = mask
            self._cached_bins = current_bins

        return self._cached_mask

    def select_action(self, obs):
        """Select action - same as DecQN but with action masking."""
        return self.actor.select_action(obs)

    def observe_first(self, obs):
        """Same as DecQN."""
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Same as DecQN."""
        if hasattr(self, "last_obs"):
            if isinstance(action, torch.Tensor):
                discrete_action = continuous_to_discrete_action(
                    self.config, self.action_discretizer, action
                )
            else:
                discrete_action = action

            self.replay_buffer.add(
                self.last_obs, discrete_action, reward, next_obs, done
            )

        self.last_obs = (
            next_obs.detach() if isinstance(next_obs, torch.Tensor) else next_obs
        )

    def maybe_grow_action_space(self, episode_return):
        """Check if action space should grow."""
        if (
            self.episode_count > 200
            and len(self.replay_buffer) > self.config.min_replay_size * 2
        ):
            if self.scheduler.should_grow(self.episode_count, episode_return):
                old_bins = self.action_discretizer.num_bins
                growth_occurred = self.action_discretizer.grow_action_space()
                if growth_occurred:
                    current_bins = self.action_discretizer.num_bins
                    self.growth_history.append(current_bins)
                    self.logger.info(
                        f"Episode {self.episode_count}: Grew from {old_bins} to {current_bins} bins"
                    )
                    if hasattr(self, "_cached_mask"):
                        delattr(self, "_cached_mask")
                    return True
        return False

    def update(self):
        """Update networks - mostly same as DecQN with action masking."""
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
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(
                next_obs_encoded, action_mask
            )
            q1_next_online, q2_next_online = self.q_network(
                next_obs_encoded, action_mask
            )
            if self.config.decouple:
                q_next_online = 0.5 * (q1_next_online + q2_next_online)
                next_actions = q_next_online.argmax(dim=2)
                batch_indices = torch.arange(
                    q1_next_target.shape[0], device=self.device
                )
                dim_indices = torch.arange(
                    self.action_discretizer.action_dim, device=self.device
                )
                q1_selected = q1_next_target[
                    batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
                ]
                q2_selected = q2_next_target[
                    batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
                ]
                q_target_per_dim = 0.5 * (q1_selected + q2_selected)
                q_target_values = (
                    q_target_per_dim.sum(dim=1) / self.action_discretizer.action_dim
                )
                targets = rewards + discounts * q_target_values * (~dones).float()
            else:
                next_actions = (0.5 * (q1_next_online + q2_next_online)).argmax(dim=1)
                q_next_target = 0.5 * (q1_next_target + q2_next_target)
                q_target_values = q_next_target.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones
        if self.config.decouple:
            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            dim_indices = torch.arange(
                self.action_discretizer.action_dim, device=self.device
            )
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

        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        total_loss = calculate_losses(
            td_error1,
            td_error2,
            self.config.use_double_q,
            self.q_optimizer,
            self.encoder,
            self.encoder_optimizer if self.encoder else None,
            weights,
            self.config.huber_loss_parameter,
        )
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

        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()
        priorities = (
            0.5 * (priorities1 + priorities2)
            if self.config.use_double_q
            else priorities1
        )
        self.replay_buffer.update_priorities(indices, priorities)

        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        mean_abs_td_error = torch.abs(td_error1).mean().item()
        mean_squared_td_error = (td_error1**2).mean().item()
        return {
            "loss": total_loss.item(),
            "mean_abs_td_error": mean_abs_td_error,
            "mean_squared_td_error": mean_squared_td_error,
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item() if self.config.use_double_q else 0,
            "current_bins": self.action_discretizer.num_bins,
        }

    def end_episode(self, episode_return):
        """Handle end of episode and potential growth."""
        self.episode_count += 1
        self.maybe_grow_action_space(episode_return)

    def update_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Same as DecQN."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        self.actor.epsilon = self.epsilon

    def get_growth_info(self):
        """Get information about current growth state."""
        return {
            "current_bins": self.action_discretizer.num_bins,
            "growth_history": self.growth_history,
            "max_bins": self.config.max_bins,
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
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "growth_history": self.growth_history,
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
            checkpoint["encoder_optimizer_state_dict"] = (
                self.encoder_optimizer.state_dict()
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore state. Returns episode number."""
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            self.training_step = checkpoint.get("training_step", 0)
            self.epsilon = checkpoint.get("epsilon", self.config.epsilon)
            self.episode_count = checkpoint.get("episode_count", 0)

            if "growth_history" in checkpoint:
                self.growth_history = checkpoint["growth_history"]

            if "action_discretizer_current_bins" in checkpoint:
                self.action_discretizer.num_bins = checkpoint[
                    "action_discretizer_current_bins"
                ]

            if "action_discretizer_current_growth_idx" in checkpoint:
                self.action_discretizer.current_growth_idx = checkpoint[
                    "action_discretizer_current_growth_idx"
                ]
                self.action_discretizer.action_bins = self.action_discretizer.all_action_bins[
                    self.action_discretizer.num_bins
                ]

            if "replay_buffer_buffer" in checkpoint:
                self.replay_buffer.buffer = checkpoint["replay_buffer_buffer"]
                self.replay_buffer.position = checkpoint["replay_buffer_position"]
                self.replay_buffer.priorities = checkpoint["replay_buffer_priorities"]
                self.replay_buffer.max_priority = checkpoint["replay_buffer_max_priority"]
                self.replay_buffer.to_device(self.device)

            if "scheduler_returns_history" in checkpoint:
                from collections import deque
                self.scheduler.returns_history = deque(
                    checkpoint["scheduler_returns_history"],
                    maxlen=self.scheduler.window_size,
                )
                self.scheduler.last_growth_episode = checkpoint.get(
                    "scheduler_last_growth_episode", 0
                )

            if self.encoder and "encoder_state_dict" in checkpoint:
                self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                self.encoder_optimizer.load_state_dict(
                    checkpoint["encoder_optimizer_state_dict"]
                )

            if hasattr(self, "_cached_mask"):
                delattr(self, "_cached_mask")

            self.logger.info(f"Loaded checkpoint from episode {checkpoint['episode']}")
            self.logger.info(f"Current resolution: {self.action_discretizer.num_bins} bins")
            self.logger.info(f"Growth history: {self.growth_history}")

            return checkpoint["episode"]
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
