import torch
import torch.optim as optim
import numpy as np

from src.common.actors import CustomDiscreteFeedForwardActor
from src.gqn.critic import GrowingQCritic
from src.common.encoder import VisionEncoder
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.utils import huber_loss, continuous_to_discrete_action
from src.gqn.scheduler import GrowingScheduler
from src.gqn.discretizer import GrowingActionDiscretizer
from src.common.logger import Logger


class GrowingQNAgent(Logger):
    """Growing Q-Networks Agent - minimal implementation using DecQN as base."""

    def __init__(self, config, obs_shape, action_spec, working_dir):
        super().__init__(working_dir)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.action_discretizer = GrowingActionDiscretizer(
            action_spec, max_bins=config.max_bins, decouple=config.decouple
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
        """Get action mask for current resolution level - simplified approach."""
        current_bins = self.action_discretizer.num_bins
        max_bins = self.config.max_bins

        if self.config.decouple:
            mask = torch.zeros(
                self.action_discretizer.action_dim,
                max_bins,
                dtype=torch.bool,
                device=self.device,
            )
            for dim in range(self.action_discretizer.action_dim):
                mask[dim, :current_bins] = True
            return mask
        else:
            total_actions = current_bins**self.action_discretizer.action_dim
            mask = torch.zeros(
                max_bins**self.action_discretizer.action_dim,
                dtype=torch.bool,
                device=self.device,
            )
            mask[:total_actions] = True
            return mask

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
                growth_occurred = self.action_discretizer.grow_action_space()
                if growth_occurred:
                    current_bins = self.action_discretizer.current_bins
                    self.growth_history.append(current_bins)
                    self.logger.info(
                        f"Episode {self.episode_count}: Growing to {current_bins} bins"
                    )
                    return True
        return False

    def update(self):
        """Update networks - mostly same as DecQN with action masking."""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return {}

        batch = self.replay_buffer.sample(self.config.batch_size)
        if batch is None:
            return {}
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = batch
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        discounts = discounts.to(self.device)
        weights = weights.to(self.device)
        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

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

        loss1 = huber_loss(td_error1, getattr(self.config, "huber_loss_parameter", 1.0))
        loss2 = (
            huber_loss(td_error2, getattr(self.config, "huber_loss_parameter", 1.0))
            if self.config.use_double_q
            else torch.zeros_like(loss1)
        )
        loss1 = (loss1 * weights).mean()
        loss2 = (
            (loss2 * weights).mean()
            if self.config.use_double_q
            else torch.zeros_like(loss1)
        )

        total_loss = loss1 + loss2

        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()
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

        # Update target network
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
