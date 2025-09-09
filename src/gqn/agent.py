import torch
import torch.optim as optim
import numpy as np

from src.deqn.actors import CustomDiscreteFeedForwardActor
from src.deqn.critic import CriticDQN  # Reuse DecQN critic
from src.common.encoder import VisionEncoder
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.agent_utils import huber_loss, continuous_to_discrete_action


class GrowingQNAgent:
    """Growing Q-Networks Agent - minimal implementation using DecQN as base."""

    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec

        # Growing action discretizer
        from src.gqn.discretizer import GrowingActionDiscretizer
        self.action_discretizer = GrowingActionDiscretizer(
            action_spec,
            max_bins=config.max_bins,
            decouple=config.decouple
        )

        # Growing scheduler
        from src.gqn.scheduler import GrowingScheduler
        self.scheduler = GrowingScheduler(config.num_episodes)

        # Setup encoder (same as DecQN)
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=config.learning_rate
            )
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)

        # Reuse DecQN critic but with max_bins for output dimension
        # Modify config temporarily to use max_bins
        original_num_bins = config.max_bins
        config.num_bins = config.max_bins  # Use max_bins for network size

        self.q_network = CriticDQN(config, encoder_output_size, action_spec).to(self.device)
        self.target_q_network = CriticDQN(config, encoder_output_size, action_spec).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Restore original config
        config.num_bins = original_num_bins

        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )

        # Replay buffer (same as DecQN)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.max_replay_size,
            alpha=config.priority_exponent,
            beta=config.importance_sampling_exponent,
            n_step=config.adder_n_step,
            discount=config.discount,
        )
        self.replay_buffer.device = self.device

        # Actor (same as DecQN)
        self.actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=config.epsilon,
            decouple=config.decouple,
        )

        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.epsilon = config.epsilon
        self.growth_history = [self.action_discretizer.current_bins]

    def get_current_action_mask(self):
        """Get action mask for current resolution level - simplified approach."""
        current_bins = self.action_discretizer.current_bins
        max_bins = self.config.max_bins

        if self.config.decouple:
            mask = torch.zeros(self.action_discretizer.action_dim, max_bins,
                               dtype=torch.bool, device=self.device)


            for dim in range(self.action_discretizer.action_dim):
                mask[dim, :current_bins] = True
            return mask
        else:
            total_actions = current_bins ** self.action_discretizer.action_dim
            mask = torch.zeros(max_bins ** self.action_discretizer.action_dim,
                               dtype=torch.bool, device=self.device)
            mask[:total_actions] = True
            return mask

    def select_action(self, obs, evaluate=False):
        """Select action - same as DecQN but with action masking."""
        return self.actor.select_action(obs)

    def observe_first(self, obs):
        """Same as DecQN."""
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Same as DecQN."""
        if hasattr(self, "last_obs"):
            if isinstance(action, torch.Tensor):
                discrete_action = continuous_to_discrete_action(self.config, self.action_discretizer, action)
            else:
                discrete_action = action

            self.replay_buffer.add(self.last_obs, discrete_action, reward, next_obs, done)

        self.last_obs = next_obs.detach() if isinstance(next_obs, torch.Tensor) else next_obs

    def maybe_grow_action_space(self, episode_return):
        """Check if action space should grow."""
        if self.episode_count > 100:  # Allow some initial learning
            if self.scheduler.should_grow(self.episode_count, episode_return):
                growth_occurred = self.action_discretizer.grow_action_space()
                if growth_occurred:
                    current_bins = self.action_discretizer.current_bins
                    self.growth_history.append(current_bins)
                    print(f"Episode {self.episode_count}: Growing to {current_bins} bins")
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

        # Move to device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        discounts = discounts.to(self.device)
        weights = weights.to(self.device)

        # Encode observations
        if self.encoder:
            obs_encoded = self.encoder(obs)
            with torch.no_grad():
                next_obs_encoded = self.encoder(next_obs)
        else:
            obs_encoded = obs.flatten(1)
            next_obs_encoded = next_obs.flatten(1)

        # Get Q-values
        q1_current, q2_current = self.q_network(obs_encoded)

        # Apply action masking
        action_mask = self.get_current_action_mask()
        mask_value = -1e6

        if self.config.decouple:
            # Mask per-dimension Q-values
            q1_current = torch.where(action_mask.unsqueeze(0), q1_current,
                                     torch.full_like(q1_current, mask_value))
            q2_current = torch.where(action_mask.unsqueeze(0), q2_current,
                                     torch.full_like(q2_current, mask_value))

        # Target computation (same as DecQN with masking)
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded)
            q1_next_online, q2_next_online = self.q_network(next_obs_encoded)

            # Apply masking to next Q-values too
            if self.config.decouple:
                q1_next_target = torch.where(action_mask.unsqueeze(0), q1_next_target,
                                             torch.full_like(q1_next_target, mask_value))
                q2_next_target = torch.where(action_mask.unsqueeze(0), q2_next_target,
                                             torch.full_like(q2_next_target, mask_value))
                q1_next_online = torch.where(action_mask.unsqueeze(0), q1_next_online,
                                             torch.full_like(q1_next_online, mask_value))
                q2_next_online = torch.where(action_mask.unsqueeze(0), q2_next_online,
                                             torch.full_like(q2_next_online, mask_value))

            # Double DQN target computation (same logic as DecQN)
            if self.config.decouple:
                q_next_online = 0.5 * (q1_next_online + q2_next_online)
                next_actions = q_next_online.argmax(dim=2)

                batch_indices = torch.arange(q1_next_target.shape[0], device=self.device)
                dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

                q1_selected = q1_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]
                q2_selected = q2_next_target[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions]

                q_target_per_dim = 0.5 * (q1_selected + q2_selected)
                q_target_values = q_target_per_dim.sum(dim=1) / self.action_discretizer.action_dim

                targets = rewards + discounts * q_target_values * (~dones).float()
            else:
                next_actions = (0.5 * (q1_next_online + q2_next_online)).argmax(dim=1)
                q_next_target = 0.5 * (q1_next_target + q2_next_target)
                q_target_values = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones

        # Current Q-values for selected actions (same as DecQN)
        if self.config.decouple:
            batch_indices = torch.arange(q1_current.shape[0], device=self.device)
            dim_indices = torch.arange(self.action_discretizer.action_dim, device=self.device)

            q1_per_dim = q1_current[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions]
            q2_per_dim = q2_current[batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions]

            q1_selected = q1_per_dim.sum(dim=1) / self.action_discretizer.action_dim
            q2_selected = q2_per_dim.sum(dim=1) / self.action_discretizer.action_dim
        else:
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)
            q1_selected = q1_current.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_selected = q2_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss computation (same as DecQN)
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        loss1 = huber_loss(td_error1, getattr(self.config, "huber_loss_parameter", 1.0))
        loss2 = huber_loss(td_error2, getattr(self.config, "huber_loss_parameter",
                                              1.0)) if self.config.use_double_q else torch.zeros_like(loss1)

        loss1 = (loss1 * weights).mean()
        loss2 = (loss2 * weights).mean() if self.config.use_double_q else torch.zeros_like(loss1)

        total_loss = loss1 + loss2

        # Optimization (same as DecQN)
        self.q_optimizer.zero_grad()
        if self.encoder:
            self.encoder_optimizer.zero_grad()

        total_loss.backward()

        if getattr(self.config, "clip_gradients", False):
            clip_norm = getattr(self.config, "clip_gradients_norm", 40.0)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), clip_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

        # Update priorities (same as DecQN)
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()
        priorities = 0.5 * (priorities1 + priorities2) if self.config.use_double_q else priorities1
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": total_loss.item(),
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item() if self.config.use_double_q else 0,
            "current_bins": self.action_discretizer.current_bins,
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
            "current_bins": self.action_discretizer.current_bins,
            "growth_history": self.growth_history,
            "max_bins": self.config.max_bins,
        }