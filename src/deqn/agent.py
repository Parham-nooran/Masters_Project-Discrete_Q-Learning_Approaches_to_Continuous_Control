import torch.optim as optim

from src.common.actors import CustomDiscreteFeedForwardActor
from src.common.encoder import *
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import *
from src.deqn.critic import *
from src.gqn.discretizer import GrowingActionDiscretizer


class DecQNAgent:
    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec
        self.action_discretizer = GrowingActionDiscretizer(
            action_spec, config.num_bins, config.decouple
        )
        if config.use_pixels:
            self.encoder = VisionEncoder(config, config.num_pixels).to(self.device)
            encoder_output_size = config.layer_size_bottleneck
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=config.learning_rate
            )
        else:
            self.encoder = None
            encoder_output_size = np.prod(obs_shape)
        self.q_network = CriticDQN(config, encoder_output_size, action_spec).to(
            self.device
        )
        self.target_q_network = CriticDQN(config, encoder_output_size, action_spec).to(
            self.device
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
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
        self.eval_actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=0.0,
            decouple=config.decouple,
        )
        self.training_step = 0
        self.epsilon = config.epsilon

    def update_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        self.actor.epsilon = self.epsilon

    def select_action(self, obs, evaluate=False):
        if evaluate:
            return self.eval_actor.select_action(obs)
        else:
            return self.actor.select_action(obs)

    def observe_first(self, obs):
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Observe transition and store in replay buffer."""
        if hasattr(self, "last_obs"):
            if isinstance(action, torch.Tensor):
                discrete_action = continuous_to_discrete_action(
                    self.config, self.action_discretizer, action
                )
            else:
                discrete_action = action

            self.store_transition(
                self.last_obs, discrete_action, reward, next_obs, done
            )
        if isinstance(next_obs, torch.Tensor):
            self.last_obs = next_obs.detach()
        else:
            self.last_obs = next_obs

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        # Convert discrete action back if needed
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def save_checkpoint(self, path, episode):
        """Save agent checkpoints."""
        checkpoint = {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "config.py": self.config,
            "training_step": self.training_step,
        }
        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = (
                self.encoder_optimizer.state_dict()
            )
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load agent checkpoints."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "replay_buffer_buffer" in checkpoint:
            self.replay_buffer.buffer = checkpoint["replay_buffer_buffer"]
            self.replay_buffer.position = checkpoint["replay_buffer_position"]
            self.replay_buffer.priorities = checkpoint["replay_buffer_priorities"]
            self.replay_buffer.max_priority = checkpoint["replay_buffer_max_priority"]
            self.replay_buffer.to_device(self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)

        if self.encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
            self.actor.epsilon = self.epsilon
        if "config_dict" in checkpoint:
            config_dict = checkpoint["config_dict"]
            for key, value in config_dict.items():
                setattr(self.config, key, value)
        return checkpoint["episode"]

    def make_epsilon_greedy_policy(self, q_values, epsilon, decouple=False):
        """Create epsilon-greedy policy from Q-values."""
        batch_size = q_values[0].shape[0]

        if isinstance(q_values, tuple):
            q_max = torch.max(q_values[0], q_values[1])
        else:
            q_max = q_values

        if decouple:

            num_dims = q_max.shape[1]
            num_bins = q_max.shape[2]

            return get_combined_random_and_greedy_actions(
                q_max, num_dims, num_bins, batch_size, epsilon, self.device
            )
        else:
            if torch.rand(1).item() < epsilon:
                return torch.randint(
                    0, q_max.shape[1], (batch_size,), device=self.device
                )
            else:
                return q_max.argmax(dim=1)

    def update(self):
        batch = check_and_sample_batch_from_replay_buffer(
            self.replay_buffer, self.config.min_replay_size, self.config.num_bins
        )
        if batch is None:
            return {}
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = (
            get_batch_components(batch, self.device)
        )
        obs_encoded, next_obs_encoded = encode_observation(self.encoder, obs, next_obs)

        q1_current, q2_current = self.q_network(obs_encoded)
        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded)
            q1_next_online, q2_next_online = self.q_network(next_obs_encoded)

            if self.config.decouple:
                q1_next_online_reshaped = q1_next_online.view(
                    q1_next_online.shape[0], self.action_discretizer.action_dim, -1
                )
                q2_next_online_reshaped = q2_next_online.view(
                    q2_next_online.shape[0], self.action_discretizer.action_dim, -1
                )
                q_next_online = 0.5 * (
                    q1_next_online_reshaped + q2_next_online_reshaped
                )
                next_actions = q_next_online.argmax(dim=2)
                q1_target_reshaped = q1_next_target.view(
                    q1_next_target.shape[0], self.action_discretizer.action_dim, -1
                )
                q2_target_reshaped = q2_next_target.view(
                    q2_next_target.shape[0], self.action_discretizer.action_dim, -1
                )
                batch_indices = torch.arange(
                    q1_target_reshaped.shape[0], device=self.device
                )
                dim_indices = torch.arange(
                    self.action_discretizer.action_dim, device=self.device
                )

                q1_selected = q1_target_reshaped[
                    batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
                ]
                q2_selected = q2_target_reshaped[
                    batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), next_actions
                ]
                q_target_per_dim = 0.5 * (q1_selected + q2_selected)
                q_target_values = (
                    q_target_per_dim.sum(dim=1) / self.action_discretizer.action_dim
                )
                targets = rewards + discounts * q_target_values * (~dones).float()
            else:
                next_actions = self.make_epsilon_greedy_policy(
                    (q1_next_online, q2_next_online), 0.0, False
                )
                q_next_target = 0.5 * (q1_next_target + q2_next_target)
                q_target_values = q_next_target.gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                targets = rewards + discounts * q_target_values * ~dones

        if self.config.decouple:
            # For decoupled case: implement Equation 2 from paper
            # Q(st, at) = (Σⱼ Qⱼ(st, aⱼₜ)) / M

            q1_reshaped = q1_current.view(
                q1_current.shape[0], self.action_discretizer.action_dim, -1
            )
            q2_reshaped = q2_current.view(
                q2_current.shape[0], self.action_discretizer.action_dim, -1
            )

            batch_indices = torch.arange(q1_reshaped.shape[0], device=self.device)
            dim_indices = torch.arange(
                self.action_discretizer.action_dim, device=self.device
            )

            q1_per_dim = q1_reshaped[
                batch_indices.unsqueeze(1), dim_indices.unsqueeze(0), actions
            ]
            q2_per_dim = q2_reshaped[
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
        mse_loss1 = (td_error1**2).mean()
        mse_loss2 = (
            (td_error2**2).mean()
            if self.config.decouple
            else torch.zeros_like(mse_loss1)
        )

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
            "mse_loss1": mse_loss1.item(),
            "mse_loss2": mse_loss2.item() if self.config.use_double_q else 0,
        }
