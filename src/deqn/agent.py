import torch
import torch.optim as optim

from src.common.actors import CustomDiscreteFeedForwardActor
from src.common.encoder import VisionEncoder
from src.common.replay_buffer import PrioritizedReplayBuffer
from src.common.training_utils import (
    continuous_to_discrete_action,
    get_batch_components,
    encode_observation,
    calculate_losses,
    check_and_sample_batch_from_replay_buffer,
)
from src.deqn.critic import CriticDQN
from src.deqn.discretizer import Discretizer


class DecQNAgent:
    """
    Decoupled Q-Networks agent for continuous control.

    Implements the DecQN algorithm from "Solving Continuous Control via Q-learning"
    using value decomposition and bang-bang action discretization.
    """

    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = obs_shape
        self.action_spec = action_spec

        self.action_discretizer = Discretizer(
            action_spec, config.num_bins, config.decouple
        )

        encoder_output_size = self._setup_encoder()
        self._setup_networks(encoder_output_size)
        self._setup_replay_buffer()
        self._setup_actors()

        self.training_step = 0
        self.epsilon = config.epsilon

    def _setup_encoder(self):
        """Initialize encoder for pixel observations."""
        if self.config.use_pixels:
            self.encoder = VisionEncoder(
                self.config, self.config.num_pixels
            ).to(self.device)
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(), lr=self.config.learning_rate
            )
            return self.config.layer_size_bottleneck

        self.encoder = None
        return int(torch.prod(torch.tensor(self.obs_shape)))

    def _setup_networks(self, encoder_output_size):
        """Initialize Q-networks."""
        self.q_network = CriticDQN(
            self.config, encoder_output_size, self.action_spec
        ).to(self.device)

        self.target_q_network = CriticDQN(
            self.config, encoder_output_size, self.action_spec
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.q_optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

    def _setup_replay_buffer(self):
        """Initialize prioritized replay buffer."""
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.max_replay_size,
            alpha=self.config.priority_exponent,
            beta=self.config.importance_sampling_exponent,
            n_step=self.config.adder_n_step,
            discount=self.config.discount,
        )
        self.replay_buffer.device = self.device

    def _setup_actors(self):
        """Initialize training and evaluation actors."""
        self.actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=self.config.epsilon,
            decouple=self.config.decouple,
        )

        self.eval_actor = CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=0.0,
            decouple=self.config.decouple,
        )

    def update_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay epsilon for exploration."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
        self.actor.epsilon = self.epsilon

    def select_action(self, obs, evaluate=False):
        """Select action using epsilon-greedy policy."""
        actor = self.eval_actor if evaluate else self.actor
        return actor.select_action(obs)

    def observe_first(self, obs):
        """Initialize episode observation."""
        self.last_obs = obs

    def observe(self, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        if not hasattr(self, "last_obs"):
            return

        discrete_action = self._get_discrete_action(action)
        self.replay_buffer.add(
            self.last_obs, discrete_action, reward, next_obs, done
        )

        self.last_obs = next_obs.detach() if isinstance(next_obs, torch.Tensor) else next_obs

    def _get_discrete_action(self, action):
        """Convert action to discrete if necessary."""
        if isinstance(action, torch.Tensor):
            return continuous_to_discrete_action(
                self.config, self.action_discretizer, action
            )
        return action

    def update(self):
        """
        Perform one update step on the Q-networks.

        Implements Equation 1 from the paper with decoupled Q-values.
        """
        batch = check_and_sample_batch_from_replay_buffer(
            self.replay_buffer, self.config.min_replay_size, self.config.num_bins
        )
        if batch is None:
            return {}

        obs, actions, rewards, next_obs, dones, discounts, weights, indices = \
            get_batch_components(batch, self.device)

        obs_encoded, next_obs_encoded = encode_observation(
            self.encoder, obs, next_obs
        )

        q1_selected, q2_selected, targets = self._compute_td_targets(
            obs_encoded, next_obs_encoded, actions, rewards, dones, discounts
        )

        td_error1, td_error2, total_loss = self._compute_loss(
            q1_selected, q2_selected, targets, weights
        )

        self._update_networks()
        self._update_replay_priorities(indices, td_error1, td_error2)
        self._update_target_network()

        return self._get_metrics(total_loss, td_error1, q1_selected, q2_selected)

    def _compute_td_targets(self, obs_encoded, next_obs_encoded, actions,
                            rewards, dones, discounts):
        """Compute TD targets using double Q-learning."""
        q1_current, q2_current = self.q_network(obs_encoded)

        with torch.no_grad():
            q1_next_target, q2_next_target = self.target_q_network(next_obs_encoded)
            q1_next_online, q2_next_online = self.q_network(next_obs_encoded)

            if self.config.decouple:
                targets = self._compute_decoupled_targets(
                    q1_next_online, q2_next_online,
                    q1_next_target, q2_next_target,
                    rewards, dones, discounts
                )
            else:
                targets = self._compute_coupled_targets(
                    q1_next_online, q2_next_online,
                    q1_next_target, q2_next_target,
                    rewards, dones, discounts
                )

        q1_selected, q2_selected = self._select_q_values(
            q1_current, q2_current, actions
        )

        return q1_selected, q2_selected, targets

    def _compute_decoupled_targets(self, q1_online, q2_online,
                                   q1_target, q2_target,
                                   rewards, dones, discounts):
        """
        Compute decoupled targets following Equation 2 and 4 from paper.

        Q(s,a) = (1/M) * sum_i Q_i(s, a_i)
        """
        batch_size = q1_online.shape[0]
        action_dim = self.action_discretizer.action_dim

        q1_reshaped = q1_online.view(batch_size, action_dim, -1)
        q2_reshaped = q2_online.view(batch_size, action_dim, -1)
        q_avg = 0.5 * (q1_reshaped + q2_reshaped)

        next_actions = q_avg.argmax(dim=2)

        q1_target_reshaped = q1_target.view(batch_size, action_dim, -1)
        q2_target_reshaped = q2_target.view(batch_size, action_dim, -1)

        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        dim_idx = torch.arange(action_dim, device=self.device).unsqueeze(0)

        q1_selected = q1_target_reshaped[batch_idx, dim_idx, next_actions]
        q2_selected = q2_target_reshaped[batch_idx, dim_idx, next_actions]

        q_target_values = 0.5 * (q1_selected + q2_selected).sum(dim=1) / action_dim

        return rewards + discounts * q_target_values * (~dones).float()

    def _compute_coupled_targets(self, q1_online, q2_online,
                                 q1_target, q2_target,
                                 rewards, dones, discounts):
        """Compute coupled targets for non-decoupled mode."""
        q_avg = 0.5 * (q1_online + q2_online)
        next_actions = q_avg.argmax(dim=1)

        q_target = 0.5 * (q1_target + q2_target)
        q_target_values = q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        return rewards + discounts * q_target_values * ~dones

    def _select_q_values(self, q1, q2, actions):
        """Select Q-values for taken actions."""
        if self.config.decouple:
            return self._select_decoupled_q_values(q1, q2, actions)
        return self._select_coupled_q_values(q1, q2, actions)

    def _select_decoupled_q_values(self, q1, q2, actions):
        """Select Q-values for decoupled actions following Equation 2."""
        batch_size = q1.shape[0]
        action_dim = self.action_discretizer.action_dim

        q1_reshaped = q1.view(batch_size, action_dim, -1)
        q2_reshaped = q2.view(batch_size, action_dim, -1)

        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        dim_idx = torch.arange(action_dim, device=self.device).unsqueeze(0)

        q1_per_dim = q1_reshaped[batch_idx, dim_idx, actions]
        q2_per_dim = q2_reshaped[batch_idx, dim_idx, actions]

        q1_selected = q1_per_dim.sum(dim=1) / action_dim
        q2_selected = q2_per_dim.sum(dim=1) / action_dim

        return q1_selected, q2_selected

    def _select_coupled_q_values(self, q1, q2, actions):
        """Select Q-values for coupled actions."""
        if actions.ndim > 1:
            actions = actions.squeeze(-1)

        q1_selected = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_selected = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        return q1_selected, q2_selected

    def _compute_loss(self, q1_selected, q2_selected, targets, weights):
        """Compute Huber loss for Q-networks."""
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected

        total_loss = calculate_losses(
            td_error1, td_error2,
            self.config.use_double_q,
            self.q_optimizer,
            self.encoder,
            self.encoder_optimizer if self.encoder else None,
            weights,
            self.config.huber_loss_parameter,
        )

        return td_error1, td_error2, total_loss

    def _update_networks(self):
        """Apply gradients and update networks."""
        if self.config.clip_gradients:
            clip_norm = self.config.clip_gradients_norm
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), clip_norm)
            if self.encoder:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_norm)

        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

    def _update_replay_priorities(self, indices, td_error1, td_error2):
        """Update priorities in replay buffer."""
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()

        priorities = (
            0.5 * (priorities1 + priorities2)
            if self.config.use_double_q
            else priorities1
        )

        self.replay_buffer.update_priorities(indices, priorities)

    def _update_target_network(self):
        """Periodically update target network."""
        self.training_step += 1
        if self.training_step % self.config.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _get_metrics(self, total_loss, td_error1, q1_selected, q2_selected):
        """Collect training metrics."""
        return {
            "loss": total_loss.item(),
            "mean_abs_td_error": torch.abs(td_error1).mean().item(),
            "mean_squared_td_error": (td_error1 ** 2).mean().item(),
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item() if self.config.use_double_q else 0,
            "mse_loss1": (td_error1 ** 2).mean().item(),
            "mse_loss2": (td_error1 ** 2).mean().item() if self.config.use_double_q else 0,
        }

    def save_checkpoint(self, path, episode):
        """Save agent checkpoint."""
        checkpoint = {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
        }

        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = \
                self.encoder_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(
            checkpoint["target_q_network_state_dict"]
        )
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon)
        self.actor.epsilon = self.epsilon

        if self.encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )

        return checkpoint["episode"]