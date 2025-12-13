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


def _compute_average_q_values(q1, q2):
    """Compute average of two Q-value tensors."""
    return 0.5 * (q1 + q2)


def _select_argmax_actions(q_values):
    """Select actions with maximum Q-values."""
    return q_values.argmax(dim=1)


def _gather_q_values(q_values, actions):
    """Gather Q-values for specified actions."""
    return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


def _compute_target_values(q_target_values, rewards, discounts, dones):
    """Compute TD target values with reward and discount."""
    return rewards + discounts * q_target_values * ~dones


def _compute_coupled_targets(
    q1_online, q2_online, q1_target, q2_target, rewards, dones, discounts
):
    """Compute coupled targets for non-decoupled mode."""
    q_avg = _compute_average_q_values(q1_online, q2_online)
    next_actions = _select_argmax_actions(q_avg)

    q_target = _compute_average_q_values(q1_target, q2_target)
    q_target_values = _gather_q_values(q_target, next_actions)

    return _compute_target_values(q_target_values, rewards, discounts, dones)


def _ensure_1d_actions(actions):
    """Ensure actions are 1-dimensional."""
    if actions.ndim > 1:
        return actions.squeeze(-1)
    return actions


def _select_coupled_q_values(q1, q2, actions):
    """Select Q-values for coupled actions."""
    actions = _ensure_1d_actions(actions)
    q1_selected = _gather_q_values(q1, actions)
    q2_selected = _gather_q_values(q2, actions)
    return q1_selected, q2_selected


class DecQNAgent:
    """
    Decoupled Q-Networks agent for continuous control.

    Implements the DecQN algorithm from "Solving Continuous Control via Q-learning"
    using value decomposition and bang-bang action discretization.
    """

    def __init__(self, config, obs_shape, action_spec):
        self.config = config
        self.device = self._initialize_device()
        self.obs_shape = obs_shape
        self.action_spec = action_spec

        self.action_discretizer = self._create_discretizer()
        encoder_output_size = self._setup_encoder()

        self._setup_networks(encoder_output_size)
        self._setup_replay_buffer()
        self._setup_actors()

        self.training_step = 0
        self.epsilon = config.epsilon

    def _initialize_device(self):
        """Initialize computation device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_discretizer(self):
        """Create action discretizer."""
        return Discretizer(self.action_spec, self.config.num_bins, self.config.decouple)

    def _calculate_state_input_size(self):
        """Calculate input size for state observations."""
        return int(torch.prod(torch.tensor(self.obs_shape)))

    def _setup_encoder(self):
        """Initialize encoder for pixel observations."""
        if not self.config.use_pixels:
            self.encoder = None
            return self._calculate_state_input_size()

        self.encoder = self._create_vision_encoder()
        self.encoder_optimizer = self._create_encoder_optimizer()
        return self.config.layer_size_bottleneck

    def _create_vision_encoder(self):
        """Create and initialize vision encoder."""
        encoder = VisionEncoder(self.config, self.config.num_pixels)
        return encoder.to(self.device)

    def _create_encoder_optimizer(self):
        """Create optimizer for encoder."""
        return optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)

    def _setup_networks(self, encoder_output_size):
        """Initialize Q-networks."""
        self.q_network = self._create_q_network(encoder_output_size)
        self.target_q_network = self._create_target_network(encoder_output_size)
        self.q_optimizer = self._create_q_optimizer()

    def _create_q_network(self, encoder_output_size):
        """Create online Q-network."""
        network = CriticDQN(self.config, encoder_output_size, self.action_spec)
        return network.to(self.device)

    def _create_target_network(self, encoder_output_size):
        """Create target Q-network with copied weights."""
        network = CriticDQN(self.config, encoder_output_size, self.action_spec)
        network.load_state_dict(self.q_network.state_dict())
        return network.to(self.device)

    def _create_q_optimizer(self):
        """Create optimizer for Q-network."""
        return optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)

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
        self.actor = self._create_actor(epsilon=self.config.epsilon)
        self.eval_actor = self._create_actor(epsilon=0.0)

    def _create_actor(self, epsilon):
        """Create actor with specified exploration rate."""
        return CustomDiscreteFeedForwardActor(
            policy_network=self.q_network,
            encoder=self.encoder,
            action_discretizer=self.action_discretizer,
            epsilon=epsilon,
            decouple=self.config.decouple,
        )

    def _decay_epsilon(self, decay_rate, min_epsilon):
        """Apply decay to epsilon value."""
        return max(min_epsilon, self.epsilon * decay_rate)

    def update_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay epsilon for exploration."""
        self.epsilon = self._decay_epsilon(decay_rate, min_epsilon)
        self.actor.epsilon = self.epsilon

    def select_action(self, obs, evaluate=False):
        """Select action using epsilon-greedy policy."""
        actor = self.eval_actor if evaluate else self.actor
        return actor.select_action(obs)

    def observe_first(self, obs):
        """Initialize episode observation."""
        self.last_obs = obs

    def _convert_to_discrete_action(self, action):
        """Convert continuous action to discrete representation."""
        return continuous_to_discrete_action(
            self.config, self.action_discretizer, action
        )

    def _is_tensor_action(self, action):
        """Check if action is a tensor."""
        return isinstance(action, torch.Tensor)

    def _get_discrete_action(self, action):
        """Convert action to discrete if necessary."""
        if self._is_tensor_action(action):
            return self._convert_to_discrete_action(action)
        return action

    def _detach_observation(self, obs):
        """Detach observation from computation graph."""
        if isinstance(obs, torch.Tensor):
            return obs.detach()
        return obs

    def observe(self, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        if not hasattr(self, "last_obs"):
            return

        discrete_action = self._get_discrete_action(action)
        self.replay_buffer.add(self.last_obs, discrete_action, reward, next_obs, done)
        self.last_obs = self._detach_observation(next_obs)

    def _sample_batch(self):
        """Sample batch from replay buffer."""
        return check_and_sample_batch_from_replay_buffer(
            self.replay_buffer, self.config.min_replay_size, self.config.num_bins
        )

    def _unpack_batch(self, batch):
        """Unpack batch into components."""
        return get_batch_components(batch, self.device)

    def _encode_observations(self, obs, next_obs):
        """Encode observations using encoder."""
        return encode_observation(self.encoder, obs, next_obs)

    def update(self):
        """
        Perform one update step on the Q-networks.

        Implements Equation 1 from the paper with decoupled Q-values.
        """
        batch = self._sample_batch()
        if batch is None:
            return {}

        components = self._unpack_batch(batch)
        obs, actions, rewards, next_obs, dones, discounts, weights, indices = components

        obs_encoded, next_obs_encoded = self._encode_observations(obs, next_obs)

        q1_selected, q2_selected, targets = self._compute_td_targets(
            obs_encoded, next_obs_encoded, actions, rewards, dones, discounts
        )

        td_error1, td_error2, total_loss = self._compute_loss(
            q1_selected, q2_selected, targets, weights
        )

        self._update_networks()
        self._update_replay_priorities(indices, td_error1, td_error2)
        self._update_target_network()

        return self._get_metrics(
            total_loss, td_error1, td_error2, q1_selected, q2_selected
        )

    def _get_current_q_values(self, obs_encoded):
        """Get Q-values from online network."""
        return self.q_network(obs_encoded)

    def _get_next_q_values(self, next_obs_encoded):
        """Get Q-values from both networks for next state."""
        with torch.no_grad():
            q1_target, q2_target = self.target_q_network(next_obs_encoded)
            q1_online, q2_online = self.q_network(next_obs_encoded)
        return q1_online, q2_online, q1_target, q2_target

    def _calculate_targets(
        self, q1_online, q2_online, q1_target, q2_target, rewards, dones, discounts
    ):
        """Calculate TD targets based on decoupling mode."""
        if self.config.decouple:
            return self._compute_decoupled_targets(
                q1_online, q2_online, q1_target, q2_target, rewards, dones, discounts
            )
        return _compute_coupled_targets(
            q1_online, q2_online, q1_target, q2_target, rewards, dones, discounts
        )

    def _compute_td_targets(
        self, obs_encoded, next_obs_encoded, actions, rewards, dones, discounts
    ):
        """Compute TD targets using double Q-learning."""
        q1_current, q2_current = self._get_current_q_values(obs_encoded)

        q1_online, q2_online, q1_target, q2_target = self._get_next_q_values(
            next_obs_encoded
        )

        targets = self._calculate_targets(
            q1_online, q2_online, q1_target, q2_target, rewards, dones, discounts
        )

        q1_selected, q2_selected = self._select_q_values(
            q1_current, q2_current, actions
        )

        return q1_selected, q2_selected, targets

    def _reshape_q_values(self, q_values, batch_size):
        """Reshape Q-values for decoupled mode."""
        action_dim = self.action_discretizer.action_dim
        return q_values.view(batch_size, action_dim, -1)

    def _select_best_actions(self, q1_reshaped, q2_reshaped):
        """Select best actions from averaged Q-values."""
        q_avg = _compute_average_q_values(q1_reshaped, q2_reshaped)
        return q_avg.argmax(dim=2)

    def _create_batch_indices(self, batch_size):
        """Create batch indices for indexing."""
        return torch.arange(batch_size, device=self.device).unsqueeze(1)

    def _create_dimension_indices(self):
        """Create dimension indices for indexing."""
        action_dim = self.action_discretizer.action_dim
        return torch.arange(action_dim, device=self.device).unsqueeze(0)

    def _gather_target_values(
        self, q_target_reshaped, batch_idx, dim_idx, next_actions
    ):
        """Gather target Q-values using advanced indexing."""
        return q_target_reshaped[batch_idx, dim_idx, next_actions]

    def _average_q_values_across_dimensions(self, q1_selected, q2_selected):
        """Average Q-values across action dimensions."""
        action_dim = self.action_discretizer.action_dim
        q_avg = _compute_average_q_values(q1_selected, q2_selected)
        return q_avg.sum(dim=1) / action_dim

    def _compute_decoupled_targets(
        self, q1_online, q2_online, q1_target, q2_target, rewards, dones, discounts
    ):
        """
        Compute decoupled targets following Equation 2 and 4 from paper.

        Q(s,a) = (1/M) * sum_i Q_i(s, a_i)
        """
        batch_size = q1_online.shape[0]

        q1_reshaped = self._reshape_q_values(q1_online, batch_size)
        q2_reshaped = self._reshape_q_values(q2_online, batch_size)
        next_actions = self._select_best_actions(q1_reshaped, q2_reshaped)

        q1_target_reshaped = self._reshape_q_values(q1_target, batch_size)
        q2_target_reshaped = self._reshape_q_values(q2_target, batch_size)

        batch_idx = self._create_batch_indices(batch_size)
        dim_idx = self._create_dimension_indices()

        q1_selected = self._gather_target_values(
            q1_target_reshaped, batch_idx, dim_idx, next_actions
        )
        q2_selected = self._gather_target_values(
            q2_target_reshaped, batch_idx, dim_idx, next_actions
        )

        q_target_values = self._average_q_values_across_dimensions(
            q1_selected, q2_selected
        )

        return _compute_target_values(q_target_values, rewards, discounts, dones)

    def _select_q_values(self, q1, q2, actions):
        """Select Q-values for taken actions."""
        if self.config.decouple:
            return self._select_decoupled_q_values(q1, q2, actions)
        return _select_coupled_q_values(q1, q2, actions)

    def _gather_per_dimension_q_values(self, q_reshaped, batch_idx, dim_idx, actions):
        """Gather Q-values for each action dimension."""
        return q_reshaped[batch_idx, dim_idx, actions]

    def _select_decoupled_q_values(self, q1, q2, actions):
        """Select Q-values for decoupled actions following Equation 2."""
        batch_size = q1.shape[0]

        q1_reshaped = self._reshape_q_values(q1, batch_size)
        q2_reshaped = self._reshape_q_values(q2, batch_size)

        batch_idx = self._create_batch_indices(batch_size)
        dim_idx = self._create_dimension_indices()

        q1_per_dim = self._gather_per_dimension_q_values(
            q1_reshaped, batch_idx, dim_idx, actions
        )
        q2_per_dim = self._gather_per_dimension_q_values(
            q2_reshaped, batch_idx, dim_idx, actions
        )

        action_dim = self.action_discretizer.action_dim
        q1_selected = q1_per_dim.sum(dim=1) / action_dim
        q2_selected = q2_per_dim.sum(dim=1) / action_dim

        return q1_selected, q2_selected

    def _calculate_td_errors(self, targets, q1_selected, q2_selected):
        """Calculate temporal difference errors."""
        td_error1 = targets - q1_selected
        td_error2 = targets - q2_selected
        return td_error1, td_error2

    def _compute_loss(self, q1_selected, q2_selected, targets, weights):
        """Compute Huber loss for Q-networks."""
        td_error1, td_error2 = self._calculate_td_errors(
            targets, q1_selected, q2_selected
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

        return td_error1, td_error2, total_loss

    def _clip_network_gradients(self, network):
        """Clip gradients for a network."""
        clip_norm = self.config.clip_gradients_norm
        torch.nn.utils.clip_grad_norm_(network.parameters(), clip_norm)

    def _apply_gradient_clipping(self):
        """Apply gradient clipping to all networks."""
        if not self.config.clip_gradients:
            return

        self._clip_network_gradients(self.q_network)
        if self.encoder:
            self._clip_network_gradients(self.encoder)

    def _step_optimizers(self):
        """Perform optimization step for all optimizers."""
        self.q_optimizer.step()
        if self.encoder:
            self.encoder_optimizer.step()

    def _update_networks(self):
        """Apply gradients and update networks."""
        self._apply_gradient_clipping()
        self._step_optimizers()

    def _compute_priorities(self, td_error1, td_error2):
        """Compute priorities from TD errors."""
        priorities1 = torch.abs(td_error1).detach().cpu().numpy()
        priorities2 = torch.abs(td_error2).detach().cpu().numpy()

        if self.config.use_double_q:
            return 0.5 * (priorities1 + priorities2)
        return priorities1

    def _update_replay_priorities(self, indices, td_error1, td_error2):
        """Update priorities in replay buffer."""
        priorities = self._compute_priorities(td_error1, td_error2)
        self.replay_buffer.update_priorities(indices, priorities)

    def _should_update_target(self):
        """Check if target network should be updated."""
        return self.training_step % self.config.target_update_period == 0

    def _sync_target_network(self):
        """Synchronize target network with online network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _update_target_network(self):
        """Periodically update target network."""
        self.training_step += 1
        if self._should_update_target():
            self._sync_target_network()

    def _get_metrics(self, total_loss, td_error1, td_error2, q1_selected, q2_selected):
        """Collect training metrics."""
        return {
            "loss": total_loss.item(),
            "mean_abs_td_error": torch.abs(td_error1).mean().item(),
            "mean_squared_td_error": (td_error1**2).mean().item(),
            "q1_mean": q1_selected.mean().item(),
            "q2_mean": q2_selected.mean().item() if self.config.use_double_q else 0,
            "mse_loss1": (td_error1**2).mean().item(),
            "mse_loss2": (
                (td_error2**2).mean().item() if self.config.use_double_q else 0
            ),
        }

    def _create_checkpoint_dict(self, episode):
        """Create checkpoint dictionary with all necessary state."""
        return {
            "episode": episode,
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
        }

    def _add_encoder_to_checkpoint(self, checkpoint):
        """Add encoder state to checkpoint if encoder exists."""
        if self.encoder:
            checkpoint["encoder_state_dict"] = self.encoder.state_dict()
            checkpoint["encoder_optimizer_state_dict"] = (
                self.encoder_optimizer.state_dict()
            )

    def save_checkpoint(self, path, episode):
        """Save agent checkpoint."""
        checkpoint = self._create_checkpoint_dict(episode)
        self._add_encoder_to_checkpoint(checkpoint)
        torch.save(checkpoint, path)

    def _load_network_states(self, checkpoint):
        """Load network states from checkpoint."""
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

    def _load_training_state(self, checkpoint):
        """Load training state from checkpoint."""
        self.training_step = checkpoint.get("training_step", 0)
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon)
        self.actor.epsilon = self.epsilon

    def _load_encoder_state(self, checkpoint):
        """Load encoder state from checkpoint if available."""
        if self.encoder and "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.encoder_optimizer.load_state_dict(
                checkpoint["encoder_optimizer_state_dict"]
            )

    def load_checkpoint(self, path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._load_network_states(checkpoint)
        self._load_training_state(checkpoint)
        self._load_encoder_state(checkpoint)

        return checkpoint["episode"]
